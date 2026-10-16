# Development

The analyzer is a plain Python package (`src/analyzer`). A thin Gradio app (`app.py`) runs in Docker on a Hugging Face Space and calls it. There are three loops, from fastest to slowest:

```
 1. analyzer loop (seconds)    2. parity check (minutes)    3. deploy (minutes)
 ──────────────────────────    ─────────────────────────    ──────────────────────────────
 edit src/analyzer/*     ──►   docker build + run     ──►   git push space-dev main
 uv run python -m analyzer.batch                            → check the -dev Space
 uv run gradio app.py (UI work)                             git push space main (on purpose)
                                                            → the URL evaluators use
```

Most of the work happens in loop 1. Use loop 2 before each deploy. Loop 3 is the only one that changes anything evaluators can see.

## Prerequisites

| Tool | Why | Install |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Python 3.12, dependencies, lockfile | `brew install uv`, or the release binary from github.com/astral-sh/uv/releases if Homebrew can't build it (Intel macOS 15) |
| ffmpeg | decodes every audio format | `brew install ffmpeg` |
| Docker | parity check against the Space image | Docker Desktop or OrbStack |
| Hugging Face account `allen-videolytics` | hosting | write token from huggingface.co/settings/tokens |

## First-time setup

```sh
git clone https://github.com/allenkfan/tone-and-noise-analyzer.git
cd tone-and-noise-analyzer
git clone https://github.com/allenkfan/tone-and-noise-analyzer-private.git private
uv sync                      # creates .venv with Python 3.12 and all dependencies
```

Set `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD` in `private/.env`. The app won't start without them.

## Layout

```
├── README.md              # Hugging Face Space front matter + overview
├── Dockerfile             # the image the Space runs
├── pyproject.toml, uv.lock, .python-version
├── app.py                 # Gradio UI only: login, upload, progress, results, CSV download
├── src/analyzer/
│   ├── schema.py          # output schema with the exact enum values from the brief
│   ├── audio.py           # ffmpeg decode → float32, 16 kHz, channels kept
│   ├── pipeline.py        # analyze(samples, sr) -> Result   ← model work goes here
│   ├── manifest.py        # folder/ZIP collection, lenient CSV parsing, validation report
│   ├── batch.py           # per-file runner shared by the app and the CLI; CSV writer
│   └── metrics.py         # per-field accuracy, tone macro-F1, confusion matrix
├── tests/                 # synthetic audio only
├── docs/                  # this file; later the technical memo
└── private/               # gitignored clone of the private repo: real calls, brief, .env
```

## Loop 1: developing the analyzer

The pipeline makes this call for each file:

```python
from analyzer.pipeline import analyze
result = analyze(samples, sr)   # samples: float32 array, shape (channels, n), sr = 16000
```

The example calls are 48 kHz stereo Opus, but both channels are identical (L−R is about −90 dB). Channels don't separate customer from agent, so averaging them loses nothing for these files. Speaker separation needs diarization.

`analyze` returns a `schema.Result`. Its field names and enum values come straight from the brief, and Pydantic rejects anything else. Everything else is already wired up: decoding, timing, per-file error handling, CSV output, metrics, and the dashboard. When `analyze` gets better, the CLI and the dashboard get better with it.

Run it on the labeled examples:

```sh
uv run python -m analyzer.batch private/data/example            # → out/predictions.csv + scores
uv run python -m analyzer.batch path/to/folder --out out/run2   # any folder or .zip
```

The CLI prints per-field accuracy, tone macro-F1 and a tone confusion matrix whenever the manifest has labels. It also prints audio minutes and processing time per audio minute, which feed the latency analysis.

Adding dependencies:

```sh
uv add librosa             # updates pyproject.toml and uv.lock; commit both
```

For torch, use the CPU-only wheels. The CUDA build adds gigabytes to the image and the Space has no GPU:

```toml
# pyproject.toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchaudio = { index = "pytorch-cpu" }
```

Model weights: download them inside `Dockerfile` (`RUN python -c "..."`) so the Space never fetches them at startup. Never commit them.

Tests:

```sh
uv run pytest
```

Tests generate their own audio. This repo is public, so real calls never go in `tests/`.

The three labeled calls are a smoke test, not a validation set. Don't tune thresholds to them. Validation needs held-out data that is independent of what you tune on.

## Loop 1b: working on the dashboard

```sh
uv run gradio app.py       # hot reload on save, http://localhost:7860
uv run python app.py       # exactly how the container starts it
```

To test a batch, upload `private/data/example` as a folder or zip it up first. Also try the failure cases: a corrupt file, a CSV row with no audio, an audio file missing from the CSV, a `.txt` file.

## Loop 2: parity check (Docker)

```sh
docker build -t tna .
docker run --rm -p 7860:7860 --env-file private/.env tna
```

This is the same image the Space builds. A Space build takes minutes and its logs are clumsy, so catch failures here. The image installs ffmpeg, runs as UID 1000 and listens on `0.0.0.0:7860`, as Hugging Face expects.

## Loop 3: deploying to Hugging Face Spaces

### One-time setup

1. Create two Spaces at huggingface.co/new-space, owned by `allen-videolytics`, with SDK **Docker** (blank) and visibility **Public**. Evaluators can't reach a private Space, and the app's own login protects the public one.
   - `tone-and-noise-analyzer-dev`: deploy every change here first
   - `tone-and-noise-analyzer`: the URL you submit
2. In each Space, go to **Settings → Variables and secrets** and add the secrets `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD`. They reach the app as environment variables.
3. Add the remotes. When git asks for a password, use the write token:
   ```sh
   git remote add space-dev https://huggingface.co/spaces/allen-videolytics/tone-and-noise-analyzer-dev
   git remote add space     https://huggingface.co/spaces/allen-videolytics/tone-and-noise-analyzer
   ```
4. The first push to each Space needs `--force`, because Hugging Face creates the Space with its own README commit:
   ```sh
   git push --force space-dev main
   ```

The YAML front matter at the top of `README.md` is the Space configuration (`sdk: docker`, `app_port: 7860`). GitHub shows it as a small table.

### Each change

```sh
git push origin main         # GitHub
git push space-dev main      # rebuilds the -dev Space; watch the Logs tab
# run the example batch on the -dev Space, then:
git push space main          # the submission Space
```

Keep these pushes manual. Every push restarts the Space, and a restart in the middle of a batch kills that batch.

### Smoke test against a live Space

```python
from gradio_client import Client, handle_file
client = Client("allen-videolytics/tone-and-noise-analyzer-dev", auth=(user, password))
report, table, csv_path, footer, *_ = client.predict(
    None, handle_file("evaluation_batch.zip"), api_name="/analyze_batch")   # (folder files, zip)
```

The same call works against `http://127.0.0.1:7860` for a local run.

## Before submission and during evaluation

- [ ] Upgrade the submission Space's hardware to paid CPU and set its sleep time to never. Free Spaces pause after about 48 hours idle.
- [ ] Give evaluators the direct URL, https://allen-videolytics-tone-and-noise-analyzer.hf.space. The login cookie fails when the app is embedded in the huggingface.co Space page.
- [ ] Tag the release: `git tag v1.0 && git push origin v1.0`. `uv.lock` pins every dependency.
- [ ] Do not push to `space` during the evaluation window. Keep working on `space-dev`.
- [ ] Run the example batch once against the submission URL and confirm the CSV downloads.

## Rules and gotchas

- **Nothing from `private/` gets committed here.** `.gitignore` blocks `private/`, `.env*` and audio files. Check `git status` before every commit anyway.
- **Binaries.** Hugging Face rejects files over 10 MB unless they go through LFS/Xet. Weights belong in the Docker build, not in git.
- **Privacy.** Uploaded audio is deleted after processing, and Gradio's cache is cleared hourly. Nothing about the audio is logged except file names, statuses and timings. The memo should say so.
- **Logs.** The app logs one line per file (name, status, seconds). Find them in the Space's **Logs** tab.
- **Visibility.** The Space's Files tab shows the source code. That's fine because the GitHub repo is public too, but secrets go only in Space secrets.
