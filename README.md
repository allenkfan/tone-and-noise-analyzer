---
title: Tone and Noise Analyzer
sdk: docker
app_port: 7860
short_description: Emotional tone and background-noise analysis for call audio
---

# Tone and Noise Analyzer

Classifies the emotional tone of call audio and flags background noise, audio-quality problems, speaker overlap and long silences. It returns one structured JSON result per clip.

The dashboard (`app.py`, Gradio) takes a batch as a folder or ZIP of audio files plus a CSV manifest. It validates the batch, processes each file, shows progress and per-file errors, and offers the results as a CSV download. `python -m analyzer.batch` does the same from the command line.

## Quick start

```sh
git clone https://github.com/allenkfan/tone-and-noise-analyzer.git
cd tone-and-noise-analyzer
git clone https://github.com/allenkfan/tone-and-noise-analyzer-private.git private   # data + credentials
uv sync
uv run python -m analyzer.batch private/data/example   # CLI → out/predictions.csv
uv run python app.py                                    # dashboard → http://localhost:7860
```

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg. Audio, labels and credentials live only in `private/` and never belong in this repo.

## Docs

- [docs/dashboard.md](docs/dashboard.md): how to use the hosted dashboard (sign in, batch format, results, troubleshooting)
- [docs/development.md](docs/development.md): the development and deployment workflow
- [docs/technical-memo.md](docs/technical-memo.md): approaches tested, architecture, validation, cost, latency and the external API disclosure

## Third-party models and data

- Acoustic models, run locally through onnxruntime with weights pinned in `src/analyzer/weights.py`: [Silero VAD](https://github.com/snakers4/silero-vad) (MIT), [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) CNN14 16 kHz (MIT) and [DNSMOS](https://github.com/microsoft/DNS-Challenge) P.835 (MIT).
- `src/analyzer/audioset_groups.tsv` is derived from the [AudioSet ontology](https://github.com/audioset/ontology) by Google, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
