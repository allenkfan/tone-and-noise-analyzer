"""Smoke-test a running dashboard end to end with a synthetic batch (no real calls).

    uv run python scripts/smoke_dashboard.py allen-videolytics/tone-and-noise-analyzer-dev
    uv run python scripts/smoke_dashboard.py http://127.0.0.1:7860

Credentials come from DASHBOARD_USERNAME / DASHBOARD_PASSWORD in the environment or private/.env.
Checks login, ZIP and folder uploads (WAV, Opus/OGG, MP3, a corrupt file, a missing file, a .txt,
the brief's spaced CSV), per-file statuses, the CSV download and the empty-upload error.
Exits 1 if any check fails.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from gradio_client import Client, handle_file

from analyzer.batch import CSV_COLUMNS
from analyzer.schema import Result

EXPECTED = {
    "clip_wav.wav": "ok",
    "clip_opus.ogg": "ok",
    "clip_noise.mp3": "ok",
    "broken.wav": "failed",
    "missing.wav": "skipped",
}
LABEL = json.dumps({
    "emotional_tone": "neutral", "emotional_intensity": "low", "background_noise_present": False,
    "background_noise_type": "", "background_noise_severity": "none", "audio_quality": "clear",
    "speaker_overlap_present": False, "long_silence_present": False, "confidence": 0.5,
})
failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        failures.append(what)


def make_batch(folder: Path) -> None:
    def ffmpeg(*args: str) -> None:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)

    ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=4", "-ar", "16000", str(folder / "clip_wav.wav"))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=500:duration=6", "-ac", "2", "-ar", "48000",
           "-c:a", "libopus", str(folder / "clip_opus.ogg"))
    ffmpeg("-f", "lavfi", "-i", "anoisesrc=d=5:c=pink:a=0.1", "-ar", "22050", "-c:a", "libmp3lame",
           str(folder / "clip_noise.mp3"))
    (folder / "broken.wav").write_bytes(b"not audio")
    (folder / "readme.txt").write_text("notes")
    label = LABEL.replace('"', '""')
    (folder / "labels.csv").write_text(
        "name, result_json\n"
        f'clip_wav.wav, " {label}"\n'
        "clip_opus.ogg,\nclip_noise.mp3,\nbroken.wav,\nmissing.wav,\n"
    )


def check_run(label: str, out) -> None:
    report, _table, csv_file, footer, *_ = out
    check("4 of 5 files will be analyzed" in report, f"{label}: validation report counts files")
    check("missing.wav: listed in the manifest but not uploaded" in report, f"{label}: missing file reported")
    check("readme.txt: not an audio file" in report, f"{label}: non-audio file reported")
    check("Scored 1 labeled clips" in (footer or ""), f"{label}: labeled clip scored")
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        check(reader.fieldnames == CSV_COLUMNS, f"{label}: CSV columns")
        rows = {r["name"]: r for r in reader}
    statuses = {name: rows.get(name, {}).get("status") for name in EXPECTED}
    check(statuses == EXPECTED, f"{label}: per-file statuses {statuses}")
    for name, status in EXPECTED.items():
        if status == "ok" and name in rows:
            try:
                Result.model_validate(json.loads(rows[name]["result_json"]))
                valid = True
            except ValueError:
                valid = False
            check(valid, f"{label}: {name} result_json matches the schema")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    url = target if target.startswith("http") else f"https://{target.replace('/', '-')}.hf.space"
    load_dotenv(Path(__file__).resolve().parent.parent / "private" / ".env")
    user, password = os.environ.get("DASHBOARD_USERNAME"), os.environ.get("DASHBOARD_PASSWORD")
    if not user or not password:
        sys.exit("Set DASHBOARD_USERNAME and DASHBOARD_PASSWORD (environment or private/.env).")
    print(f"Testing {url}")

    status = httpx.get(url + "/gradio_api/info", timeout=60).status_code
    check(status == 401, f"API without login is refused (HTTP {status})")
    try:
        Client(url, auth=(user, "wrong-password"), verbose=False)
        check(False, "wrong password is rejected")
    except ValueError:
        check(True, "wrong password is rejected")
    client = Client(url, auth=(user, password), verbose=False)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "evaluation_batch"
        folder.mkdir()
        make_batch(folder)
        zip_path = Path(tmp) / "evaluation_batch.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in folder.iterdir():
                zf.write(p, f"evaluation_batch/{p.name}")

        check_run("ZIP", client.predict(None, handle_file(str(zip_path)), api_name="/analyze_batch"))
        check_run("Folder", client.predict([handle_file(str(p)) for p in sorted(folder.iterdir())], None,
                                           api_name="/analyze_batch"))
    try:
        client.predict(None, None, api_name="/analyze_batch")
        check(False, "empty submit shows an error")
    except Exception as exc:
        check("Upload a folder or a ZIP archive first" in str(exc), "empty submit shows an error")

    print(f"\n{'FAILED: ' + str(len(failures)) + ' check(s)' if failures else 'All checks passed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
