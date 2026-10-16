"""Hosted dashboard: login, batch upload, validation, progress, results and CSV download."""

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from analyzer import manifest
from analyzer.batch import DISPLAY_COLUMNS, Row, records, run, summary, write_csv
from analyzer.metrics import score

load_dotenv(Path(__file__).parent / "private" / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

INSTRUCTIONS = """
# Tone and Noise Analyzer
Upload a batch: the audio files plus one CSV manifest with a `name` column (the exact audio filename)
and an optional `result_json` column. Choose a folder or a ZIP archive, then press **Analyze**.
Supported audio: WAV, MP3, OGG/Opus, FLAC, M4A/AAC, WebM. Uploaded audio is deleted after processing.
"""


def analyze_batch(folder_files, zip_file, progress=gr.Progress()):
    uploads = [Path(f) for f in folder_files or []] + ([Path(zip_file)] if zip_file else [])
    workdir = Path(tempfile.mkdtemp(prefix="batch-"))
    try:
        if zip_file:
            try:
                batch = manifest.from_zip(Path(zip_file), workdir)
            except zipfile.BadZipFile:
                raise gr.Error("That file is not a valid ZIP archive.")
        elif folder_files:
            batch = manifest.from_uploads([Path(f) for f in folder_files])
        else:
            raise gr.Error("Upload a folder or a ZIP archive first.")

        report = _report(batch, both=bool(zip_file and folder_files))
        rows: list[Row] = []
        yield report, _table(rows), None, "", gr.skip(), gr.skip()
        total = len(batch.items)
        for i, row in enumerate(run(batch), start=1):
            rows.append(row)
            progress(i / total, desc=f"{i} of {total} files")
            yield report, _table(rows), None, "", gr.skip(), gr.skip()

        csv_path = Path(tempfile.mkdtemp(prefix="results-")) / "predictions.csv"
        write_csv(rows, csv_path)
        scores = score(rows)
        footer = summary(rows) + (f"\n\n{scores.to_markdown()}" if scores else "")
        yield report, _table(rows), str(csv_path), footer, None, None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        for path in uploads:
            path.unlink(missing_ok=True)


def _report(batch: manifest.Batch, both: bool) -> str:
    source = f"`{batch.manifest_name}`" if batch.manifest_name else "none"
    lines = [f"**{len(batch.runnable)} of {len(batch.items)} files will be analyzed.** Manifest: {source}", ""]
    if both:
        lines.append("- Both a folder and a ZIP were uploaded; using the ZIP")
    lines += [f"- {issue}" for issue in batch.issues]
    return "\n".join(lines)


def _table(rows: list[Row]) -> pd.DataFrame:
    return pd.DataFrame(records(rows), columns=DISPLAY_COLUMNS)


with gr.Blocks(title="Tone and Noise Analyzer", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(INSTRUCTIONS)
    with gr.Tab("Folder"):
        folder = gr.File(file_count="directory", label="Batch folder")
    with gr.Tab("ZIP"):
        zipped = gr.File(file_types=[".zip"], label="Batch ZIP")
    analyze_button = gr.Button("Analyze", variant="primary")
    report = gr.Markdown()
    table = gr.Dataframe(headers=DISPLAY_COLUMNS, interactive=False, wrap=True, label="Results")
    download = gr.File(label="Results CSV")
    footer = gr.Markdown()
    analyze_button.click(analyze_batch, [folder, zipped], [report, table, download, footer, folder, zipped])


if __name__ == "__main__":
    user, password = os.environ.get("DASHBOARD_USERNAME"), os.environ.get("DASHBOARD_PASSWORD")
    if not user or not password:
        raise SystemExit("Set DASHBOARD_USERNAME and DASHBOARD_PASSWORD "
                         "(private/.env locally, Space secrets on Hugging Face).")
    demo.queue(default_concurrency_limit=1).launch(
        auth=(user, password),
        auth_message="Sign in with the credentials you were given.",
        ssr_mode=False,
    )
