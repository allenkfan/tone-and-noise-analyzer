"""Run a batch file by file. Shared by the dashboard and the CLI.

    python -m analyzer.batch <folder-or-zip> [--out DIR]
"""

import argparse
import csv
import json
import logging
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from analyzer import manifest
from analyzer.audio import SAMPLE_RATE, AudioError, load_audio
from analyzer.metrics import score
from analyzer.pipeline import analyze
from analyzer.schema import FIELDS, Result

log = logging.getLogger("analyzer")

TIMING_COLUMNS = ["audio_seconds", "processing_seconds"]
DISPLAY_COLUMNS = ["name", "status", "error", *FIELDS, *TIMING_COLUMNS]
CSV_COLUMNS = ["name", "result_json", "status", "error", *FIELDS, *TIMING_COLUMNS]


@dataclass
class Row:
    name: str
    status: str  # ok | failed | skipped
    error: str = ""
    result: Result | None = None
    label: dict | None = None
    audio_seconds: float | None = None
    processing_seconds: float | None = None


def run(batch: manifest.Batch) -> Iterator[Row]:
    """Yield one row per item. A file that fails becomes a failed row; it never stops the batch."""
    for item in batch.items:
        if item.problem:
            row = Row(item.name, "skipped", item.problem, label=item.label)
        else:
            start = time.perf_counter()
            try:
                samples = load_audio(item.path)
                result = analyze(samples, SAMPLE_RATE)
            except Exception as exc:
                row = Row(item.name, "failed", _describe(exc), label=item.label)
            else:
                row = Row(item.name, "ok", result=result, label=item.label,
                          audio_seconds=round(samples.shape[1] / SAMPLE_RATE, 3))
            row.processing_seconds = round(time.perf_counter() - start, 3)
        log.info("%s %s %s %s", row.name, row.status, row.processing_seconds, row.error)
        yield row


def records(rows: Sequence[Row]) -> list[dict]:
    out = []
    for row in rows:
        result = row.result.model_dump() if row.result else {}
        out.append({
            "name": row.name,
            "result_json": json.dumps(result) if result else "",
            "status": row.status,
            "error": row.error,
            **{f: _cell(result.get(f, "")) for f in FIELDS},
            "audio_seconds": row.audio_seconds if row.audio_seconds is not None else "",
            "processing_seconds": row.processing_seconds if row.processing_seconds is not None else "",
        })
    return out


def write_csv(rows: Sequence[Row], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records(rows))


def summary(rows: Sequence[Row]) -> str:
    counts = {s: sum(r.status == s for r in rows) for s in ("ok", "failed", "skipped")}
    audio_min = sum(r.audio_seconds or 0 for r in rows if r.status == "ok") / 60
    proc_s = sum(r.processing_seconds or 0 for r in rows if r.status == "ok")
    text = f"**{counts['ok']} ok · {counts['failed']} failed · {counts['skipped']} skipped**"
    if audio_min:
        text += f" · {audio_min:.2f} audio min in {proc_s:.1f} s ({proc_s / audio_min:.2f} s per audio min)"
    return text


def _describe(exc: Exception) -> str:
    if isinstance(exc, AudioError):
        return str(exc)
    if isinstance(exc, ValidationError):
        return f"analyzer returned an invalid result: {exc.errors()[0]['msg']}"
    return f"{type(exc).__name__}: {exc}"


def _cell(value):
    return str(value).lower() if isinstance(value, bool) else value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m analyzer.batch",
                                     description="Analyze a folder or ZIP of audio files with a CSV manifest.")
    parser.add_argument("source", type=Path, help="batch folder or .zip file")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory (default: out)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        if args.source.suffix.lower() == ".zip":
            batch = manifest.from_zip(args.source, Path(tmp))
        else:
            batch = manifest.from_folder(args.source)
        for issue in batch.issues:
            print(f"! {issue}")
        rows = list(run(batch))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "predictions.csv")
    print(f"\nWrote {args.out / 'predictions.csv'}")
    print(summary(rows))
    if scores := score(rows):
        print("\n" + scores.to_markdown())


if __name__ == "__main__":
    main()
