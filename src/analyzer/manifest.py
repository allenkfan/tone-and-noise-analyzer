"""Turn an uploaded folder or ZIP into a validated batch.

Validation never fails the whole batch. Problems are reported in `Batch.issues`,
and files that can't be processed carry a `problem` instead.
"""

import csv
import io
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from analyzer.audio import SUPPORTED_EXTENSIONS

JUNK_NAMES = {"Thumbs.db", "desktop.ini"}


@dataclass
class Item:
    name: str  # exact filename as listed in the manifest (or as uploaded)
    path: Path | None  # None when the manifest lists a file that wasn't uploaded
    label: dict | None = None  # expected result_json, when the manifest has one
    problem: str = ""  # why the item can't be processed; empty when it can


@dataclass
class Batch:
    items: list[Item]
    manifest_name: str | None
    issues: list[str] = field(default_factory=list)

    @property
    def runnable(self) -> list[Item]:
        return [item for item in self.items if not item.problem]


class ManifestError(Exception):
    """The CSV manifest can't be used at all."""


def from_folder(folder: Path) -> Batch:
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and not _is_junk(p.relative_to(folder).parts)]
    return build(files)


def from_uploads(paths: list[Path]) -> Batch:
    """Files uploaded through the browser; only their names are meaningful."""
    return build([p for p in paths if not _is_junk((p.name,))])


def from_zip(zip_path: Path, dest: Path) -> Batch:
    """Extract `zip_path` into `dest`, flattening folders, and build a batch from it."""
    files = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            if info.is_dir() or not parts or _is_junk(parts):
                continue
            # One directory per member keeps duplicate names apart, and using
            # only the last path part keeps every write inside `dest`.
            target = dest / str(len(files)) / parts[-1]
            target.parent.mkdir(parents=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            files.append(target)
    return build(files)


def build(files: list[Path]) -> Batch:
    issues: list[str] = []
    by_name: dict[str, Path] = {}
    for path in files:
        if path.name in by_name:
            issues.append(f"{path.name}: more than one file has this name; using the first")
        else:
            by_name[path.name] = path

    csvs = sorted(name for name in by_name if name.lower().endswith(".csv"))
    audio = {name: path for name, path in by_name.items() if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS}

    rows: list[tuple[str, dict | None]] | None = None
    manifest_name = csvs[0] if csvs else None
    if len(csvs) > 1:
        issues.append(f"Found {len(csvs)} CSV files; using {manifest_name} as the manifest")
    if manifest_name:
        try:
            rows, row_issues = parse_manifest(by_name[manifest_name].read_text(encoding="utf-8-sig", errors="replace"))
            issues.extend(row_issues)
        except ManifestError as exc:
            issues.append(f"{manifest_name}: {exc}; processing every audio file instead")
            manifest_name = None
    else:
        issues.append("No CSV manifest found; processing every audio file")

    items: list[Item] = []
    listed: set[str] = set()
    for name, label in rows or []:
        key = PurePosixPath(name.replace("\\", "/")).name
        listed.add(key)
        if key in audio:
            items.append(Item(name, audio[key], label))
        elif key in by_name:
            items.append(Item(name, by_name[key], label, problem="unsupported file type"))
            issues.append(f"{name}: unsupported file type")
        else:
            items.append(Item(name, None, label, problem="listed in the manifest but not uploaded"))
            issues.append(f"{name}: listed in the manifest but not uploaded")

    for name in sorted(audio.keys() - listed):
        items.append(Item(name, audio[name]))
        if rows is not None:
            issues.append(f"{name}: not listed in the manifest; processed anyway")

    for name in sorted(by_name.keys() - audio.keys() - listed - set(csvs)):
        issues.append(f"{name}: not an audio file; skipped")

    return Batch(items, manifest_name, issues)


def parse_manifest(text: str) -> tuple[list[tuple[str, dict | None]], list[str]]:
    """Parse the `name` and `result_json` columns. Tolerates spaces after commas, a BOM and extra columns."""
    reader = csv.reader(io.StringIO(text), skipinitialspace=True)
    header = next(reader, None)
    if not header:
        raise ManifestError("the CSV is empty")
    columns = [h.strip().lower() for h in header]
    if "name" not in columns:
        raise ManifestError("the CSV has no 'name' column")
    name_i = columns.index("name")
    label_i = columns.index("result_json") if "result_json" in columns else None

    rows: list[tuple[str, dict | None]] = []
    issues: list[str] = []
    seen: set[str] = set()
    for line_no, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            continue
        name = row[name_i].strip() if name_i < len(row) else ""
        if not name:
            issues.append(f"Manifest line {line_no}: empty name; skipped")
            continue
        if name in seen:
            issues.append(f"{name}: listed more than once in the manifest; using the first row")
            continue
        seen.add(name)
        raw = row[label_i].strip() if label_i is not None and label_i < len(row) else ""
        label = None
        if raw:
            try:
                label = json.loads(raw)
            except ValueError:
                label = None
            if not isinstance(label, dict):
                issues.append(f"{name}: result_json is not a JSON object; label ignored")
                label = None
        rows.append((name, label))
    return rows, issues


def _is_junk(parts: tuple[str, ...]) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in parts) or parts[-1] in JUNK_NAMES
