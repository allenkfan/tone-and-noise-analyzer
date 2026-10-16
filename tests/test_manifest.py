import zipfile
from pathlib import Path

from analyzer import manifest


def test_parses_the_spaced_csv_from_the_brief():
    text = 'name, result_json\ncall_001.wav, " {""emotional_tone"": ""frustrated""}"\ncall_002.mp3,\n'
    rows, issues = manifest.parse_manifest(text)
    assert rows == [("call_001.wav", {"emotional_tone": "frustrated"}), ("call_002.mp3", None)]
    assert issues == []


def test_tolerates_bom_extra_columns_and_bad_json(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text('﻿id,Name,result_json\n1,a.wav,{not json}\n2,a.wav,\n3,,\n', encoding="utf-8")
    rows, issues = manifest.parse_manifest(csv_path.read_text(encoding="utf-8-sig"))
    assert rows == [("a.wav", None)]
    assert any("not a JSON object" in i for i in issues)
    assert any("more than once" in i for i in issues)
    assert any("empty name" in i for i in issues)


def test_folder_validation(batch_dir):
    batch = manifest.from_folder(batch_dir)
    items = {item.name: item for item in batch.items}
    assert batch.manifest_name == "labels.csv"
    assert [i.name for i in batch.items] == ["good_1.wav", "good_2.wav", "corrupt.wav", "missing.wav", "unlisted.wav"]
    assert items["good_1.wav"].label["emotional_tone"] == "neutral"
    assert items["missing.wav"].problem == "listed in the manifest but not uploaded"
    assert {i.name for i in batch.runnable} == {"good_1.wav", "good_2.wav", "corrupt.wav", "unlisted.wav"}
    assert any("unlisted.wav: not listed" in i for i in batch.issues)
    assert any("notes.txt: not an audio file" in i for i in batch.issues)
    assert not any(".DS_Store" in i for i in batch.issues)


def test_missing_manifest_processes_all_audio(batch_dir):
    (batch_dir / "labels.csv").unlink()
    batch = manifest.from_folder(batch_dir)
    assert batch.manifest_name is None
    assert len(batch.runnable) == 4
    assert any("No CSV manifest" in i for i in batch.issues)


def test_manifest_without_name_column_falls_back(batch_dir):
    (batch_dir / "labels.csv").write_text("file,result_json\ngood_1.wav,\n")
    batch = manifest.from_folder(batch_dir)
    assert batch.manifest_name is None
    assert any("no 'name' column" in i for i in batch.issues)


def test_zip_with_top_folder_macos_junk_and_traversal(batch_dir, tmp_path):
    zip_path = tmp_path / "batch.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in batch_dir.iterdir():
            zf.write(path, f"evaluation_batch/{path.name}")
        zf.writestr("__MACOSX/evaluation_batch/._good_1.wav", b"junk")
        zf.writestr("../escape.wav", b"junk")
    dest = tmp_path / "extracted"
    dest.mkdir()
    batch = manifest.from_zip(zip_path, dest)
    assert batch.manifest_name == "labels.csv"
    assert len(batch.items) == 5
    assert all(Path(item.path).is_relative_to(dest) for item in batch.items if item.path)
    assert not (tmp_path / "escape.wav").exists()


def test_uploads_use_file_names_only(batch_dir):
    batch = manifest.from_uploads(sorted(batch_dir.iterdir()))
    assert batch.manifest_name == "labels.csv"
    assert len(batch.items) == 5
