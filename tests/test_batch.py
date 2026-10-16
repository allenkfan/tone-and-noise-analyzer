import csv
import json

import pytest

from analyzer import manifest
from analyzer.audio import AudioError, load_audio
from analyzer.batch import CSV_COLUMNS, main, run, write_csv
from analyzer.metrics import score
from analyzer.schema import Result
from conftest import write_wav


def test_load_audio_resamples_and_keeps_channels(tmp_path):
    samples = load_audio(write_wav(tmp_path / "stereo.wav", seconds=0.5, sr=8000, channels=2))
    assert samples.dtype == "float32"
    assert samples.shape == (2, 8000)


def test_load_audio_rejects_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not audio")
    with pytest.raises(AudioError):
        load_audio(path)


def test_one_bad_file_does_not_stop_the_batch(batch_dir):
    rows = {row.name: row for row in run(manifest.from_folder(batch_dir))}
    assert rows["good_1.wav"].status == "ok"
    assert rows["good_2.wav"].status == "ok"
    assert rows["unlisted.wav"].status == "ok"
    assert rows["corrupt.wav"].status == "failed" and rows["corrupt.wav"].error
    assert rows["missing.wav"].status == "skipped"
    assert rows["good_1.wav"].audio_seconds == 1.0


def test_csv_keeps_names_and_schema(batch_dir, tmp_path):
    rows = list(run(manifest.from_folder(batch_dir)))
    out = tmp_path / "predictions.csv"
    write_csv(rows, out)
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        records = {r["name"]: r for r in reader}
    Result.model_validate(json.loads(records["good_1.wav"]["result_json"]))
    assert records["good_1.wav"]["background_noise_present"] == "false"
    assert records["missing.wav"]["result_json"] == ""


def test_scores_only_labeled_clips(batch_dir):
    scores = score(list(run(manifest.from_folder(batch_dir))))
    assert scores.n == 1
    assert scores.accuracy["emotional_tone"] == 1.0
    assert "background_noise_type" not in scores.accuracy
    assert scores.tone_macro_f1 == 1.0


def test_cli_writes_predictions(batch_dir, tmp_path, capsys):
    main([str(batch_dir), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "predictions.csv").exists()
    assert "Scored 1 labeled clips" in capsys.readouterr().out
