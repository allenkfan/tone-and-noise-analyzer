import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_ser_model(monkeypatch):
    """Keep the 1.3 GB speech-emotion model out of tests; the pipeline then runs without it."""
    from analyzer import ser

    def disabled():
        raise RuntimeError("the SER model is disabled in tests")
    monkeypatch.setattr(ser, "_load", disabled)


def write_wav(path: Path, seconds: float = 1.0, sr: int = 8000, channels: int = 1) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    tone = (0.1 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.repeat(tone[:, None], channels, axis=1).tobytes())
    return path


LABEL = ('{"emotional_tone":"neutral","emotional_intensity":"low","background_noise_present":false,'
         '"background_noise_type":"","background_noise_severity":"none","audio_quality":"clear",'
         '"speaker_overlap_present":false,"long_silence_present":false,"confidence":0.9}')


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    """Two good clips, one corrupt clip, one missing clip, one unlisted clip and junk files."""
    folder = tmp_path / "batch"
    folder.mkdir()
    write_wav(folder / "good_1.wav")
    write_wav(folder / "good_2.wav", channels=2)
    write_wav(folder / "unlisted.wav")
    (folder / "corrupt.wav").write_bytes(b"not audio at all")
    (folder / "notes.txt").write_text("hello")
    (folder / ".DS_Store").write_bytes(b"\0")
    label = LABEL.replace('"', '""')
    (folder / "labels.csv").write_text(
        "name, result_json\n"
        f'good_1.wav, "{label}"\n'
        "good_2.wav,\n"
        "corrupt.wav,\n"
        "missing.wav,\n"
    )
    return folder
