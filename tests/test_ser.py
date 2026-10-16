"""Speech emotion windows, summaries and fusion, with the WavLM model replaced by a stub."""

import numpy as np
import pytest
import torch

from analyzer import pipeline, ser
from analyzer.ser import SER

SR = ser.SR


class LoudnessModel:
    """Stands in for the WavLM model: arousal grows with the window's RMS."""

    def __call__(self, x):
        rms = x.pow(2).mean(dim=1).sqrt()
        return torch.stack([rms.clamp(0, 1), torch.full_like(rms, 0.5), torch.full_like(rms, 0.4)], dim=1)


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(ser, "_model", lambda: (LoudnessModel(), 0.0, 1.0))


def tone(seconds, amp=0.1):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_windows_cover_the_clip_with_a_2s_hop():
    assert len(ser.windows(tone(10))) == 4  # starts 0, 2, 4, 6
    assert len(ser.windows(tone(11))) == 5  # plus a final window ending at the clip's end
    assert len(ser.windows(tone(2.5))) == 1 and len(ser.windows(tone(0.5))) == 0


def test_silent_windows_are_skipped():
    clip = np.concatenate([tone(4), np.zeros(8 * SR, np.float32)])
    assert len(ser.windows(clip, gate=False)) == 5 and len(ser.windows(clip)) == 2


def test_measure_summarises_windows(stub_model):
    clip = np.concatenate([tone(4, 0.05), tone(4, 0.4)])  # quiet half, loud half
    s = ser.measure(np.stack([clip, clip]), SR)
    assert s.windows == 3 and s.arousal_max > s.arousal_p50 > 0
    assert s.valence_p50 == pytest.approx(0.4) and s.dominance_p50 == pytest.approx(0.5)
    assert set(s.as_context()) == {"windows", "arousal_p50", "arousal_p90", "arousal_max",
                                   "valence_p10", "valence_p50", "dominance_p50"}


def test_measure_returns_none_on_silence(stub_model):
    assert ser.measure(np.zeros(5 * SR, np.float32), SR) is None


def voice(arousal):
    return SER(4, arousal, arousal, arousal, 0.4, 0.5, 0.5)


@pytest.mark.parametrize("tone_in, arousal, expected", [
    ("neutral", 0.50, ("frustrated", "medium", "raised")),
    ("neutral", 0.30, ("neutral", "low", "kept")),
    ("frustrated", 0.80, ("upset", "low", "raised")),
    ("frustrated", 0.50, ("frustrated", "low", "kept")),
    ("upset", 0.20, ("upset", "low", "kept")),
    ("satisfied", 0.90, ("satisfied", "low", "kept")),
])
def test_fuse_table(tone_in, arousal, expected, monkeypatch):
    monkeypatch.setattr(ser, "AROUSAL_RAISE", 0.35)
    monkeypatch.setattr(ser, "AROUSAL_UPSET", 0.65)
    assert ser.fuse(tone_in, "low", voice(arousal)) == expected


def test_fuse_without_voice_keeps_the_llm_answer():
    assert ser.fuse("upset", "high", None) == ("upset", "high", "llm")


def test_ser_goes_to_the_llm_as_context():
    from analyzer import llm

    text = llm._acoustics_text({"ser": voice(0.8).as_context()})
    assert "90th percentile 0.80" in text and "Signal analysis" not in text


def test_pipeline_runs_without_the_model(batch_dir):
    from analyzer.audio import SAMPLE_RATE, load_audio

    result = pipeline.analyze(load_audio(batch_dir / "good_1.wav"), SAMPLE_RATE)  # no key, no SER model
    assert result.emotional_tone == "neutral" and result.confidence == pipeline.FALLBACK_CONFIDENCE


def test_measure_scores_only_the_given_turns(stub_model):
    clip = np.concatenate([tone(6, 0.05), tone(6, 0.5)])  # quiet speaker, then a loud one
    everyone = ser.measure(clip, SR)
    loud = ser.measure(clip, SR, turns=[(6.0, 12.0)])
    assert loud.arousal_p50 > everyone.arousal_p50 and loud.windows == 2
    assert ser.measure(clip, SR, turns=[(1.0, 1.5)]) is None  # shorter than a window
    assert ser.measure(clip, SR, turns=[]) is None


def test_guard_blocks_raises_on_positive_words(monkeypatch):
    monkeypatch.setattr(ser, "AROUSAL_RAISE", 0.35)
    assert ser.fuse("neutral", "low", voice(0.8), "positive") == ("neutral", "low", "guarded")
    assert ser.fuse("neutral", "low", voice(0.8), "positive", guard=False)[0] == "frustrated"
    assert ser.fuse("neutral", "low", voice(0.8), "negative")[0] == "frustrated"
