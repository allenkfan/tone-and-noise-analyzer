import json

import numpy as np
import pytest

from analyzer import acoustic, llm, pipeline
from analyzer.acoustic import FRAME_S, SR, Acoustics

RNG = np.random.default_rng(0)


def tone(seconds: float, level_db: float = -20.0, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sqrt(2) * 10 ** (level_db / 20) * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def noise(seconds: float, level_db: float) -> np.ndarray:
    return (10 ** (level_db / 20) * RNG.standard_normal(int(seconds * SR))).astype(np.float32)


@pytest.fixture
def fake_vad(monkeypatch):
    """Silero doesn't take synthetic tones for speech, so stand in for it: speech = the
    frames listed in `segments` (seconds)."""
    segments: list[tuple[float, float]] = []

    def probs(x):
        p = np.zeros(-(-len(x) // acoustic.VAD_FRAME), np.float32)
        for a, b in segments:
            p[int(a / FRAME_S) : int(b / FRAME_S)] = 0.95
        return p

    monkeypatch.setattr(acoustic, "speech_probs", probs)
    return segments


def acoustics(**overrides) -> Acoustics:
    base = dict(duration_s=30.0, speech_ratio=0.6, longest_silence_s=2.0, snr_db=40.0, clipping_ratio=0.0,
                bandwidth_hz=4000.0, rms_dbfs=-22.0, noise_tags=[], long_silence_present=False,
                speaker_overlap_present=False, overlap_ratio=0.0, overlap_longest_s=0.0,
                audio_quality="clear", background_noise_present=False, background_noise_severity="none",
                noise_type_guess="", noise_floor_dbfs=-70.0, floor_snr_db=40.0, peak_spread_db=5.0, dropouts_per_min=0.0, mos_sig=3.5,
                mos_bak=4.0, mos_ovr=3.0)
    return Acoustics(**{**base, **overrides})


def test_speech_mask_drops_bursts_and_bridges_short_gaps():
    p = np.zeros(100, np.float32)
    p[10:30] = 0.9  # 640 ms of speech
    p[31:50] = 0.9  # after a 32 ms dip
    p[70:72] = 0.9  # a 64 ms blip
    mask = acoustic.speech_mask(p, pad_s=0)
    assert mask[10:50].all()
    assert not mask[60:].any()


def test_long_gap_without_speech_is_long_silence(fake_vad):
    fake_vad += [(0.5, 5.0), (17.0, 22.0)]
    x = np.concatenate([tone(5), np.zeros(12 * SR, np.float32), tone(5)])
    ac = acoustic.measure(x, SR)
    assert ac.longest_silence_s == pytest.approx(12, abs=0.5)
    assert ac.long_silence_present


def test_ordinary_pauses_are_not_long_silence(fake_vad):
    fake_vad += [(0.0, 5.0), (8.0, 13.0)]
    ac = acoustic.measure(np.concatenate([tone(5), np.zeros(3 * SR, np.float32), tone(5)]), SR)
    assert not ac.long_silence_present


def test_snr_compares_speech_with_the_pauses(fake_vad):
    fake_vad += [(0.0, 4.0), (8.0, 12.0)]
    speech = np.concatenate([tone(4), np.zeros(4 * SR, np.float32), tone(4)])
    ac = acoustic.measure(speech + noise(12, -45), SR)
    assert ac.snr_db == pytest.approx(25, abs=1.5)
    assert ac.noise_floor_dbfs == pytest.approx(-45, abs=1.5)
    assert ac.rms_dbfs == pytest.approx(-20, abs=1)


def test_no_pauses_means_no_snr(fake_vad):
    fake_vad += [(0.0, 10.0)]
    ac = acoustic.measure(tone(10), SR)
    assert ac.snr_db is None and ac.noise_floor_dbfs is None


def test_clipping_ratio_separates_clipped_from_clean():
    clean = (0.05 * RNG.laplace(size=3 * SR) * np.linspace(0.2, 1, 3 * SR)).astype(np.float32)  # speech-like
    clipped = np.clip(clean * 8, -0.7, 0.7)
    assert acoustic.clipping_ratio(clean) < 0.002
    assert acoustic.clipping_ratio(clipped) > 0.03


def test_bandwidth_falls_when_high_frequencies_are_removed():
    wide = noise(4, -20)
    spec = np.fft.rfft(wide)
    spec[np.fft.rfftfreq(len(wide), 1 / SR) > 1500] = 0
    narrow = np.fft.irfft(spec, len(wide)).astype(np.float32)
    frames = lambda x: x[: len(x) // 512 * 512].reshape(-1, 512)  # noqa: E731
    assert acoustic.bandwidth(frames(wide)) > 6000
    assert acoustic.bandwidth(frames(narrow)) < 2000


def test_dropouts_count_zeroed_frames_inside_speech():
    x = tone(10, -20)
    mask = np.ones(-(-len(x) // acoustic.VAD_FRAME), bool)
    assert acoustic.dropouts_per_min(x, mask) == 0
    for k in range(10):
        x[k * SR + 4000 : k * SR + 4320] = 0  # 20 ms holes, once a second
    assert acoustic.dropouts_per_min(x, mask) == pytest.approx(60, abs=6)


def test_dropouts_late_in_a_long_call_line_up_with_the_speech_mask():
    x = tone(180, -20)
    mask = np.zeros(-(-len(x) // acoustic.VAD_FRAME), bool)
    mask[int(149 / FRAME_S) : int(151 / FRAME_S)] = True  # speech only around 150 s
    x[150 * SR : 150 * SR + 320] = 0
    assert acoustic.dropouts_per_min(x, mask) > 0


def test_quality_rules_ignore_noise():
    noisy = acoustics(snr_db=0.0, noise_floor_dbfs=-25.0, noise_tags=[("TV or radio", 0.9)], mos_bak=1.5)
    acoustic.decide(noisy)
    assert noisy.audio_quality == "clear"
    assert noisy.background_noise_present and noisy.background_noise_severity == "high"


def test_noise_rules_ignore_quality():
    broken = acoustics(clipping_ratio=0.2, bandwidth_hz=900.0, rms_dbfs=-55.0, dropouts_per_min=60.0)
    acoustic.decide(broken)
    assert broken.audio_quality == "severely_impaired"
    assert not broken.background_noise_present
    assert broken.background_noise_severity == "none" and broken.noise_type_guess == ""


def test_as_context_is_flat_json():
    ctx = acoustics(noise_tags=[("music", 0.51234)], snr_db=None).as_context()
    assert json.loads(json.dumps(ctx)) == ctx
    assert ctx["noise_tags"] == [["music", 0.512]] and ctx["snr_db"] is None


def test_measure_runs_the_real_models_on_short_and_silent_audio():
    for x in (np.zeros(SR // 5, np.float32), noise(3, -30), np.stack([tone(2), tone(2)])):
        ac = acoustic.measure(x, SR)
        assert 0 <= ac.speech_ratio <= 1
        assert 1 <= ac.mos_sig <= 5.5
        assert all(g in acoustic.GROUPS for g, _ in ac.noise_tags)
        assert 0 <= ac.overlap_ratio <= 1 and not ac.speaker_overlap_present


def test_hiss_is_not_speech_and_is_tagged_as_noise():
    ac = acoustic.measure(noise(12, -25), SR)
    assert ac.speech_ratio < 0.1
    assert ac.noise_tags and ac.noise_tags[0][0] in {"static", "background noise", "hum or buzz"}


def test_groups_cover_the_tagger_labels():
    assert len(acoustic.AUDIOSET_LABELS) == 527
    assert acoustic.NOISE_GROUP[acoustic.AUDIOSET_LABELS.index("Speech")] == ""
    assert acoustic.NOISE_GROUP[acoustic.AUDIOSET_LABELS.index("Dial tone")] == ""
    assert acoustic.NOISE_GROUP[acoustic.AUDIOSET_LABELS.index("Television")] == "TV or radio"
    for pop in ("Static", "Hiss", "Crackle", "Burst, pop", "Gunshot, gunfire", "Firecracker"):
        assert acoustic.NOISE_GROUP[acoustic.AUDIOSET_LABELS.index(pop)] == "static"


def judgement(**overrides) -> llm.Judgement:
    base = dict(emotional_tone="frustrated", emotional_intensity="medium", background_noise_type="television",
                speaker_overlap_present=True, confidence=0.7, rationale="", usage={})
    return llm.Judgement(**{**base, **overrides})


def test_merge_takes_physical_fields_from_acoustics_and_the_rest_from_the_llm():
    ac = acoustics(background_noise_present=True, background_noise_severity="medium",
                   noise_type_guess="TV or radio", audio_quality="slightly_impaired", long_silence_present=True)
    r = pipeline.merge(ac, judgement())
    assert (r.emotional_tone, r.emotional_intensity, r.confidence) == ("frustrated", "medium", 0.7)
    assert (r.background_noise_present, r.background_noise_severity, r.background_noise_type) == \
        (True, "medium", "television")
    assert r.audio_quality == "slightly_impaired" and r.long_silence_present


def test_overlap_comes_from_acoustics_not_the_llm():
    assert pipeline.merge(acoustics(speaker_overlap_present=True), judgement(speaker_overlap_present=False)) \
        .speaker_overlap_present
    assert not pipeline.merge(acoustics(), judgement(speaker_overlap_present=True)).speaker_overlap_present


def test_overlap_needs_a_sustained_stretch():
    brief = acoustics(overlap_ratio=0.02, overlap_longest_s=0.3)  # a backchannel
    sustained = acoustics(overlap_ratio=0.1, overlap_longest_s=2.5)
    acoustic.decide(brief), acoustic.decide(sustained)
    assert not brief.speaker_overlap_present and sustained.speaker_overlap_present


def test_overlap_stats_bridge_flicker_and_measure_the_longest_run():
    speech = np.ones(1000)
    overlap = np.zeros(1000)
    overlap[100:160] = 0.9
    overlap[163:220] = 0.9  # a 3-frame dip inside the same overlap
    overlap[500:510] = 0.9
    ratio, longest = acoustic.overlap_stats(speech, overlap)
    assert longest == pytest.approx(120 * acoustic.SEG_FRAME / SR)
    assert ratio == pytest.approx(130 / 1000)
    speech[:150] = 0  # bridged or not, overlap only counts where someone is talking
    assert acoustic.overlap_stats(speech, overlap)[0] == pytest.approx(80 / 850)


def test_merge_keeps_noise_type_consistent_with_presence():
    assert pipeline.merge(acoustics(), judgement()).background_noise_type == ""
    present = acoustics(background_noise_present=True, background_noise_severity="low", noise_type_guess="music")
    assert pipeline.merge(present, judgement(background_noise_type=" ")).background_noise_type == "music"


def test_analyze_falls_back_when_the_llm_is_unavailable(monkeypatch, caplog):
    def down(*args, **kwargs):
        raise llm.LLMUnavailable("no key")

    monkeypatch.setattr(llm, "judge", down)
    r = pipeline.analyze(np.stack([noise(3, -40)]), SR)
    assert (r.emotional_tone, r.emotional_intensity) == ("neutral", "low")
    assert r.confidence == pipeline.FALLBACK_CONFIDENCE
    assert sum("fallback" in m for m in caplog.messages) == 1
