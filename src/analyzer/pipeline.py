"""Per-clip analysis: acoustic measurements on the box, the audio LLM for the subjective fields."""

import logging

import numpy as np

from analyzer import acoustic, llm
from analyzer.schema import Result

log = logging.getLogger("analyzer")

FALLBACK_CONFIDENCE = 0.2


def analyze(samples: np.ndarray, sr: int) -> Result:
    """Analyze one clip. `samples` is float32 with shape (channels, n) at `sr` Hz.

    Silence, overlap, quality and noise presence and severity come from `acoustic`. Tone,
    intensity, noise type and confidence come from `llm`; without it they fall back to defaults.
    """
    ac = acoustic.measure(samples, sr)
    try:
        judgement = llm.judge(samples, sr, acoustics=ac.as_context())
    except llm.LLMUnavailable as exc:
        log.warning("llm unavailable, acoustic fallback: %s", exc)
        judgement = None
    return merge(ac, judgement)


def merge(ac: acoustic.Acoustics, judgement: llm.Judgement | None) -> Result:
    noise_type = ""
    if ac.background_noise_present:
        heard = judgement.background_noise_type.strip() if judgement else ""
        noise_type = heard or ac.noise_type_guess or "background noise"
    return Result(
        emotional_tone=judgement.emotional_tone if judgement else "neutral",
        emotional_intensity=judgement.emotional_intensity if judgement else "low",
        background_noise_present=ac.background_noise_present,
        background_noise_type=noise_type,
        background_noise_severity=ac.background_noise_severity,
        audio_quality=ac.audio_quality,
        speaker_overlap_present=ac.speaker_overlap_present,  # measured; the LLM never reported overlap
        long_silence_present=ac.long_silence_present,
        confidence=judgement.confidence if judgement else FALLBACK_CONFIDENCE,
    )
