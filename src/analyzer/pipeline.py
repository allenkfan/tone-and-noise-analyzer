"""Per-clip analysis: acoustic measurements on the box, the audio LLM for the subjective fields."""

import logging

import numpy as np

from analyzer import acoustic, llm, ser
from analyzer.schema import Result

log = logging.getLogger("analyzer")

FALLBACK_CONFIDENCE = 0.2


def analyze(samples: np.ndarray, sr: int) -> Result:
    """Analyze one clip. `samples` is float32 with shape (channels, n) at `sr` Hz.

    Silence, overlap, quality and noise presence and severity come from `acoustic`. Tone,
    intensity, noise type and confidence come from `llm`; without it they fall back to defaults.
    The voice's arousal from `ser`, over all speech, goes to the LLM as context; then the
    arousal over the customer's turns, as the LLM reports them, adjusts its tone (`ser.fuse`).
    """
    ac = acoustic.measure(samples, sr)
    voice = _voice(samples, sr)
    context = ac.as_context() | ({"ser": voice.as_context()} if voice else {})
    try:
        judgement = llm.judge(samples, sr, acoustics=context)
    except llm.LLMUnavailable as exc:
        log.warning("llm unavailable, acoustic fallback: %s", exc)
        judgement = None
    if voice and judgement and judgement.customer_turns:
        voice = _voice(samples, sr, judgement.customer_turns) or voice
    return merge(ac, judgement, voice)


def _voice(samples: np.ndarray, sr: int, turns: list[tuple[float, float]] | None = None) -> "ser.SER | None":
    try:
        return ser.measure(samples, sr, turns)
    except Exception as exc:  # the voice model is optional; the clip still gets an answer
        log.warning("ser unavailable: %s", exc)
        return None


def merge(ac: acoustic.Acoustics, judgement: llm.Judgement | None, voice: "ser.SER | None" = None) -> Result:
    noise_type = ""
    if ac.background_noise_present:
        heard = judgement.background_noise_type.strip() if judgement else ""
        noise_type = heard or ac.noise_type_guess or "background noise"
    tone, intensity = (judgement.emotional_tone, judgement.emotional_intensity) if judgement else ("neutral", "low")
    if ser.FUSE:
        tone, intensity, _ = ser.fuse(tone, intensity, voice, judgement.words_sentiment if judgement else "")
    return Result(
        emotional_tone=tone,
        emotional_intensity=intensity,
        background_noise_present=ac.background_noise_present,
        background_noise_type=noise_type,
        background_noise_severity=ac.background_noise_severity,
        audio_quality=ac.audio_quality,
        speaker_overlap_present=ac.speaker_overlap_present,  # measured; the LLM never reported overlap
        long_silence_present=ac.long_silence_present,
        confidence=judgement.confidence if judgement else FALLBACK_CONFIDENCE,
    )
