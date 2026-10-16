"""Per-clip analysis. This is where the models go."""

import numpy as np

from analyzer.schema import Result


def analyze(samples: np.ndarray, sr: int) -> Result:
    """Analyze one clip. `samples` is float32 with shape (channels, n) at `sr` Hz.

    Placeholder baseline: answers "nothing detected" for every clip, with zero
    confidence. Replace with real models.
    """
    return Result(
        emotional_tone="neutral",
        emotional_intensity="low",
        background_noise_present=False,
        background_noise_type="",
        background_noise_severity="none",
        audio_quality="clear",
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.0,
    )
