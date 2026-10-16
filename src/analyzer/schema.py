"""Output schema from the trial brief. Field order is the order results are written in."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EmotionalTone = Literal["neutral", "satisfied", "frustrated", "upset", "distressed"]
Intensity = Literal["low", "medium", "high"]
Severity = Literal["none", "low", "medium", "high"]
AudioQuality = Literal["clear", "slightly_impaired", "severely_impaired"]

TONES: tuple[str, ...] = EmotionalTone.__args__


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotional_tone: EmotionalTone
    emotional_intensity: Intensity
    background_noise_present: bool
    background_noise_type: str  # empty string when no noise is present
    background_noise_severity: Severity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)


FIELDS: list[str] = list(Result.model_fields)
