"""Decode any ffmpeg-readable audio file to float32 PCM."""

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".oga", ".opus", ".flac", ".m4a", ".aac", ".webm"}
DECODE_TIMEOUT_SECONDS = 600


class AudioError(Exception):
    """The file could not be decoded as audio."""


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return the first audio stream as float32 with shape (channels, n), resampled to `sr`."""
    channels = _channel_count(path)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
        timeout=DECODE_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise AudioError(f"could not decode audio: {_last_line(proc.stderr, path)}")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if samples.size < channels:
        raise AudioError("file contains no audio samples")
    return np.ascontiguousarray(samples[: samples.size - samples.size % channels].reshape(-1, channels).T)


def _channel_count(path: Path) -> int:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
         "-of", "csv=p=0", str(path)],
        capture_output=True,
        timeout=DECODE_TIMEOUT_SECONDS,
    )
    out = proc.stdout.decode().strip()
    if proc.returncode != 0:
        raise AudioError(f"not a readable audio file: {_last_line(proc.stderr, path)}")
    if not out.isdigit() or int(out) < 1:
        raise AudioError("file has no audio stream")
    return int(out)


def _last_line(stderr: bytes, path: Path) -> str:
    """Last ffmpeg error line, with the server-side path reduced to the file name."""
    lines = stderr.decode(errors="replace").strip().splitlines()
    line = lines[-1] if lines else "unknown error"
    return line.replace(str(path.resolve()), path.name).replace(str(path), path.name)
