"""Audio LLM judge for the subjective fields: tone, intensity, noise type and overlap.

One Gemini call per clip. The audio goes inline in the request, re-encoded to
16 kHz mono Opus, so nothing is uploaded to the Files API or cached on disk.
The physical fields (silence, quality, noise presence and severity) are measured
by `analyzer.acoustic` and passed in as context; the model doesn't decide them.

    python -m analyzer.llm <folder-or-zip> [--model ID] [--workers N] [--out DIR]

The dev runner scores the fields owned here against a labeled folder.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from analyzer.schema import EmotionalTone, Intensity

log = logging.getLogger("analyzer")

MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "v6"

REQUEST_TIMEOUT_SECONDS = 90
HTTP_ATTEMPTS = 3  # per call, on timeouts, 408, 429 and 5xx
RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]
MAX_OUTPUT_TOKENS = 2048
MAX_INLINE_BYTES = 19 * 1024 * 1024  # inline request limit is 20 MB

# After this many consecutive failed calls, skip the API for a while so a batch
# during an outage takes the fallback quickly instead of waiting out every timeout.
BREAKER_FAILURES = 3
BREAKER_SECONDS = 120


@dataclass(frozen=True)
class ModelSpec:
    text_usd: float  # $ per 1M input tokens, text
    audio_usd: float  # $ per 1M input tokens, audio
    output_usd: float  # $ per 1M output tokens, thinking included
    thinking: dict  # google.genai ThinkingConfig fields
    temperature: float | None  # None leaves the model's default


# Paid-tier standard prices, checked 2026-10-16 at ai.google.dev/gemini-api/docs/pricing.
# Thinking is held at the minimum: thought tokens bill as output and would dominate the cost.
# Gemini 3 models are meant to run at their default temperature; 3.5 Flash-Lite runs at 0, which
# on the tune split kept accuracy (0.377 vs 0.390, McNemar p = 0.61) and made answers more
# repeatable. The 2.5 models are cheaper
# but return 404 "no longer available to new users" for this account's key.
MODELS: dict[str, ModelSpec] = {
    "gemini-3.5-flash-lite": ModelSpec(0.30, 0.30, 2.50, {"thinking_level": "minimal"}, 0.0),
    # Introductory prices through 2026-12-31; $1.50 in and $7.50 out from 2027-01-01.
    "gemini-3.6-flash": ModelSpec(0.75, 0.75, 3.75, {"thinking_level": "minimal"}, None),
    "gemini-3.8-flash": ModelSpec(0.75, 0.75, 3.75, {"thinking_level": "low"}, None),
}

# Piecewise-linear maps from the model's self-reported confidence to observed tone
# accuracy, as (raw, calibrated) points. Fitted with --fit-calibration on the CREMA-D
# tune split (300 clips, prompt v6, temperature 0, with the acoustic context); points backed by
# fewer than 5 clips are merged into their neighbour. A model without an entry reports raw confidence.
CALIBRATION: dict[str, list[tuple[float, float]]] = {
    "gemini-3.5-flash-lite": [(0.8, 0.2), (0.85, 0.308), (0.9, 0.378), (0.95, 0.406), (1.0, 0.406)],
}

SYSTEM_PROMPT = """\
You listen to recordings of customer-service phone calls and label how the customer \
feels, plus two properties of the recording. Reply only with the JSON object described \
by the response schema.

WHOSE TONE
A call has an agent, who works for the company (greets the caller, often names the \
company, follows a script, asks verification questions, offers help), and a customer, \
who called about a problem or request. Both voices are on the same channel, so decide \
from what is said which voice is the customer. Judge tone and \
intensity from the customer alone; the agent's scripted warmth or calm says \
nothing about the customer. If only one person can be heard, judge that person.

TONE: the main emotion the customer expresses over the clip. If it shifts, \
choose the one that is most prominent and sustained.
- neutral: no clear positive or negative emotion; routine and matter-of-fact.
- satisfied: clearly positive; pleased, relieved, grateful or appreciative.
- frustrated: annoyed, impatient or dissatisfied, without strong anger or distress.
- upset: clearly angry, agitated or strongly dissatisfied.
- distressed: highly emotional, overwhelmed, panicked, crying, or otherwise escalated.

INTENSITY: how strongly the customer expresses that tone.
- low: subtle or mild.
- medium: clear and sustained.
- high: strong or escalated; the kind of call someone should look at.

Evidence for tone: combine what the customer says (complaints, thanks, repeated \
requests, threats to leave) with how they say it (strain, sighs, sarcasm, a shaking \
or tearful voice, pace). Loudness by itself is not evidence of frustration, anger or \
distress: people talk loudly on a bad line, in a noisy place, or by habit. A routine \
"thanks, bye" is not by itself satisfaction.

NOISE: the dominant non-speech sound behind the conversation, in a few \
plain lowercase words, for example: television, music, office chatter, road noise, \
keyboard typing, wind, mechanical noise, static, crowd, dog barking. The voices of the \
agent and the customer are not background noise; voices from a TV or from other people \
in the room are. A steady hiss or static behind the speech counts as noise, but codec \
distortion, dropouts, muffled or quiet speech are line quality, not background noise. \
Return an empty string when there is no meaningful background sound; barely \
perceptible artifacts don't count.

OVERLAP: true when two or more people talk at the same time for long \
enough to affect understanding or analysis. Brief backchannels ("mm-hm", "yeah") and a \
single overlapping word don't count.

CONFIDENCE: your probability, from 0 to 1, that your tone is the one a careful \
human annotator would choose. Go lower when the customer says little, the emotion is \
mixed or borderline between two tones, or the speech is hard to hear.

RATIONALE: one or two sentences: which voice you took as the customer and the main \
evidence for the tone."""

USER_PROMPT = "Label this call."


class LLMUnavailable(Exception):
    """No API key, or the API failed after retries. The caller falls back to defaults."""


@dataclass
class Judgement:
    emotional_tone: EmotionalTone
    emotional_intensity: Intensity
    background_noise_type: str  # open text; "" when acoustics say no noise
    speaker_overlap_present: bool
    confidence: float  # 0..1, calibrated when CALIBRATION is set
    rationale: str  # one or two sentences, logged at DEBUG, never written to the CSV
    usage: dict = field(default_factory=dict)  # tokens, estimated USD, seconds


# The response schema. Rationale comes first so the model states its evidence before the
# labels: without it, tone accuracy on the tune split fell from 0.39 to 0.28-0.34 (prompts
# v3 to v5, paired McNemar p <= 0.015). Keys are short because output tokens cost 8x input.
class _Answer(BaseModel):
    rationale: str
    tone: EmotionalTone
    intensity: Intensity
    noise: str
    overlap: bool
    confidence: float = Field(ge=0.0, le=1.0)


def api_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY") or ""


def judge(samples: np.ndarray, sr: int, acoustics: dict | None = None, *, model: str = MODEL) -> Judgement:
    """Judge one clip. `samples` is float32 with shape (channels, n) or (n,) at `sr` Hz.

    Raises LLMUnavailable when there is no key or the call fails, so the caller can fall back.
    """
    key = api_key()
    if not key:
        raise LLMUnavailable("no Gemini API key (set GEMINI_API_KEY)")
    if _breaker.is_open():
        raise LLMUnavailable("skipped: the API failed repeatedly; retrying shortly")
    spec = MODELS[model]
    audio = encode_audio(samples, sr)
    request = dict(model=model, contents=_contents(audio, acoustics), config=_config(spec))

    start = time.perf_counter()
    try:
        response, answer = _call(_client(key), request)
    except LLMUnavailable:
        _breaker.failed()
        raise
    _breaker.succeeded()
    seconds = time.perf_counter() - start

    noise_type = answer.noise.strip().lower()
    if _noise_absent(acoustics):
        noise_type = ""
    usage = _usage(response, spec, model, seconds) | {"raw_confidence": answer.confidence}
    log.info("llm %s: %d audio tokens, %d output tokens, $%.5f, %.1f s",
             model, usage["audio_tokens"], usage["output_tokens"], usage["usd"], seconds)
    log.debug("llm rationale: %s", answer.rationale)
    return Judgement(answer.tone, answer.intensity, noise_type, answer.overlap,
                     calibrate(answer.confidence, model), answer.rationale, usage)


def calibrate(raw: float, model: str = MODEL) -> float:
    return calibrate_with(CALIBRATION.get(model, []), raw)


def calibrate_with(points: list[tuple[float, float]], raw: float) -> float:
    if not points:
        return raw
    xs, ys = zip(*points)
    return float(np.interp(raw, xs, ys))


def fit_calibration(raw: list[float], correct: list[bool]) -> list[tuple[float, float]]:
    """Isotonic fit of observed accuracy against raw confidence (pool adjacent violators).

    Returns (raw, calibrated) points for CALIBRATION, one per distinct raw value.
    """
    groups: dict[float, tuple[int, int]] = {}
    for r, c in zip(raw, correct):
        n, k = groups.get(r, (0, 0))
        groups[r] = (n + 1, k + int(c))
    blocks: list[list] = []  # [raw values, clips, correct]
    for x in sorted(groups):
        blocks.append([[x], *groups[x]])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][1] > blocks[-1][2] / blocks[-1][1]:
            xs, n, k = blocks.pop()
            blocks[-1][0] += xs
            blocks[-1][1] += n
            blocks[-1][2] += k
    return [(round(x, 3), round(k / n, 3)) for xs, n, k in blocks for x in xs]


def calibration_error(confidence: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: the clip-weighted gap between confidence and accuracy per bin."""
    conf, hit = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    which = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[which == b].mean() - hit[which == b].mean()) * np.sum(which == b)
                     for b in range(bins) if np.any(which == b)) / len(conf))


def encode_audio(samples: np.ndarray, sr: int) -> bytes:
    """Average the channels and encode as 16 kHz mono Opus in an Ogg container, in memory.

    The API bills audio per second, not per byte, so the bitrate is chosen for fidelity.
    """
    mono = samples.mean(axis=0) if samples.ndim == 2 else samples
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
             "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", "-f", "ogg", "-"],
            input=np.ascontiguousarray(mono, dtype="<f4").tobytes(),
            capture_output=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LLMUnavailable(f"could not encode audio for the API: {exc}") from exc
    if proc.returncode != 0:
        raise LLMUnavailable(f"could not encode audio for the API: {proc.stderr.decode(errors='replace').strip()}")
    if len(proc.stdout) > MAX_INLINE_BYTES:
        raise LLMUnavailable("clip is too long to send inline")
    return proc.stdout


def _contents(audio: bytes, acoustics: dict | None) -> list:
    from google.genai import types

    parts = [types.Part.from_bytes(data=audio, mime_type="audio/ogg"), types.Part.from_text(text=USER_PROMPT)]
    if acoustics is not None:
        parts.append(types.Part.from_text(text=_acoustics_text(acoustics)))
    return [types.Content(role="user", parts=parts)]


def _acoustics_text(acoustics: dict) -> str:
    """The part of the measurements the judge needs, in a line instead of a JSON dump.

    Silence, quality and noise severity are decided by the measurements and left out.
    """
    if _noise_absent(acoustics):
        return 'Signal analysis found no background noise: answer noise "".'
    tags = ", ".join(f"{label} {score:.2f}" for label, score in (acoustics.get("noise_tags") or [])[:3])
    found = "background noise" if acoustics.get("background_noise_present") else "no decision on noise"
    return f"Signal analysis: {found}" + (f"; sound-event tags (hints only): {tags}." if tags else ".")


def _config(spec: ModelSpec):
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=_Answer.model_json_schema(),
        thinking_config=types.ThinkingConfig(**spec.thinking),
        temperature=spec.temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _call(client, request: dict):
    """Call the API; retry once if the reply doesn't fit the schema. HTTP retries happen in the client."""
    import httpx
    from google.genai import errors

    for attempt in (1, 2):
        try:
            response = client.models.generate_content(**request)
        except (errors.APIError, httpx.HTTPError, OSError) as exc:
            raise LLMUnavailable(f"Gemini API error: {type(exc).__name__}: {exc}") from exc
        try:
            return response, _Answer.model_validate_json(response.text or "")
        except ValidationError as exc:
            log.warning("llm reply didn't match the schema (attempt %d): %s", attempt, exc.errors()[0]["msg"])
    raise LLMUnavailable("Gemini reply didn't match the schema twice")


_clients: dict[str, object] = {}
_clients_lock = threading.Lock()


def _client(key: str):
    with _clients_lock:
        if key not in _clients:
            from google import genai
            from google.genai import types

            _clients[key] = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(
                    attempts=HTTP_ATTEMPTS, initial_delay=1.0, max_delay=8.0, http_status_codes=RETRY_STATUS_CODES),
            ))
        return _clients[key]


def _noise_absent(acoustics: dict | None) -> bool:
    return acoustics is not None and acoustics.get("background_noise_present") is False


def _usage(response, spec: ModelSpec, model: str, seconds: float) -> dict:
    meta = response.usage_metadata
    prompt = meta.prompt_token_count or 0
    audio = sum(d.token_count or 0 for d in meta.prompt_tokens_details or [] if str(d.modality).endswith("AUDIO"))
    output = meta.candidates_token_count or 0
    thoughts = meta.thoughts_token_count or 0
    usd = ((prompt - audio) * spec.text_usd + audio * spec.audio_usd + (output + thoughts) * spec.output_usd) / 1e6
    return {"model": model, "prompt_tokens": prompt, "audio_tokens": audio, "output_tokens": output,
            "thought_tokens": thoughts, "usd": usd, "seconds": round(seconds, 3)}


class _Breaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until

    def failed(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= BREAKER_FAILURES:
                self._open_until = time.monotonic() + BREAKER_SECONDS
                log.warning("llm: %d failures in a row; using the fallback for %d s", self._failures, BREAKER_SECONDS)

    def succeeded(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0


_breaker = _Breaker()


# ---------------------------------------------------------------- dev runner

OWN_FIELDS = ["emotional_tone", "emotional_intensity", "speaker_overlap_present"]
PHYSICAL_DEFAULTS = {"background_noise_present": False, "background_noise_severity": "none",
                     "audio_quality": "clear", "long_silence_present": False}


def _judge_item(item, model: str, with_acoustics: bool):
    from analyzer.audio import SAMPLE_RATE, load_audio
    from analyzer.batch import Row
    from analyzer.schema import Result

    start = time.perf_counter()
    try:
        samples = load_audio(item.path)
        context = None
        if with_acoustics:
            from analyzer import acoustic

            context = acoustic.measure(samples, SAMPLE_RATE).as_context()
        j = judge(samples, SAMPLE_RATE, context, model=model)
    except Exception as exc:
        return Row(item.name, "failed", f"{type(exc).__name__}: {exc}", label=item.label), None
    physical = {k: (item.label or {}).get(k, v) for k, v in PHYSICAL_DEFAULTS.items()}
    result = Result(emotional_tone=j.emotional_tone, emotional_intensity=j.emotional_intensity,
                    background_noise_type=j.background_noise_type,
                    speaker_overlap_present=j.speaker_overlap_present, confidence=j.confidence, **physical)
    row = Row(item.name, "ok", result=result, label=item.label,
              audio_seconds=round(samples.shape[1] / SAMPLE_RATE, 3),
              processing_seconds=round(time.perf_counter() - start, 3))
    return row, j


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    from analyzer import manifest
    from analyzer.metrics import score

    parser = argparse.ArgumentParser(prog="python -m analyzer.llm",
                                     description="Score the LLM judge's fields on a labeled folder or ZIP.")
    parser.add_argument("source", type=Path, help="batch folder or .zip file with a labels CSV")
    parser.add_argument("--model", default=MODEL, choices=sorted(MODELS))
    parser.add_argument("--workers", type=int, default=4, help="parallel API calls (default: 4)")
    parser.add_argument("--limit", type=int, help="judge only the first N clips")
    parser.add_argument("--out", type=Path, default=Path("out/llm"), help="output directory (default: out/llm)")
    parser.add_argument("--acoustics", action="store_true",
                        help="measure each clip and pass the measurements to the judge, as the pipeline does")
    parser.add_argument("--fit-calibration", action="store_true",
                        help="fit CALIBRATION on these clips' tone accuracy (use the tune split only)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    load_dotenv(Path("private/.env"))
    if not api_key():
        parser.exit(1, "No Gemini API key: set GEMINI_API_KEY in private/.env\n")

    with tempfile.TemporaryDirectory() as tmp:
        if args.source.suffix.lower() == ".zip":
            batch = manifest.from_zip(args.source, Path(tmp))
        else:
            batch = manifest.from_folder(args.source)
        items = batch.runnable[: args.limit]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            done = []
            for row, j in pool.map(lambda item: _judge_item(item, args.model, args.acoustics), items):
                done.append((row, j))
                pred = f"{j.emotional_tone}/{j.emotional_intensity} {j.confidence:.2f}" if j else row.error
                print(f"{len(done):>4}/{len(items)} {row.name}: {pred}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.model}.jsonl"
    with open(out_path, "w") as f:
        for row, j in done:
            f.write(json.dumps({"name": row.name, "prompt_version": PROMPT_VERSION, "label": row.label,
                                "status": row.status, "error": row.error,
                                "judgement": asdict(j) if j else None, "audio_seconds": row.audio_seconds}) + "\n")

    rows = [row for row, _ in done]
    judged = [(row, j) for row, j in done if j]
    print(f"\nWrote {out_path}  ·  model {args.model}  ·  prompt {PROMPT_VERSION}")
    print(f"{len(judged)} judged, {len(done) - len(judged)} failed")
    if not judged:
        return

    print("\n| Clip | Label tone/intensity | Predicted | Conf | Overlap label/pred | Noise type label / pred |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row, j in judged:
        label = row.label or {}
        print(f"| {row.name} | {label.get('emotional_tone', '?')}/{label.get('emotional_intensity', '?')} "
              f"| {j.emotional_tone}/{j.emotional_intensity} | {j.confidence:.2f} "
              f"| {label.get('speaker_overlap_present', '?')}/{j.speaker_overlap_present} "
              f"| {label.get('background_noise_type', '?')!r} / {j.background_noise_type!r} |")

    # Cost = a fixed part per clip (prompt text and output) plus audio tokens per second of audio.
    spec = MODELS[args.model]
    audio_min = sum(row.audio_seconds for row, _ in judged) / 60
    usd = sum(j.usage["usd"] for _, j in judged)
    audio_tokens = sum(j.usage["audio_tokens"] for _, j in judged)
    audio_usd_per_min = audio_tokens / audio_min * spec.audio_usd / 1e6
    fixed_usd = (usd - audio_tokens * spec.audio_usd / 1e6) / len(judged)
    call_s = sorted(j.usage["seconds"] for _, j in judged)
    print(f"\n{len(judged)} clips, {audio_min:.2f} audio min, ${usd:.5f} total · "
          f"{audio_tokens / audio_min:.0f} audio tokens per min · ${fixed_usd:.6f} fixed per clip")
    print("Projected $ per audio min (% of the $0.003 ceiling): " + ", ".join(
        f"{s} s clip ${audio_usd_per_min + fixed_usd * 60 / s:.5f} ({(audio_usd_per_min + fixed_usd * 60 / s) / 0.003:.0%})"
        for s in (30, 60, 180)))
    print(f"API seconds per clip: median {call_s[len(call_s) // 2]:.1f}, max {call_s[-1]:.1f}")

    if scores := score(rows):
        scores = replace(scores, accuracy={f: a for f, a in scores.accuracy.items() if f in OWN_FIELDS})
        print("\n" + scores.to_markdown())

    toned = [(row.label["emotional_tone"] == j.emotional_tone, j) for row, j in judged
             if row.label and "emotional_tone" in row.label]
    if toned:
        correct = [c for c, _ in toned]
        print(f"\nTone ECE (10 bins): {calibration_error([j.confidence for _, j in toned], correct):.3f} "
              f"on the reported confidence{' (calibrated)' if args.model in CALIBRATION else ''}")
        if args.fit_calibration:
            raw = [j.usage["raw_confidence"] for _, j in toned]
            points = fit_calibration(raw, correct)
            fitted = [calibrate_with(points, r) for r in raw]
            print(f"In-sample ECE after fitting: {calibration_error(fitted, correct):.3f}")
            print(f"CALIBRATION = {points}")


if __name__ == "__main__":
    main()
