# Technical memo

Status 2026-10-16: delivered as is. Sections marked **(acoustic)** are written by the session that owns the deterministic measurements; everything else covers the audio LLM judge.

## 1. Summary

The analyzer is a hybrid. Deterministic signal measurements on the server decide the physical fields (`long_silence_present`, `audio_quality`, `background_noise_present`, `background_noise_severity`). One call per clip to an audio-capable LLM, Google's `gemini-3.5-flash-lite`, decides the subjective fields (`emotional_tone`, `emotional_intensity`, `background_noise_type`, `speaker_overlap_present`), with the measurements passed in as context. If the API is unavailable, the batch still completes with a documented fallback.

Headline numbers for the LLM half:

| | Value | Source |
| --- | --- | --- |
| Tone accuracy / macro-F1, held-out acted speech | 0.377 / 0.325 (chance 0.20) | CREMA-D test split, 300 clips, scored once |
| API cost per audio minute | $0.00059 at 3-minute calls, $0.00124 at 30-second calls (20% and 41% of the $0.003 ceiling) | measured on 603 clips |
| API latency | median 1.4 s per short clip; 1.8 to 3.0 s per real call (31 to 172 s long), about 1.8 s per audio minute | measured |

Tone accuracy is the weak point, and section 4 explains why: the model reads emotion mainly from the words, and barely from the voice. Section 9 lists the follow-ups that address it.

## 2. Approaches considered

| Approach | What it is | Outcome |
| --- | --- | --- |
| Placeholder baseline | Every field "nothing detected", confidence 0 | Floor: tone accuracy 0.33 and macro-F1 0.17 on the 3 example calls |
| A. Deterministic per-field measurements, no tone model | Silence, quality, noise presence and severity from signal measurements | Kept for the physical fields **(acoustic: results)** |
| B. Audio LLM end to end | One Gemini call returns every field | Not used alone: the brief scores quality and noise separately and warns against inferring noise from poor quality; measurements are more reliable for those |
| **C. Hybrid (selected)** | A for the physical fields, the LLM only for the four subjective fields, with A's measurements in the prompt | Selected: best available accuracy on every field without new model training, inside the cost ceiling |
| D. Local ASR + text model + speech-emotion model | faster-whisper transcript, text sentiment, emotion2vec, small fusion classifier | Not built; 2 to 3 days and needs call-like training data. Part of it is follow-up 1 |
| E. Local speech-emotion model + acoustic features only | No transcript | Not built; expected weakest on satisfied versus neutral |

Other external models were ruled out on cost or capability: OpenAI's audio models and Whisper cost $0.006 per minute or more; Claude takes no audio input, so it would need local speech recognition first; Gemini models above Flash-Lite cost two to three times as much (section 5) and did worse in our test (section 4).

## 3. Final architecture

```
audio file ──ffmpeg──► float32, 16 kHz
                         │
                         ├─► acoustic.measure ──► silence, quality, noise presence/severity, noise tags (acoustic)
                         │                              │ as JSON context
                         └─► llm.judge ◄────────────────┘
                               │ 16 kHz mono Opus, sent inline to gemini-3.5-flash-lite
                               ▼
                         tone, intensity, noise type, overlap, confidence
                               │
                         pipeline.analyze merges both into one Result
```

`llm.judge` (`src/analyzer/llm.py`):

- **Request.** The clip is averaged to mono and re-encoded to 16 kHz Opus in memory (24 kbps; the API bills per second of audio, not per byte) and sent inline. The Files API, which stores uploads, is never used, and nothing is written to disk.
- **Prompt.** The system prompt restates the brief's definitions in our own words. It tells the model to judge the **customer**, whom it identifies from context (the agent greets, names the company, follows a script), since both voices share one channel; not to infer frustration or distress from loudness alone; and not to treat line-quality problems as background noise. The measurements arrive as JSON with the instruction that the physical fields are already decided; when they say no noise, the noise type is forced to an empty string.
- **Structured output.** Gemini's JSON-schema mode with the enum values from the brief, validated again with Pydantic. A one- or two-sentence rationale comes first in the schema so the model states its evidence before the labels; the rationale is logged only at DEBUG and never written to the CSV.
- **Thinking** is held at the model's minimum (`minimal`). Thought tokens bill as output at $2.50 per million and would dominate the cost.
- **Reliability.** 90 s timeout; up to 3 attempts with exponential backoff on timeouts, 408, 429 and 5xx; one retry if the reply fails the schema. Any remaining failure raises `LLMUnavailable`, and `pipeline.analyze` falls back to neutral tone, low intensity, no overlap, the acoustic noise-type guess and confidence 0.2. After 3 consecutive failures, a breaker skips the API for 120 s, so a batch during an outage finishes in seconds instead of waiting out every timeout. With no API key configured, the app starts and runs entirely on the fallback.

**(acoustic: the measurements, the noise tagger, and how `pipeline.analyze` merges and sets the final confidence.)**

## 4. Validation

### 4.1 Tone and intensity

**Data.** No call-like corpus with tone labels is available under a usable license this week (MSP-Podcast needs an institutional signature; IEMOCAP needs a USC agreement). We built a validation set from CREMA-D (ODbL 1.0 / DbCL 1.0): 7,442 clips of 91 actors speaking 12 fixed sentences in six emotions, each rated by crowd listeners from the voice alone. `private/tools/make_val_tone.py` (seed 20261016) builds it:

- Only clips whose voice-only majority vote matches the acted emotion are kept, so each label is what listeners actually heard.
- Mapping to the brief's tones, which is lossy: happy → satisfied; neutral → neutral; anger → upset, except anger rated low-intensity → frustrated (a proxy: acted corpora have no "frustrated", and the brief defines it as annoyance without strong anger); fear or sadness rated 65 or more out of 100 for intensity → distressed; disgust and milder fear or sadness are dropped.
- Intensity comes from the listeners' mean rating: below 55 low, 55 to 70 medium, 70 and above high. Neutral clips carry no intensity label.
- Every clip passes through a telephone chain (300 to 3,400 Hz band-pass, 16 kHz mono, Opus 16 kbps).
- **Leakage control.** Actors are split 61 / 30 into tune and test, stratified by sex, so no voice appears in both. Each split holds 60 clips per tone (300 clips). Prompts and calibration were chosen on tune only; test was scored once, at the end. The three labeled example calls were never used for tuning.

**Metric.** Tone accuracy and macro-F1 over the five tones; intensity accuracy on clips with an intensity label.

**Tune split** (300 clips), used to choose the model and prompt:

| Model, prompt | Tone accuracy | Tone macro-F1 | Intensity accuracy |
| --- | --- | --- | --- |
| gemini-3.5-flash-lite, v1 (selected) | 0.390 | 0.356 | 0.421 |
| gemini-3.5-flash-lite, v2: adds "judge the voice as much as the words" | 0.383 | 0.341 | 0.442 |
| gemini-3.6-flash, v1 | 0.320 | 0.268 | 0.396 |

**Test split** (300 clips, gemini-3.5-flash-lite, prompt v1, scored once):

| Tone accuracy | Tone macro-F1 | Intensity accuracy | Expected calibration error |
| --- | --- | --- | --- |
| 0.377 | 0.325 | 0.450 | 0.029 |

| Tone | Precision | Recall | F1 | Clips |
| --- | --- | --- | --- | --- |
| neutral | 0.31 | 0.95 | 0.47 | 60 |
| satisfied | 0.75 | 0.05 | 0.09 | 60 |
| frustrated | 0.34 | 0.35 | 0.34 | 60 |
| upset | 0.55 | 0.37 | 0.44 | 60 |
| distressed | 0.83 | 0.17 | 0.28 | 60 |

Confusion matrix (rows: label, columns: prediction):

| | neutral | satisfied | frustrated | upset | distressed |
| --- | --- | --- | --- | --- | --- |
| neutral | 57 | 0 | 3 | 0 | 0 |
| satisfied | 50 | 3 | 3 | 2 | 2 |
| frustrated | 34 | 0 | 21 | 5 | 0 |
| upset | 18 | 0 | 20 | 22 | 0 |
| distressed | 23 | 1 | 15 | 11 | 10 |

The test result matches tune (0.390 / 0.356), so tuning didn't overfit; there was little to tune.

**What the errors show.** In every run the model calls most emotional clips neutral. Its rationales describe the words ("speaks calmly about the airplane being full") rather than the voice, and CREMA-D's sentences are emotionally neutral text, so the emotion is only in the delivery. Asking the model to weigh the voice (v2) changed nothing, and the stronger, pricier model was worse. Our reading: these models infer emotion mostly from lexical content and barely from prosody.

**How far this transfers to real calls.** It cuts both ways. Real customers say emotional things, which these models do use, so acted, lexically neutral speech probably understates call accuracy. But the three labeled example calls, which were not used for tuning, scored 0 of 3 on tone. Three clips can't support a number, but they don't contradict the finding.

**Overlap and noise type.** CREMA-D clips are single-speaker and clean, so they test only false positives: overlap was predicted false on all 600 clips. No call-like data with overlap or noise-type labels exists here. **(acoustic: the synthetic set for overlap and noise.)**

### 4.2 Physical fields

**(acoustic: synthetic construction, metrics, results.)**

### 4.3 Confidence calibration

The model reports its own confidence. On the tune split it said 0.80 to 0.95 for every clip, while accuracy was flat at about 0.25 to 0.40 across that range: overconfident and barely informative. We fitted an isotonic map (pool-adjacent-violators) from raw confidence to observed tone accuracy on tune (`python -m analyzer.llm … --fit-calibration`) and apply it in `judge`:

| Raw | 0.80 | 0.85 | 0.90 | 0.95 |
| --- | --- | --- | --- | --- |
| Calibrated | 0.25 | 0.39 | 0.39 | 0.40 |

Expected calibration error (10 bins): 0.526 on tune before calibration; 0.029 on test after calibration. The calibration was fitted on acted speech and transfers to calls only approximately. It is per model; another model ID reports raw confidence until it is fitted.

**Ambiguous tone.** When the customer's tone shifts, the prompt asks for the most prominent and sustained one; borderline cases should get lower confidence, but the raw scores show the model rarely does that, which is why calibration matters. When the API fails, confidence is 0.2.

## 5. Cost

### 5.1 API (Gemini)

Measured over the 600 validation clips and the 3 example calls, using paid-tier standard prices as of 2026-10-16:

| | gemini-3.5-flash-lite |
| --- | --- |
| Price | $0.30 per million input tokens (text and audio alike), $2.50 per million output tokens including thinking |
| Audio tokens | 1,512 per audio minute (about 25 per second) → **$0.00045 per audio minute** |
| Fixed per call | 661 prompt-text tokens + about 79 output tokens, no thinking tokens → **$0.00040 per clip** |
| Cost per audio minute | $0.00045 + $0.00040 / clip length in minutes |

| Clip length | 10 s | 30 s | 60 s | 3 min |
| --- | --- | --- | --- | --- |
| $ per audio minute | 0.00285 | 0.00124 | 0.00085 | 0.00059 |
| Share of the $0.003 ceiling | 95% | 41% | 28% | 20% |

The fixed part dominates short clips; the API part of the cost reaches the ceiling at clips of about 9 seconds. Comparison, also measured: `gemini-3.6-flash` costs $0.00075 fixed per clip and $0.00113 per audio minute ($0.00138 per minute at 3-minute calls, $0.00263 at 30 seconds), at introductory prices that double on 2027-01-01. Gemini's batch mode would halve the API cost, but its turnaround (target 24 hours) doesn't suit a dashboard where the evaluator waits for results. `gemini-2.5-flash-lite`, cheaper on paper ($0.10 text, $0.40 output), is no longer available to new API users.

### 5.2 Compute and hosting

**(acoustic: measured compute per audio minute for the measurements, and the hosting assumption.)**

## 6. Latency

API seconds per call, measured with 8 calls in flight on the validation sets and one call at a time on the example calls:

| Clips | Median | Max | Per audio minute |
| --- | --- | --- | --- |
| CREMA-D, about 2.5 s each (600) | 1.4 s | 2.8 s | n/a (dominated by fixed overhead) |
| Example calls, 31 to 172 s (3) | 2.2 s | 3.0 s | 1.8 s |

The batch runner processes files one at a time, so a 100-call batch spends about 3 to 4 minutes waiting on the API. **(acoustic: decode and measurement time, end-to-end seconds per audio minute.)**

## 7. External API disclosure

- **Provider and model:** Google Gemini Developer API, model `gemini-3.5-flash-lite` (stable), paid tier, via the official `google-genai` SDK.
- **Pricing assumptions:** section 5.1. Prices checked 2026-10-16.
- **Customer audio leaves our infrastructure.** Each clip is sent to Google as an inline request: the audio re-encoded to 16 kHz mono, plus the signal measurements. Nothing is uploaded to the Files API or Google Cloud Storage.
- **Retention (Google's paid-tier terms, as of 2026-10-16):** prompts and responses are not used to improve Google's products. They are logged "for a limited period of time" solely to detect abuse, with no stated retention period. Data may be stored transiently or cached in any country where Google or its agents have facilities. No zero-retention option is described for this API.
- **What the response contains:** labels, a confidence and a one- or two-sentence rationale. No transcript is requested. The rationale is never written to the CSV or logged at INFO.
- **During development,** about 20 CREMA-D clips (public acted speech) were sent before billing was enabled, under the free tier's terms, which allow Google to use the content. No customer audio was sent under those terms.
- **If sending audio to a third party is not acceptable,** follow-up 1 (a local speech-emotion model) plus local speech recognition is the on-premises path.

## 8. Privacy and operations

- Uploaded audio is deleted after processing. Logs hold file names, statuses, timings and the API's token counts and cost; never audio, transcripts or rationales.
- The API key is read from `GEMINI_API_KEY` (falling back to `API_KEY`) and lives only in `private/.env` and the Space's secrets.
- Failure handling: section 3. A missing or invalid key does not stop the app; every clip silently takes the fallback. Check the Space's logs for `llm` lines after deploying.

## 9. Limitations and next steps

**Known failure modes**

- **Tone collapses to neutral** when the emotion is in the voice rather than the words (section 4.1). Expect the most errors on satisfied and frustrated customers who stay polite.
- **Customer identification** relies on the model telling agent from customer on a single channel; it may judge the wrong speaker on calls where the agent is the emotional one.
- **Overconfidence:** raw confidence is near 0.9 regardless of correctness. Calibration lowers it, but the calibration was fitted on acted speech.
- **Short clips** cost more per minute (section 5.1).
- **Dependency on an external service:** outages, rate limits or price changes affect results. The fallback keeps batches running, but its tone is always neutral.

**Follow-up improvements, in order of expected gain**

1. **Add a local speech-emotion model for the prosody signal.** emotion2vec or a wav2vec2 emotion model, exported to ONNX to keep the image free of PyTorch. Pass its scores to Gemini as context, like the noise tags, and fuse them with Gemini's judgement in a small classifier fitted on the tune split. This targets the measured weakness directly, and gives a reasonable tone in the fallback too. Cost: compute only.
2. **Make Gemini attend to the whole call.** Add a short speaker-attributed transcript field before the labels in the response schema (discarded after parsing, never logged; about +$0.0005 per audio minute), or allow `low` thinking. Both should help identify the customer and use the lexical evidence. Measuring the gain needs item 3.
3. **Call-like validation data.** MSP-Podcast (natural speech, academic license with an institutional signature) or synthetic two-speaker dialogues with scripted emotions. Acted single sentences can't measure customer identification, lexical cues or overlap.
4. **Concurrency.** The batch runner is sequential and the API calls are network-bound. A small thread pool (4 to 8 calls in flight, within the paid tier's rate limits) would cut batch time by about that factor.
5. **Lower the fixed cost for short clips.** Shorten the system prompt and the rationale, or use context caching for the shared prompt if the provider's minimum cacheable size allows it.
6. **Re-fit calibration** on call-like data once it exists, and fold the acoustic signals into the final confidence.

## 10. Reproducing the LLM results

```sh
python3 private/tools/make_val_tone.py                                     # builds private/data/val/tone (seed 20261016)
uv run python -m analyzer.llm private/data/val/tone/tune --fit-calibration # tune: metrics, cost, calibration fit
uv run python -m analyzer.llm private/data/val/tone/test                   # test: score once
uv run python -m analyzer.llm private/data/example                         # the 3 example calls
```

Needs `GEMINI_API_KEY` (paid tier) in `private/.env`. Model ID, prompt version (`v1`), prices and calibration points are constants at the top of `src/analyzer/llm.py`. Per-clip judgements, including token counts and cost, are written to `out/llm/<model>.jsonl`.
