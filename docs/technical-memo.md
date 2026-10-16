# Technical memo

Status 2026-10-16: delivered as is.

## 1. Summary

The analyzer is a hybrid. Deterministic signal measurements on the server decide the physical fields (`long_silence_present`, `audio_quality`, `background_noise_present`, `background_noise_severity`, `speaker_overlap_present`). One call per clip to an audio-capable LLM, Google's `gemini-3.5-flash-lite`, decides the subjective fields (`emotional_tone`, `emotional_intensity`, `background_noise_type`), with the measurements passed in as context. If the API is unavailable, the batch still completes with a documented fallback.

Headline numbers for the LLM half:

| | Value | Source |
| --- | --- | --- |
| Tone accuracy / macro-F1, held-out acted speech | 0.377 / 0.325 (chance 0.20) | CREMA-D test split, 300 clips, scored once |
| Tone on the 3 real example calls | 0 of 3 (intensity 0 of 3, and 1 to 2 of 3 at temperature 0); the placeholder baseline got 1 of 3. Overlap, now measured: 3 of 3 (the LLM got 1 of 3) | tone on builds 65c77a6 and be53f5c on the test dashboard; overlap measured locally on 6ac18eb; calls not used for tuning |
| API cost per audio minute | $0.00057 at 3-minute calls, $0.00116 at 30-second calls (19% and 39% of the $0.003 ceiling) | measured with the acoustic context, 300 clips |
| API latency | median 1.4 s per short clip; 1.8 to 3.0 s per real call (31 to 172 s long), about 1.8 s per audio minute | measured |

Tone accuracy is the weak point: on the three real calls the hybrid scores lower on tone than the constant placeholder. Section 4 explains why: the model reads emotion mainly from the words, and barely from the voice. Section 9 lists the follow-ups that address it.

## 2. Approaches considered

| Approach | What it is | Outcome |
| --- | --- | --- |
| Placeholder baseline | Every field "nothing detected", confidence 0 | Floor: tone accuracy 0.33 and macro-F1 0.17 on the 3 example calls |
| A. Deterministic per-field measurements, no tone model | Silence, quality, noise presence and severity from signal measurements | Kept for the physical fields: on a 300-clip synthetic test split, silence 0.95, noise presence 0.92, quality 0.84, severity 0.73; 11 of 12 physical answers right on the example calls (section 4.2) |
| B. Audio LLM end to end | One Gemini call returns every field | Not used alone: the brief scores quality and noise separately and warns against inferring noise from poor quality; measurements are more reliable for those |
| **C. Hybrid (selected)** | A for the physical fields, the LLM only for the four subjective fields, with A's measurements in the prompt | Selected: best available accuracy on every field without new model training, inside the cost ceiling |
| D. Local ASR + text model + speech-emotion model | faster-whisper transcript, text sentiment, emotion2vec, small fusion classifier | Not built; 2 to 3 days and needs call-like training data. Part of it is follow-up 1 |
| E. Local speech-emotion model + acoustic features only | No transcript | Not built; expected weakest on satisfied versus neutral |

Other external models were ruled out on cost or capability: OpenAI's audio models and Whisper cost $0.006 per minute or more; Claude takes no audio input, so it would need local speech recognition first; Gemini models above Flash-Lite cost two to three times as much (section 5) and did worse in our test (section 4).

## 3. Final architecture

```
audio file ──ffmpeg──► float32, 16 kHz
                         │
                         ├─► acoustic.measure ──► silence, overlap, quality, noise presence/severity, noise tags (acoustic)
                         │                              │ as JSON context
                         └─► llm.judge ◄────────────────┘
                               │ 16 kHz mono Opus, sent inline to gemini-3.5-flash-lite
                               ▼
                         tone, intensity, noise type, confidence
                               │
                         pipeline.analyze merges both into one Result
```

`llm.judge` (`src/analyzer/llm.py`):

- **Request.** The clip is averaged to mono and re-encoded to 16 kHz Opus in memory (24 kbps; the API bills per second of audio, not per byte) and sent inline. The Files API, which stores uploads, is never used, and nothing is written to disk.
- **Prompt.** The system prompt restates the brief's definitions in our own words. It tells the model to judge the **customer**, whom it identifies from context (the agent greets, names the company, follows a script), since both voices share one channel; not to infer frustration or distress from loudness alone; and not to treat line-quality problems as background noise. The measurements arrive as one line after the audio: whether noise is present and the top three sound-event tags, as hints for the noise type. Silence, quality and severity are left out because they're already decided. When the measurements say no noise, the noise type is forced to an empty string.
- **Structured output.** Gemini's JSON-schema mode with the enum values from the brief and short keys (output tokens cost eight times input tokens), validated again with Pydantic. A one- or two-sentence rationale comes first in the schema so the model states its evidence before the labels; the rationale is logged only at DEBUG and never written to the CSV.
- **Thinking** is held at the model's minimum (`minimal`). Thought tokens bill as output at $2.50 per million and would dominate the cost.
- **Temperature 0**, for reproducibility. On the tune split it kept accuracy (0.377 against 0.390 at the default temperature, paired McNemar p = 0.61) and the calibration still fits (ECE 0.030). On three repeats of the example calls, answers were identical on two calls; the third still varied, because Gemini is not fully deterministic even at 0. On the test dashboard, two runs of the released build over the example calls differed on 2 of 3 calls (tone on one, intensity on another).
- **Reliability.** 90 s timeout; up to 3 attempts with exponential backoff on timeouts, 408, 429 and 5xx; one retry if the reply fails the schema. Any remaining failure raises `LLMUnavailable`, and `pipeline.analyze` falls back to neutral tone, low intensity, the acoustic noise-type guess and confidence 0.2; overlap and the other physical fields still come from the measurements. After 3 consecutive failures, a breaker skips the API for 120 s, so a batch during an outage finishes in seconds instead of waiting out every timeout. With no API key configured, the app starts and runs entirely on the fallback.

`acoustic.measure` (`src/analyzer/acoustic.py`), all on the CPU through onnxruntime, with weights pinned by commit hash and downloaded at image build time:

- **Voice activity.** Silero VAD (MIT) gives a speech mask per 32 ms frame with hysteresis and minimum-duration rules. The mask drives everything else: the longest non-speech run (`long_silence_present` at 11 s or more), active speech level, and which frames count as pauses.
- **Noise.** A speech-to-noise ratio from the pause energy (median, so a ringback tone doesn't dominate) and a minimum-statistics floor under speech, which still sees noise that call-side suppression mutes in the pauses. A PANNs CNN14 tagger (MIT, AudioSet's 527 classes) runs on consecutive 10 s windows; its classes are mapped to about 35 plain-word noise groups (TV or radio, music, keyboard typing, static, road noise, and so on), with the speaker's own voice, telephone signalling and room acoustics mapped to "not noise". Noise is present when the strongest noise group scores 0.25 or more, or the speech-to-noise ratio is under 20 dB, taking the worse of the floor under speech and the pause estimate (the latter only when the pauses are above -50 dBFS). Severity comes from the ratio alone: medium at 18 dB or less, high at 6 dB or less. The top groups go to the LLM as hints for the free-text noise type.
- **Quality.** Four independent faults, none of which read the noise measures: flattened speech peaks (clipping), bandwidth of the speech spectrum, active speech level, and short dropouts inside speech (packet loss). Any severe fault, or two slight ones, is severely impaired. DNSMOS P.835 is also computed and passed to the LLM, but no rule uses it.
- **Overlap.** pyannote segmentation-3.0 (MIT, 6 MB as ONNX) scores half-overlapping 10 s windows in a powerset of up to three local speakers; the probability of the two-speaker classes, averaged across windows, marks overlap frame by frame. A clip has overlap when the longest stretch is at least 0.4 s and overlap covers between 0.5% and 20% of the speech. The floor means a single flicker doesn't count. The ceiling separates crosstalk, which comes in bursts, from a TV, radio or crowd behind the caller, which overlaps nearly all of the speech. The LLM's own overlap answer is still requested but not used: it was false on every one of 600 validation clips and 27 diagnostic runs.
- **Merge.** `pipeline.analyze` takes silence, overlap, quality, noise presence and severity from the measurements, and tone, intensity and noise type from the LLM. If the measurements say no noise, the type is an empty string regardless of the LLM. Confidence is the LLM's calibrated value (section 4.3), or 0.2 on the fallback path.

## 4. Validation

### 4.1 Tone and intensity

**Data.** No call-like corpus with tone labels is available under a usable license this week (MSP-Podcast needs an institutional signature; IEMOCAP needs a USC agreement). We built a validation set from CREMA-D (ODbL 1.0 / DbCL 1.0): 7,442 clips of 91 actors speaking 12 fixed sentences in six emotions, each rated by crowd listeners from the voice alone. `private/tools/make_val_tone.py` (seed 20261016) builds it:

- Only clips whose voice-only majority vote matches the acted emotion are kept, so each label is what listeners actually heard.
- Mapping to the brief's tones, which is lossy: happy → satisfied; neutral → neutral; anger → upset, except anger rated low-intensity → frustrated (a proxy: acted corpora have no "frustrated", and the brief defines it as annoyance without strong anger); fear or sadness rated 65 or more out of 100 for intensity → distressed; disgust and milder fear or sadness are dropped.
- Intensity comes from the listeners' mean rating: below 55 low, 55 to 70 medium, 70 and above high. Neutral clips carry no intensity label.
- Every clip passes through a telephone chain (300 to 3,400 Hz band-pass, 16 kHz mono, Opus 16 kbps).
- **Leakage control.** Actors are split 61 / 30 into tune and test, stratified by sex, so no voice appears in both. Each split holds 60 clips per tone (300 clips). Prompts and calibration were chosen on tune only; test was scored once, at the end. The three labeled example calls were never used for tuning.

**Metric.** Tone accuracy and macro-F1 over the five tones; intensity accuracy on clips with an intensity label. Unless a run says otherwise, the validation runs below sent the audio without the acoustic context that production adds, so they measure the judge alone.

**Tune split** (300 clips), used to choose the model and prompt:

| Model, prompt | Tone accuracy | Tone macro-F1 | Intensity accuracy |
| --- | --- | --- | --- |
| gemini-3.5-flash-lite, v1 (selected) | 0.390 | 0.356 | 0.421 |
| gemini-3.5-flash-lite, v2: adds "judge the voice as much as the words" | 0.383 | 0.341 | 0.442 |
| gemini-3.6-flash, v1 | 0.320 | 0.268 | 0.396 |

These tune numbers and the test numbers below were measured with prompt v1 and without the acoustic context. The deployed prompt v6 (short keys, a one-line acoustic summary) at temperature 0 was checked on tune with the acoustic context, as production sends it: tone accuracy 0.370, macro-F1 0.329, intensity 0.433 (paired McNemar p = 0.86 against v1 at temperature 0). The test split was not rescored.

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

**Diagnostics on the example calls.** These used the calls only to compare variants that were built elsewhere; nothing was tuned on them. Three models (`gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.8-flash`) each ran the deployed prompt three times per call, with the acoustic measurements as context, as in production.
- **Speaker:** every rationale picked the caller as the customer, not the agent.
- **What drives the errors:** none of the rationales blamed the background sound or the scene for the customer's emotion. The errors follow the same pattern as on CREMA-D. The model weighs what the customer says above how they say it, so an offensive word said flatly reads as frustration or anger, and polite wording said tensely reads as neutral.
- **Models:** none of the three got the labeled tone on more than one call, so the stronger models offer no clear gain at their higher price.
- **Stability:** at the models' default temperature, the same call changed tone between repeats in 5 of 9 model-call pairs. This is why the deployed judge runs at temperature 0 (section 3).
- **Overlap:** the LLM never reported it (0 of 27 answers), although two calls are labeled with overlap. That is why the field now comes from the overlap detector in the measurements (section 4.2).
- **Release build at temperature 0** (be53f5c, two runs on the test dashboard): tone still 0 of 3; the call labeled satisfied now reads frustrated rather than neutral; intensity 2 of 3, then 1 of 3.
- **Prompt v6** (9bc2bf5, two runs on the test dashboard): tone 0 of 3 and intensity 1 of 3, with identical tone and intensity in both runs; the crackle on the third call is now named "static"; other fields unchanged.
- **Release build v1.1** (prompt v6 with the overlap detector, two runs on the test dashboard): tone 0 of 3 and intensity 1 of 3, identical in both runs; the call labeled satisfied reads neutral again. Only the noise-type wording varied between runs.

**Overlap and noise type.** CREMA-D clips are single-speaker and clean, so they test only false positives: the LLM said no overlap on all 600 clips. No call-like data with overlap or noise-type labels exists here. The synthetic two-speaker set in section 4.2 has overlap in 30% of clips and noise with a known source class in 60%, but it was used only to tune the measurements; the LLM was not run on it because of cost, so the LLM's overlap and noise type are validated only on the example calls (overlap 1 of 3, noise type right on 1 of 2 noisy calls; `docs/example-calls-results.md`). Because the LLM never reported overlap, the field now comes from the detector in the measurements: 0.787 on the synthetic test split (section 4.2) and 3 of 3 on the example calls.

### 4.2 Physical fields

**Data.** No labeled call corpus covers silence, quality and noise, so we built one where every label comes from construction. `private/tools/make_val_physical.py` (seed 20261016) makes two-actor conversations of 20 to 50 s from CREMA-D sentences, then adds, independently of each other: a long gap of 11 to 20 s in 25% of clips (3 to 7 s pauses in another 30% as hard negatives); overlapping turns in 30%; background noise in 60% (ESC-50 events, MUSAN music and ambience, babble, a TV-like mix, generated hiss, hum and crackle) at 20, 10, 5 or 0 dB speech-to-noise, labeled low, medium, medium, high; and a quality tier (clear 50%, slightly impaired 30%, severely impaired 20%) from low level, clipping, low-pass, low-bitrate Opus or dropped frames. Every clip then passes a comfort-noise floor 40 to 55 dB under speech (labeled no noise) and either wideband Opus or a G.711 telephone band, both labeled clear. 300 tune and 300 test clips (3.5 hours each), split by CREMA-D actor, ESC-50 fold and MUSAN file. Audio is not committed; the generator rebuilds it from the seed.

**Method.** Thresholds were set by coordinate descent per field on the tune split, scoring from cached measurements, ties going to the middle of the best plateau; two values (the tag threshold, 0.12 to 0.25, and the pause-floor gate, -70 to -50 dBFS) were then moved along flat stretches of the curve, within 0.01 of the best tune accuracy, for more margin against false noise. The test split was measured and scored once. The three example calls were not used to set any threshold; one observation from call_001 (a mild clipping signature on a call labeled clear) informed how strong the synthetic clipping tier is.

**Results, test split, 300 clips.**

| Field | Accuracy |
| --- | --- |
| long_silence_present | 0.950 |
| background_noise_present | 0.917 |
| audio_quality | 0.837 |
| background_noise_severity | 0.733 |
| speaker_overlap_present | 0.787 (always false: 0.687) |

Quality never flags a clean clip (152 of 152 clear clips predicted clear). The errors are: quality, slightly impaired called clear (26 of 97), 22 of them low-bitrate Opus, which no measure detects; noise presence, 18 misses of 190 noisy clips (10 at the faintest 20 dB tier, 6 at 10 dB) and 7 false alarms on clean clips from music, household, animal and bird tags; severity, medium called low (23 of 110), 18 of them intermittent ESC-50 events such as barks and knocks, because both speech-to-noise estimates use a median or a minimum and so measure the quiet between events (speech-like noise such as TV has the same effect: it reads as speech); silence, 11 false positives, 8 of them under 0 to 5 dB noise where the VAD loses the foreground speech, and 4 misses where a TV or babble background keeps the VAD on through the gap; overlap, 26 false alarms and 38 misses. Most misses are overlaps the model measures at about half their constructed length, some under 0.4 s; 7 are real overlaps on TV or babble backgrounds, which the 20% ceiling gives up (4 of 11 found there, against 9 of 11 without it) in exchange for far fewer false alarms there (4 of 25, against 20 of 25). The overlap thresholds were swept after the others were locked, by the same procedure. The test split was scored once without the ceiling (0.747) and once with it (0.787); the ceiling was chosen on the tune split.

Confusion matrices (rows: label, columns: prediction):

| Silence | false | true |
| --- | --- | --- |
| false | 215 | 11 |
| true | 4 | 70 |

| Quality | clear | slightly_impaired | severely_impaired |
| --- | --- | --- | --- |
| clear | 152 | 0 | 0 |
| slightly_impaired | 26 | 62 | 9 |
| severely_impaired | 5 | 9 | 37 |

| Noise present | false | true |
| --- | --- | --- |
| false | 103 | 7 |
| true | 18 | 172 |

| Severity | none | low | medium | high |
| --- | --- | --- | --- | --- |
| none | 103 | 7 | 0 | 0 |
| low | 10 | 25 | 5 | 0 |
| medium | 8 | 23 | 71 | 8 |
| high | 0 | 5 | 14 | 21 |

| Overlap | false | true |
| --- | --- | --- |
| false | 180 | 26 |
| true | 38 | 56 |

**Real calls.** On the three example calls, 14 of the 15 physical-field answers match the labels, overlap included (3 of 3, against 1 of 3 from the LLM; longest measured stretches 0.24 s on call_001, labeled false, and 0.57 s and 1.47 s on the two calls labeled true). The miss is severity on call_002, whose TV is predicted low and labeled medium, the same medium-to-low pattern as on the synthetic set.

### 4.3 Confidence calibration

The model reports its own confidence. On the tune split it said 0.80 to 0.95 for every clip, while accuracy was flat at about 0.25 to 0.40 across that range: overconfident and barely informative. We fitted an isotonic map (pool-adjacent-violators) from raw confidence to observed tone accuracy on tune (`python -m analyzer.llm … --fit-calibration`) and apply it in `judge`:

| Raw | 0.80 or less | 0.85 | 0.90 | 0.95 and above |
| --- | --- | --- | --- | --- |
| Calibrated | 0.20 | 0.31 | 0.38 | 0.41 |

This is the fit for the deployed prompt (v6, temperature 0, with the acoustic context). Points backed by fewer than 5 clips are merged into their neighbour. Expected calibration error (10 bins) on tune before calibration: 0.539. The earlier fit for prompt v1 (0.25 to 0.40) scored 0.029 on test; the v6 fit was not scored on test. The calibration was fitted on acted speech and transfers to calls only approximately. It is per model; another model ID reports raw confidence until it is fitted.

**Ambiguous tone.** When the customer's tone shifts, the prompt asks for the most prominent and sustained one; borderline cases should get lower confidence, but the raw scores show the model rarely does that, which is why calibration matters. When the API fails, confidence is 0.2.

## 5. Cost

### 5.1 API (Gemini)

Measured over the 600 validation clips and the 3 example calls, using paid-tier standard prices as of 2026-10-16:

| | gemini-3.5-flash-lite |
| --- | --- |
| Price | $0.30 per million input tokens (text and audio alike), $2.50 per million output tokens including thinking |
| Audio tokens | 1,512 per audio minute (about 25 per second) → **$0.00045 per audio minute** |
| Fixed per call | about 655 prompt-text tokens (system prompt plus the one-line acoustic summary) + about 63 output tokens, no thinking tokens → **$0.00035 per clip** |
| Cost per audio minute | $0.00045 + $0.00035 / clip length in minutes |

| Clip length | 10 s | 30 s | 60 s | 3 min |
| --- | --- | --- | --- | --- |
| $ per audio minute | 0.00258 | 0.00116 | 0.00081 | 0.00057 |
| Share of the $0.003 ceiling | 86% | 39% | 27% | 19% |

The fixed part dominates short clips. These figures were measured with prompt v6 and the acoustic context, as deployed. Before v6 the context was a JSON dump and the keys were long, and the fixed part was $0.00053 to $0.00058 per clip (measured on the dev Space). On the dev Space with v6, short clips cost $0.00037 to $0.00049 per call, and the three example calls $0.00299 for 3.96 audio minutes, $0.00075 per audio minute (prompt v1: $0.00087). The API cost alone still exceeds the $0.003 ceiling for clips shorter than about 8 seconds (follow-up 5). The example calls run 31 to 172 s; the hidden set's clip lengths are unknown. Comparison, also measured: `gemini-3.6-flash` costs $0.00075 fixed per clip and $0.00113 per audio minute ($0.00138 per minute at 3-minute calls, $0.00263 at 30 seconds), at introductory prices that double on 2027-01-01. Gemini's batch mode would halve the API cost, but its turnaround (target 24 hours) doesn't suit a dashboard where the evaluator waits for results. `gemini-2.5-flash-lite`, cheaper on paper ($0.10 text, $0.40 output), is no longer available to new API users.

### 5.2 Compute and hosting

The measurements run four ONNX models on the CPU (Silero VAD, PANNs CNN14, DNSMOS, pyannote segmentation) plus signal measures. Compute cost per audio minute = hourly rate / 60 × processing time per audio minute, at full utilization:

| Hosting | Rate | Acoustic only (RTF 0.05) | Hybrid, sequential, incl. API wait (RTF 0.085) |
| --- | --- | --- | --- |
| Hugging Face Spaces, CPU Upgrade, 8 vCPU | $0.03/h | $0.000026 | $0.00004 |
| Generic 8 vCPU cloud VM (assumption, about $0.35/h) | $0.35/h | $0.00029 one worker, $0.00007 with four | $0.0005 one worker |

Spaces hosting also carries a fixed $9/month PRO fee, which matters only at low volume (it equals $0.003 per minute at about 3,000 audio minutes a month). The deployed Space bills for uptime, not per batch, so the honest production figure is the cloud-VM row. Total per audio minute for 3-minute calls is therefore about $0.0006 on Spaces and about $0.0011 on a generic VM with one worker, both under the $0.003 ceiling; the API term in section 5.1 dominates and the clip-length caveat there applies to the total.

## 6. Latency

API seconds per call, measured with 8 calls in flight on the validation sets and one call at a time on the example calls:

| Clips | Median | Max | Per audio minute |
| --- | --- | --- | --- |
| CREMA-D, about 2.5 s each (600) | 1.4 s | 2.8 s | n/a (dominated by fixed overhead) |
| Example calls, 31 to 172 s (3) | 2.2 s | 3.0 s | 1.8 s |

The batch runner processes files one at a time, so a 100-call batch spends about 3 to 4 minutes waiting on the API.

Decode and measurement, and the end-to-end figure, measured on call-length clips:

| Where | Seconds per audio minute | Notes |
| --- | --- | --- |
| Acoustic only, local Intel i5 laptop | 3.8, 4.1 with overlap | VAD 0.3, tagger 1.4, DNSMOS 1.8, overlap segmentation 0.25 (DNSMOS feeds no rule and is the cheapest cut) |
| Acoustic only, Spaces CPU Basic, 2 vCPU | 2.6 to 3.1 | real-time factor about 0.05 |
| Hybrid end to end, Spaces CPU Basic | 5.3 to 5.6 | with overlap detection and prompt v6 (6ac18eb, the 3 example calls, two runs); 4.9 to 5.3 on the release build be53f5c. A 180 s call takes about 14 to 17 s, a 60 s call about 6 s |

The first file after a restart adds about 1.1 s of model loading. Very short clips are dominated by per-file overhead (about 0.5 s each warm). A batch of 100 three-minute calls therefore finishes in about 27 minutes on the current single-worker setup.

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
- **Severity under intermittent and speech-like noise:** severity is a speech-to-noise ratio from medians and minimums, so the quiet between intermittent events, and TV or babble that reads as speech, both make medium noise look low (section 4.2, and call_002).
- **Codec and room artifacts:** low-bitrate codec artifacts, echo and reverberation are not detected by any quality rule; the field never over-flags clean audio but misses these impairments.
- **Long silence is VAD-only:** hold music or ringback past 11 s counts as silence, and heavy noise can hide speech from the VAD and create a false long silence.
- **Clipping versus limiters:** clipping is read from flattened speech peaks; an aggressive limiter or AGC in the telephony chain can look the same.
- **Overlap under background voices:** a TV, radio or crowd looks like a second talker. The 20% ceiling removes most of those false alarms, but real crosstalk on a call with a TV on is then often missed, and a heated call where the parties talk over each other for more than a fifth of the time would read as no overlap. The 0.4 s threshold is short, so long backchannels may count; the synthetic set has none to measure this.

**Follow-up improvements, in order of expected gain**

1. **Local speech-emotion model (built; published as an experimental alternative, not selected).** Tag `v1.1-speech-emotion` (branch `ser`) adds the Odyssey 2024 WavLM arousal/valence model (MIT) on CPU torch. Its scores go into Gemini's context, and a fixed rule raises Gemini's tone when the customer's voice is activated. It runs on the test dashboard, https://allen-videolytics-tone-and-noise-analyzer-dev.hf.space, which is frozen during evaluation. On the CREMA-D test split it scores tone accuracy 0.527 and macro-F1 0.491, against 0.377 and 0.325 for the single-call judge (prompt v1, the only version scored on test). On the three example calls tone stays at 0 of 3 and intensity is no better (0 of 3, then 1 of 3, against 1 of 3). It takes 33 to 36 s per audio minute on CPU Basic against 5.3 to 5.6, and about $0.0015 per audio minute in total, roughly double v1.1, because the longer answer costs more output tokens. The rule's cut points come from acted speech and need call-like data (item 3); the model also needs an ONNX export to keep torch out of the image.
2. **Separate the words from the voice in two Gemini calls (tested, not shipped).** Pass 1 reports the customer's words and their sentiment; pass 2 is given those words and judges only the delivery; a fixed table combines them. Measured on CREMA-D with the acoustic context: the voice-only judgement scored 0.287 on tune and 0.310 on test (chance 0.20), and the combined answer 0.307 and 0.313 (macro-F1 0.238 and 0.242), against 0.370 and 0.377 for the single call. On the example calls tone stayed at 0 of 3 in three repeats: the call with an insult still read as upset or frustrated, because the voice-only pass also described the voice as angry or tense. The fixed cost doubles, to $0.00078 per clip (39% of the ceiling at 3-minute calls, 82% at 30 s). Gemini doesn't hear emotion in the voice even when asked about nothing else, which leaves item 1 as the path. The code is on branch `two-pass`.
3. **Call-like validation data.** MSP-Podcast (natural speech, academic license with an institutional signature) or synthetic two-speaker dialogues with scripted emotions. Acted single sentences can't measure customer identification, lexical cues or overlap.
4. **Concurrency.** The batch runner is sequential and the API calls are network-bound. A small thread pool (4 to 8 calls in flight, within the paid tier's rate limits) would cut batch time by about that factor.
5. **Lower the fixed cost further for short clips.** Prompt v6 (short keys, a one-line acoustic summary in place of the JSON dump) already cut it from about $0.00055 to $0.00035 per clip with tone accuracy unchanged. Dropping the rationale or shortening the definitions brought it to about $0.0002, but tone accuracy fell to 0.28 to 0.34 (p ≤ 0.015). Context caching doesn't apply: Gemini's implicit cache needs a prefix of at least 4,096 tokens on these models, and the prompt is about 900. Below about $0.0003 per clip, the remaining choices are a cheaper model or one call per several clips.
6. **Overlap: tell call participants from background voices.** Done in part: overlap now comes from pyannote segmentation, with a 20% ceiling that treats near-constant overlap as a TV, radio or crowd (section 3). Next, compare the levels of the two speaker tracks so a quieter background voice is ignored without the ceiling, which would recover crosstalk on calls with a TV on and on heated calls, and validate on real two-party calls with backchannels.
7. **Make answers fully repeatable.** Temperature 0 (section 3) removed most of the variation between runs, but not all. Caching each clip's answer by audio hash would make reruns of the same file identical.
8. **Re-fit calibration** on call-like data once it exists, and fold the acoustic signals into the final confidence.

## 10. Reproducing the LLM results

```sh
python3 private/tools/make_val_tone.py                                     # builds private/data/val/tone (seed 20261016)
uv run python -m analyzer.llm private/data/val/tone/tune --acoustics --fit-calibration # tune: metrics, cost, calibration fit
uv run python -m analyzer.llm private/data/val/tone/test                   # test: score once
uv run python -m analyzer.llm private/data/example                         # the 3 example calls
```

Needs `GEMINI_API_KEY` (paid tier) in `private/.env`. Model ID, prompt version (`v6`), prices and calibration points are constants at the top of `src/analyzer/llm.py`. Per-clip judgements, including token counts and cost, are written to `out/llm/<model>.jsonl`.
