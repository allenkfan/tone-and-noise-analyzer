# Results on the three example calls

These are predictions for the three labeled calls that came with the trial brief, compared with their golden labels (`labels.csv` in the brief's example batch).

| | |
| --- | --- |
| Run | 2026-10-16, on the test dashboard (`allen-videolytics/tone-and-noise-analyzer-dev`, CPU Basic) |
| Build | `65c77a6`: local acoustic models (Silero VAD, PANNs CNN14, DNSMOS) plus the Gemini 3.5 Flash-Lite audio judge |
| Input | the three calls uploaded **without** labels, so the run could not see the answers |
| Scoring | `analyzer.metrics.score`, the same code the dashboard uses for labeled batches |

**Read this as a smoke test, not a validation result.** Three clips can't support conclusions about accuracy: one clip moves a field's accuracy by 0.33. The validation results are in [technical-memo.md](technical-memo.md), section 4. These calls do matter, though: they are the only real production calls available, and the hidden test set comes from the same source.

## Prediction against golden label, per call

Each cell shows *predicted / golden*. ✓ means they match.

| Field | call_001.ogg | call_002.ogg | call_003.ogg |
| --- | --- | --- | --- |
| emotional_tone | neutral / upset ✗ | upset / neutral ✗ | neutral / satisfied ✗ |
| emotional_intensity | low / high ✗ | high / medium ✗ | low / medium ✗ |
| background_noise_present | false / false ✓ | true / true ✓ | true / true ✓ |
| background_noise_type | "" / "" ✓ | tv / TV ✓ (case only) | keyboard typing / sharp static ✗ |
| background_noise_severity | none / none ✓ | low / medium ✗ | medium / medium ✓ |
| audio_quality | clear / clear ✓ | clear / clear ✓ | clear / clear ✓ |
| speaker_overlap_present | false / false ✓ | false / true ✗ | false / true ✗ |
| long_silence_present | false / false ✓ | false / false ✓ | false / false ✓ |
| confidence | 0.389 / 0.82 | 0.389 / 0.82 | 0.398 / 0.82 |

`background_noise_type` is free text and isn't scored. `confidence` isn't scored either: the golden labels use 0.82 for every call, so it carries no information.

## Per-field accuracy

The placeholder column is the scaffold's constant baseline on the same three calls. It answers "neutral, low, no noise, none, clear, no overlap, no silence" for every clip.

| Field | Build 65c77a6 | Placeholder baseline |
| --- | --- | --- |
| emotional_tone | 0.000 (0/3) | 0.333 (1/3) |
| emotional_intensity | 0.000 (0/3) | 0.000 (0/3) |
| background_noise_present | 1.000 (3/3) | 0.333 (1/3) |
| background_noise_severity | 0.667 (2/3) | 0.333 (1/3) |
| audio_quality | 1.000 (3/3) | 1.000 (3/3) |
| speaker_overlap_present | 0.333 (1/3) | 0.333 (1/3) |
| long_silence_present | 1.000 (3/3) | 1.000 (3/3) |
| **Tone macro-F1** | **0.000** | **0.167** |
| Scored fields correct, all clips | 12 of 21 | 10 of 21 |

No call is correct on every scored field.

Tone confusion matrix (rows: golden label, columns: prediction):

| | neutral | satisfied | frustrated | upset | distressed |
| --- | --- | --- | --- | --- | --- |
| neutral | 0 | 0 | 0 | 1 | 0 |
| satisfied | 1 | 0 | 0 | 0 | 0 |
| frustrated | 0 | 0 | 0 | 0 | 0 |
| upset | 1 | 0 | 0 | 0 | 0 |
| distressed | 0 | 0 | 0 | 0 | 0 |

## What the comparison shows

- **Tone and intensity are wrong on all three calls.** The tones of call_001 and call_002 come out swapped: the upset call is judged neutral and the neutral call upset. The satisfied call is judged neutral. Intensity follows the tone errors: low where the label is high, high where it is medium.
- **Noise presence is right on all three**, and severity on two. On call_002 the type matches (TV) but the severity is one step low (low vs medium). On call_003, noise is detected at the right severity but described as "keyboard typing" where the label says "sharp static".
- **Overlap is missed on both calls labeled with overlap** (call_002 and call_003).
- **Audio quality and long silence are correct**, but all three labels are "clear" and "no silence", so these calls don't test either field.
- The tone, intensity, noise-type and overlap fields come from the Gemini judge. Noise presence, severity, audio quality and silence come from the acoustic models.

## Latency and cost

| Call | Audio | Processing | s per audio min | Gemini audio tokens | Gemini cost (logged) | Gemini $ per audio min |
| --- | --- | --- | --- | --- | --- | --- |
| call_001.ogg | 30.9 s | 4.42 s | 8.56 | 774 | $0.00079 | $0.00153 |
| call_002.ogg | 35.0 s | 3.95 s | 6.78 | 875 | $0.00077 | $0.00132 |
| call_003.ogg | 171.9 s | 17.68 s | 6.17 | 4299 | $0.00188 | $0.00066 |
| **All three** | **3.96 min** | **26.04 s** | **6.57** | | **$0.00344** | **$0.00087** |

- Processing ran on CPU Basic (2 vCPU) at about 9× real time. Each Gemini call took 1.4–2.4 s of the per-clip time.
- At the CPU Upgrade price ($0.03/hr), compute adds $0.00005–0.00007 per audio minute.
- All three calls are under the $0.003 per-audio-minute ceiling. The shorter calls cost more per minute because each clip carries a fixed per-call cost. See the memo's cost section.

## Predictions in the required output format

```csv
name,result_json
call_001.ogg,"{""emotional_tone"": ""neutral"", ""emotional_intensity"": ""low"", ""background_noise_present"": false, ""background_noise_type"": """", ""background_noise_severity"": ""none"", ""audio_quality"": ""clear"", ""speaker_overlap_present"": false, ""long_silence_present"": false, ""confidence"": 0.389}"
call_002.ogg,"{""emotional_tone"": ""upset"", ""emotional_intensity"": ""high"", ""background_noise_present"": true, ""background_noise_type"": ""tv"", ""background_noise_severity"": ""low"", ""audio_quality"": ""clear"", ""speaker_overlap_present"": false, ""long_silence_present"": false, ""confidence"": 0.389}"
call_003.ogg,"{""emotional_tone"": ""neutral"", ""emotional_intensity"": ""low"", ""background_noise_present"": true, ""background_noise_type"": ""keyboard typing"", ""background_noise_severity"": ""medium"", ""audio_quality"": ""clear"", ""speaker_overlap_present"": false, ""long_silence_present"": false, ""confidence"": 0.398}"
```

## Golden labels

The labels below come from `labels.csv`, supplied with the trial brief:

```csv
name,result_json
call_001.ogg,"{""emotional_tone"":""upset"",""emotional_intensity"":""high"",""background_noise_present"":false,""background_noise_type"":"""",""background_noise_severity"":""none"",""audio_quality"":""clear"",""speaker_overlap_present"":false,""long_silence_present"":false,""confidence"":0.82}"
call_002.ogg,"{""emotional_tone"":""neutral"",""emotional_intensity"":""medium"",""background_noise_present"":true,""background_noise_type"":""TV"",""background_noise_severity"":""medium"",""audio_quality"":""clear"",""speaker_overlap_present"":true,""long_silence_present"":false,""confidence"":0.82}"
call_003.ogg,"{""emotional_tone"":""satisfied"",""emotional_intensity"":""medium"",""background_noise_present"":true,""background_noise_type"":""sharp static"",""background_noise_severity"":""medium"",""audio_quality"":""clear"",""speaker_overlap_present"":true,""long_silence_present"":false,""confidence"":0.82}"
```

## Reproducing

- **On a dashboard:** upload the three calls with a manifest that has a `name` column. Include `result_json` to get the scores directly on the results page.
- **Locally:** run `uv run python -m analyzer.batch <folder>`. It needs `GEMINI_API_KEY` and prints the same per-field accuracy and confusion matrix.

Gemini output can vary between runs, so tone and intensity may differ slightly from the table above.
