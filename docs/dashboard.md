# Using the dashboard

The dashboard takes a batch of call recordings plus a CSV manifest and returns one structured result per recording. You can review the results in the page and download them as a CSV. No command line is needed.

| Dashboard | URL | What runs there |
| --- | --- | --- |
| Test | https://allen-videolytics-tone-and-noise-analyzer-dev.hf.space | the latest build under test |
| Submission | https://allen-videolytics-tone-and-noise-analyzer.hf.space | the frozen release for evaluation (currently the placeholder build; the final model will be promoted here) |

The test dashboard runs the model under development: local acoustic models plus the Gemini audio judge. Its answers can change with every deploy. The submission dashboard still runs the **placeholder analyzer**, which gives every clip the same "neutral, nothing detected" answer with confidence 0.0, until the final model is promoted there. On the placeholder, test uploading, validation, error handling and downloads, not accuracy.

## 1. Sign in

1. Open the dashboard URL above directly. Don't use the huggingface.co/spaces/... page: it embeds the app in a frame, and the login cookie doesn't stick there.
2. Sign in with username `evaluator` and the password you were given. Team members can find it as `DASHBOARD_PASSWORD` in `private/.env`.

If the Space hasn't been used for a while, it may need about a minute to wake up before the sign-in page appears.

## 2. Prepare a batch

Put the audio files and **one CSV manifest** side by side in a single folder:

```text
evaluation_batch/
├── call_001.wav
├── call_002.mp3
├── call_003.ogg
└── labels.csv
```

The manifest needs a `name` column. A `result_json` column is optional:

```csv
name,result_json
call_001.wav,"{""emotional_tone"":""frustrated"",""emotional_intensity"":""medium"",...}"
call_002.mp3,
```

- `name` is the exact audio filename, including the extension. Matching is case-sensitive.
- `result_json` holds the expected result. When it's filled in, the dashboard scores its predictions against it. Leave it empty, or drop the column, for unlabeled audio.
- Spaces after commas, a byte-order mark, and extra columns are all fine. The manifest can have any `.csv` filename.
- Supported audio formats: WAV, MP3, OGG/Opus, FLAC, M4A/AAC, WebM.

## 3. Upload

Use **one** of the two tabs:

- **Folder:** click the upload box and pick the batch folder in the dialog. Every file in it is uploaded, and your browser may ask you to confirm. Dragging the folder itself onto the box doesn't work, because the box receives only the folder, not its contents. To drag, select the files *inside* the folder and drag those.
- **ZIP:** upload a `.zip` of the folder. A top-level folder inside the ZIP is fine, and macOS `__MACOSX` and dot-files are ignored.

If you fill in both tabs, the ZIP is used.

## 4. Analyze

Press **Analyze**. The page then shows three things in order:

1. **Validation report.** It shows how many files will be analyzed, which file is the manifest, and a list of problems found. A problem never stops the batch:

   | Message | Meaning |
   | --- | --- |
   | `listed in the manifest but not uploaded` | the CSV names a file that isn't in the batch; it's reported as skipped |
   | `not listed in the manifest; processed anyway` | an audio file has no CSV row; it's still analyzed |
   | `not an audio file; skipped` | for example, a `.txt` or `.pdf` in the folder |
   | `unsupported file type` | the CSV lists a file with an extension that isn't supported |
   | `No CSV manifest found; processing every audio file` | no `.csv` in the batch |
   | `result_json is not a JSON object; label ignored` | that row's label can't be parsed; the file is still analyzed |

2. **Progress.** The progress bar counts files, and the results table fills in one row per file as each finishes. Keep the tab open until the batch is done, because results live only in the page until you download them. Only one batch runs at a time. If someone else is running one, yours waits until theirs finishes.

3. **Results**, described in the next section.

## 5. Read the results

Each row of the **Results** table is one file:

| Column | Contents |
| --- | --- |
| `name` | the filename as listed in the manifest |
| `status` | `ok`; `failed` (the file couldn't be decoded or analyzed, with the reason in `error`); or `skipped` (not uploaded, or an unsupported type) |
| `error` | why the file failed or was skipped |
| `emotional_tone` … `confidence` | the result fields, as defined in the output schema |
| `audio_seconds`, `processing_seconds` | clip length and the time the analyzer took |

Below the table:

- **Results CSV:** click to download `predictions.csv`. It contains `name`, `result_json` (the full result, in the same format as the manifest), `status` and `error`, plus one column per field and the two timing columns. Filenames are kept exactly as they appeared in the manifest.
- **Summary line:** how many files were ok, failed and skipped, the total audio minutes, and processing seconds per audio minute.
- **Scores**, only when the manifest has labels: per-field accuracy, emotional-tone macro-F1, and a tone confusion matrix (rows are labels, columns are predictions). `background_noise_type` is free text and `confidence` is a score, so neither has an accuracy figure.

## 6. Run another batch

After a run, the uploaded audio is deleted from the server and the upload boxes are cleared. To run another batch, upload it and press **Analyze** again. Download the CSV before you start, because the new run replaces the results on the page.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The sign-in page keeps coming back | Open the `.hf.space` URL directly, not the huggingface.co Space page, and allow cookies for `hf.space` |
| "Upload a folder or a ZIP archive first." | Nothing was uploaded, or the upload hadn't finished when you pressed Analyze |
| "That file is not a valid ZIP archive." | Re-create the ZIP; only `.zip` is supported, not `.rar` or `.7z` |
| Every file is `skipped` | The manifest names don't match the uploaded files; check extensions and capitalization |
| One file is `failed: not a readable audio file` | That file is corrupt or not actually audio; the rest of the batch is unaffected |
| The page stops responding during a run | The Space may have restarted for a new deploy. Reload the page and run the batch again |

## Privacy

- Audio is processed on Hugging Face's servers, where the dashboard runs.
- On the test dashboard, each clip is also sent to Google's Gemini API to judge tone, intensity, noise type and overlap. See the external API disclosure in [technical-memo.md](technical-memo.md). The submission dashboard sends nothing to Gemini until the final model is promoted there.
- Uploaded files are deleted as soon as a batch finishes, and the app's file cache is cleared every hour.
- The logs record only filenames, statuses and timings, never audio or results.
- Usage analytics are turned off.

## Scripted use

The dashboard can also be driven from Python with `gradio_client`. This is useful for repeatable tests. See "Smoke test against a live Space" in [development.md](development.md).
