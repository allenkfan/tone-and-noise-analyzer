"""Speech emotion from the voice alone: arousal, dominance and valence from a WavLM model.

The model is 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes (MIT), the Odyssey 2024
challenge baseline trained on MSP-Podcast. It scores a stretch of speech from about 0 to 1
on arousal (calm to activated), dominance and valence (negative to positive). It runs on
4 s windows with a 2 s hop, since attention cost grows with the square of the length and
the model was trained on short utterances; a clip is summarised by percentiles over the
windows that contain sound.

The backbone is torchaudio's WavLM built from microsoft/wavlm-large's config, so no remote
code runs and no second checkpoint is downloaded; the pooling and regression head below are
copied from the model repo's pipeline_utils.py (MIT).
"""

import json
import logging
import threading
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger("analyzer")

SR = 16_000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0
MIN_WINDOW_SECONDS = 1.0  # a shorter clip or tail is scored as it is, down to this length
SILENT_DBFS = -50.0  # windows quieter than this are skipped
BATCH = 8

FUSE = True  # apply `fuse` in pipeline.merge; the context line goes to the LLM either way
GUARD = True  # never raise when the LLM reads the customer's words as positive

# Fusion with the LLM's answer (experiment 2). The voice says how activated the customer is;
# the LLM, which reads the words, keeps the polarity. Cut points on the 90th percentile of
# window arousal, chosen by grid search on the CREMA-D tune split (acted neutral speech sits
# near 0.27; real calls, agent included, read 0.6 and up, so these won't transfer as they are).
# Taking intensity from arousal lowered intensity accuracy on tune, and lowering "upset" at
# low arousal never helped, so neither is done.
AROUSAL_RAISE = 0.35  # neutral -> frustrated (medium)
AROUSAL_UPSET = 0.65  # frustrated -> upset


@dataclass
class SER:
    windows: int  # windows scored
    arousal_p50: float
    arousal_p90: float
    arousal_max: float
    valence_p10: float
    valence_p50: float
    dominance_p50: float

    def as_context(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def measure(samples: np.ndarray, sr: int, turns: list[tuple[float, float]] | None = None) -> SER | None:
    """Score one clip. `samples` is float32 with shape (channels, n) or (n,) at 16 kHz.

    With `turns` (start, end seconds), only windows inside those stretches are scored, so the
    summary describes that speaker. Returns None when no window has sound in it.
    """
    if sr != SR:
        raise ValueError(f"ser.measure needs {SR} Hz audio, got {sr}")
    mono = samples.mean(axis=0) if samples.ndim == 2 else samples
    if turns is None:
        scores = score_windows(mono)
    else:
        parts = [score_windows(mono[int(a * SR):int(b * SR)]) for a, b in turns]
        scores = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
    if len(scores) == 0:
        return None
    a, d, v = scores[:, 0], scores[:, 1], scores[:, 2]
    p = lambda x, q: float(np.percentile(x, q))  # noqa: E731
    return SER(len(scores), p(a, 50), p(a, 90), float(a.max()), p(v, 10), p(v, 50), p(d, 50))


def fuse(tone: str, intensity: str, ser: SER | None, words_sentiment: str = "",
         guard: bool | None = None) -> tuple[str, str, str]:
    """Adjust the LLM's (tone, intensity) with the voice's arousal. Returns (tone, intensity, rule).

    With the guard on, positive words block both raises: an activated voice saying thank you
    is more likely pleased than annoyed.

    | LLM tone   | arousal p90 >= AROUSAL_RAISE | >= AROUSAL_UPSET |
    | neutral    | frustrated, medium           | frustrated, medium |
    | frustrated | frustrated (kept)            | upset, intensity kept |
    | others     | kept                         | kept |
    """
    if ser is None:
        return tone, intensity, "llm"
    if (GUARD if guard is None else guard) and words_sentiment == "positive":
        return tone, intensity, "guarded"
    a = ser.arousal_p90
    if tone == "neutral" and a >= AROUSAL_RAISE:
        return "frustrated", "medium", "raised"
    if tone == "frustrated" and a >= AROUSAL_UPSET:
        return "upset", intensity, "raised"
    return tone, intensity, "kept"


def windows(mono: np.ndarray, gate: bool = True) -> list[np.ndarray]:
    """4 s windows with a 2 s hop; a clip shorter than one window is one window."""
    size, hop = int(WINDOW_SECONDS * SR), int(HOP_SECONDS * SR)
    if len(mono) <= size:
        out = [mono] if len(mono) >= MIN_WINDOW_SECONDS * SR else []
    else:
        out = [mono[i:i + size] for i in range(0, len(mono) - size + 1, hop)]
        tail = (len(mono) - size) % hop
        if tail >= MIN_WINDOW_SECONDS * SR:
            out.append(mono[-size:])
    if gate:
        out = [w for w in out if 20 * np.log10(np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-12) > SILENT_DBFS]
    return out


def score_windows(mono: np.ndarray, gate: bool = True) -> np.ndarray:
    """(windows, 3) array of arousal, dominance, valence."""
    import torch

    model, mean, std = _model()
    wins = windows(mono, gate)
    out = []
    with torch.inference_mode():
        for i in range(0, len(wins), BATCH):
            group = wins[i:i + BATCH]
            same = all(len(w) == len(group[0]) for w in group)
            batches = [group] if same else [[w] for w in group]
            for b in batches:
                x = torch.from_numpy((np.stack(b).astype(np.float32) - mean) / (std + 1e-6))
                out.append(model(x).numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


_lock = threading.Lock()
_loaded = None


def _model():
    global _loaded
    with _lock:
        if _loaded is None:
            _loaded = _load()
        return _loaded


def _load():
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    from analyzer.weights import SER as SER_WEIGHTS, WAVLM_CONFIG

    weights = Path(SER_WEIGHTS.path())
    config = json.load(open(weights.parent / "config.json"))
    backbone = json.load(open(WAVLM_CONFIG.path()))
    state = load_file(weights)
    model = SERModel(backbone, config["hidden_size"], config["classifier_hidden_layers"], config["num_classes"])
    model.load(state)
    model.eval()
    torch.set_grad_enabled(False)
    log.info("ser model loaded: %s@%s", SER_WEIGHTS.repo, SER_WEIGHTS.revision[:7])
    return model, float(config["mean"]), float(config["std"])


def _torch_modules():
    """The model's modules, defined on first use so importing this file doesn't import torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchaudio.models import wavlm_model
    from torchaudio.models.wav2vec2.utils.import_huggingface import transform_wavlm_encoder_state

    class AttentiveStatisticsPooling(nn.Module):
        # From pipeline_utils.py (MIT); every frame is valid because windows aren't padded.
        def __init__(self, dim):
            super().__init__()
            self.sap_linear = nn.Linear(dim, dim)
            self.attention = nn.Parameter(torch.empty(dim, 1))

        def forward(self, x):
            h = torch.tanh(self.sap_linear(x))
            w = F.softmax(torch.matmul(h, self.attention).squeeze(2), dim=1).unsqueeze(2)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
            return torch.cat((mu, rh), 1)

    class EmotionRegression(nn.Module):
        # From pipeline_utils.py (MIT), without dropout (inference only).
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super().__init__()
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                                   nn.ReLU(), nn.Dropout(0.0))])
            for _ in range(num_layers - 1):
                self.fc.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                             nn.ReLU(), nn.Dropout(0.0)))
            self.out = nn.Sequential(nn.Linear(hidden_dim, output_dim))

        def forward(self, x):
            for fc in self.fc:
                x = fc(x)
            return self.out(x)

    class SERModel(nn.Module):
        def __init__(self, cfg: dict, hidden: int, head_layers: int, classes: int):
            super().__init__()
            self.cfg = cfg
            self.ssl_model = wavlm_model(
                extractor_mode=f"{cfg['feat_extract_norm']}_norm",
                extractor_conv_layer_config=list(zip(cfg["conv_dim"], cfg["conv_kernel"], cfg["conv_stride"])),
                extractor_conv_bias=cfg["conv_bias"],
                encoder_embed_dim=cfg["hidden_size"],
                encoder_projection_dropout=0.0,
                encoder_pos_conv_kernel=cfg["num_conv_pos_embeddings"],
                encoder_pos_conv_groups=cfg["num_conv_pos_embedding_groups"],
                encoder_num_layers=cfg["num_hidden_layers"],
                encoder_num_heads=cfg["num_attention_heads"],
                encoder_num_buckets=cfg["num_buckets"],
                encoder_max_distance=cfg["max_bucket_distance"],
                encoder_attention_dropout=0.0,
                encoder_ff_interm_features=cfg["intermediate_size"],
                encoder_ff_interm_dropout=0.0,
                encoder_dropout=0.0,
                encoder_layer_norm_first=cfg["do_stable_layer_norm"],
                encoder_layer_drop=0.0,
                aux_num_out=None,
            )
            self.pool_model = AttentiveStatisticsPooling(hidden)
            self.ser_model = EmotionRegression(hidden * 2, hidden, head_layers, classes)

        def load(self, state: dict) -> None:
            """Load the checkpoint, renaming the Hugging Face WavLM keys to torchaudio's."""
            ssl = {k[len("ssl_model."):]: v for k, v in state.items() if k.startswith("ssl_model.")}
            sub = lambda prefix: {k[len(prefix):]: v for k, v in ssl.items() if k.startswith(prefix)}  # noqa: E731
            encoder = sub("encoder.")
            transform_wavlm_encoder_state(encoder, self.cfg["num_hidden_layers"])
            self.ssl_model.feature_extractor.load_state_dict(sub("feature_extractor."))
            self.ssl_model.encoder.feature_projection.load_state_dict(sub("feature_projection."))
            self.ssl_model.encoder.transformer.load_state_dict(encoder)
            self.pool_model.load_state_dict({k[len("pool_model."):]: v for k, v in state.items()
                                             if k.startswith("pool_model.")})
            self.ser_model.load_state_dict({k[len("ser_model."):]: v for k, v in state.items()
                                            if k.startswith("ser_model.")})

        def forward(self, x):
            hidden, _ = self.ssl_model(x)
            return self.ser_model(self.pool_model(hidden))

    return SERModel


def SERModel(*args, **kwargs):
    return _torch_modules()(*args, **kwargs)
