"""Compare predictions with the labels from a manifest."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from analyzer.schema import FIELDS, TONES

if TYPE_CHECKING:
    from analyzer.batch import Row

# Free text and the confidence score have no exact-match accuracy.
SCORED_FIELDS = [f for f in FIELDS if f not in ("background_noise_type", "confidence")]


@dataclass
class Scores:
    n: int
    accuracy: dict[str, float]
    tone_macro_f1: float
    tone_confusion: list[list[int]]  # rows: labeled tone, columns: predicted tone, in TONES order

    def to_markdown(self) -> str:
        lines = [f"**Scored {self.n} labeled clips** · tone macro-F1 {self.tone_macro_f1:.3f}", "",
                 "| Field | Accuracy |", "| --- | --- |"]
        lines += [f"| {f} | {acc:.3f} |" for f, acc in self.accuracy.items()]
        lines += ["", "Tone confusion (rows: label, columns: prediction)", "",
                  "| | " + " | ".join(TONES) + " |", "|---" * (len(TONES) + 1) + "|"]
        lines += [f"| {t} | " + " | ".join(map(str, row)) + " |" for t, row in zip(TONES, self.tone_confusion)]
        return "\n".join(lines)


def score(rows: Sequence["Row"]) -> Scores | None:
    pairs = [(row.result.model_dump(), row.label) for row in rows if row.result and row.label]
    if not pairs:
        return None

    accuracy = {}
    for f in SCORED_FIELDS:
        matches = [pred[f] == label[f] for pred, label in pairs if f in label]
        if matches:
            accuracy[f] = sum(matches) / len(matches)

    index = {t: i for i, t in enumerate(TONES)}
    confusion = [[0] * len(TONES) for _ in TONES]
    for pred, label in pairs:
        if label.get("emotional_tone") in index:
            confusion[index[label["emotional_tone"]]][index[pred["emotional_tone"]]] += 1

    return Scores(len(pairs), accuracy, _macro_f1(confusion), confusion)


def _macro_f1(confusion: list[list[int]]) -> float:
    """Macro-F1 over the tones that appear as a label or a prediction."""
    f1s = []
    for i in range(len(confusion)):
        tp = confusion[i][i]
        labeled = sum(confusion[i])
        predicted = sum(row[i] for row in confusion)
        if labeled or predicted:
            f1s.append(2 * tp / (labeled + predicted))
    return sum(f1s) / len(f1s) if f1s else 0.0
