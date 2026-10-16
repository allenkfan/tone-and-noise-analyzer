"""Model weights from the Hugging Face Hub, pinned to commits.

    python src/analyzer/weights.py      # download all of them into the Hugging Face cache

The Dockerfile runs this before it copies the code, so the download layer stays cached until a
pin changes. This file imports nothing from the package for that reason.
"""

from dataclasses import dataclass

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class Weights:
    repo: str
    revision: str
    files: tuple[str, ...]  # the first one is the ONNX graph; the rest sit next to it

    def path(self) -> str:
        """Local path of the first file, downloading every file on first use."""
        paths = [hf_hub_download(self.repo, f, revision=self.revision) for f in self.files]
        return paths[0]


VAD = Weights("istupakov/silero-vad-onnx", "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
              ("silero_vad.onnx",))  # Silero VAD 6.2, MIT
TAGGER = Weights("pranjal-pravesh/PANNs_CNN14_ONNX", "bafe3efef3ea1ea7577ee915ac1407029eef339b",
                 ("Cnn14_16k.onnx", "Cnn14_16k.onnx.data"))  # PANNs CNN14 16 kHz, AudioSet, MIT
DNSMOS = Weights("TigreGotico/dnsmos-onnx", "27691a53aa069b27be6ac957013d43b3c442da9d",
                 ("sig_bak_ovr.onnx",))  # Microsoft DNSMOS P.835, MIT
ALL = (VAD, TAGGER, DNSMOS)

if __name__ == "__main__":
    for w in ALL:
        print(w.path())
