# Tone and Noise Analyzer

Classifies the emotional tone of call audio and flags background noise, audio-quality problems, speaker overlap and long silences. It returns one structured JSON result per clip.

## Repository layout

```
.
├── README.md
└── private/        # gitignored: a clone of the private companion repo
    ├── data/       # labeled audio
    ├── doc/        # project brief
    └── .env        # endpoints and credentials
```

## Setup

```sh
git clone https://github.com/allenkfan/tone-and-noise-analyzer.git
cd tone-and-noise-analyzer
git clone https://github.com/allenkfan/tone-and-noise-analyzer-private.git private
```

Audio, labels and credentials go only in `private/`. They never belong in this repo.
