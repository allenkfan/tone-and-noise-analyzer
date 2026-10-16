"""Write src/analyzer/audioset_groups.tsv: each AudioSet class mapped to a plain-words noise group.

    uv run python tools/make_audioset_groups.py

A class takes the group of the nearest of itself and its ontology ancestors that RULES names. An
empty group means "not background noise": the speaker's own voice and body sounds, telephone
signalling, room acoustics, distortion (a quality problem, not noise) and synthetic tones.

Sources: AudioSet class_labels_indices.csv (the tagger's output order) and ontology.json from
github.com/audioset/ontology (CC BY-SA 4.0).
"""

import csv
import io
import json
import urllib.request
from collections import deque
from pathlib import Path

LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
OUT = Path(__file__).resolve().parents[1] / "src" / "analyzer" / "audioset_groups.tsv"

IGNORE = ""
RULES = {
    # specific classes that would otherwise inherit a misleading group
    "Baby cry, infant cry": "baby crying",
    "Hubbub, speech noise, speech babble": "background chatter",
    "Walk, footsteps": "footsteps",
    "Run": "footsteps",
    "Shuffle": "footsteps",
    "Television": "TV or radio",
    "Radio": "TV or radio",
    "Static": "static or hiss",
    "White noise": "static or hiss",
    "Pink noise": "static or hiss",
    "Hiss": "static or hiss",
    "Crackle": "static or hiss",
    "Mains hum": "hum or buzz",
    "Hum": "hum or buzz",
    "Buzz": "hum or buzz",
    "Throbbing": "hum or buzz",
    "Vibration": "hum or buzz",
    "Rumble": "hum or buzz",
    "Distortion": IGNORE,
    "Sidetone": IGNORE,
    "Beep, bleep": IGNORE,
    "Clicking": "clicking",
    "Tick": "clicking",
    "Tick-tock": "clock ticking",
    "Clock": "clock ticking",
    "Mechanical fan": "fan or air conditioning",
    "Air conditioning": "fan or air conditioning",
    "Printer": "printer",
    "Typing": "keyboard typing",
    "Vacuum cleaner": "appliance noise",
    "Hair dryer": "appliance noise",
    "Blender": "appliance noise",
    "Microwave oven": "appliance noise",
    "Electric shaver, electric razor": "appliance noise",
    "Water tap, faucet": "running water",
    "Sink (filling or washing)": "running water",
    "Bathtub (filling or washing)": "running water",
    "Toilet flush": "running water",
    "Dog": "dog barking",
    "Bird": "birds",
    "Wind": "wind",
    "Rain": "rain",
    "Thunderstorm": "thunder or rain",
    "Siren": "siren",
    "Vehicle horn, car horn, honking": "car horn",
    "Doorbell": "doorbell",
    "Telephone": IGNORE,  # ringback, dial and busy tones, DTMF: the call itself, not the room
    "Motor vehicle (road)": "road noise",
    "Traffic noise, roadway noise": "road noise",
    "Non-motorized land vehicle": "road noise",
    "Rail transport": "train",
    "Aircraft": "aircraft",
    "Boat, Water vehicle": "boat",
    "Engine": "engine noise",
    "Tools": "construction or power tools",
    "Explosion": "popping or crackling",  # in call audio, line pops; not gunfire
    "Clapping": "applause",
    "Applause": "applause",
    "Cheering": "background chatter",
    # whole branches
    "Human voice": IGNORE,
    "Respiratory sounds": IGNORE,
    "Digestive": IGNORE,
    "Heart sounds, heartbeat": IGNORE,
    "Otoacoustic emission": IGNORE,
    "Hands": IGNORE,
    "Whistling": IGNORE,
    "Human locomotion": IGNORE,
    "Human group actions": "background chatter",
    "Animal": "animal sounds",
    "Music": "music",
    "Water": "running water",
    "Fire": "fire crackling",
    "Natural sounds": "outdoor ambience",
    "Vehicle": "road noise",
    "Alarm": "alarm",
    "Bell": "bells",
    "Mechanisms": "mechanical noise",
    "Domestic sounds, home sounds": "household sounds",
    "Sounds of things": "household sounds",
    "Other sourceless": IGNORE,  # sine waves, chirps, pulses: test tones and signalling
    "Silence": IGNORE,
    "Source-ambiguous sounds": "household sounds",
    "Acoustic environment": IGNORE,
    "Sound reproduction": IGNORE,
    "Noise": "background noise",
    "Channel, environment and background": IGNORE,
}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def main() -> None:
    labels = list(csv.DictReader(io.StringIO(fetch(LABELS_URL))))
    ontology = {n["id"]: n for n in json.loads(fetch(ONTOLOGY_URL))}
    parents: dict[str, list[str]] = {}
    for node in ontology.values():
        for child in node["child_ids"]:
            parents.setdefault(child, []).append(node["id"])

    def group(mid: str) -> str:
        queue, seen = deque([mid]), {mid}
        while queue:
            m = queue.popleft()
            name = ontology[m]["name"]
            if name in RULES:
                return RULES[name]
            for p in parents.get(m, []):
                if p not in seen:
                    seen.add(p)
                    queue.append(p)
        raise ValueError(f"no rule covers {ontology[mid]['name']}")

    lines = [f"{row['display_name']}\t{group(row['mid'])}" for row in labels]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} classes to {OUT}")


if __name__ == "__main__":
    main()
