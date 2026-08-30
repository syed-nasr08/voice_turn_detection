"""Build the turn-detection dataset.

Sources:
  1. DailyDialog (parquet mirror: pixelsandpointers/better_daily_dialog)
     - COMPLETE:   full utterances (each utterance in the corpus is a full turn)
     - INCOMPLETE: synthetic truncations of those utterances, biased toward cut
       points that guarantee incompleteness (after conjunctions, prepositions,
       articles, auxiliaries) plus some random mid-utterance cuts.
  2. Curated hard cases (data/hard_cases.py), expanded with random digits/names.

STT realism: a fraction of every bucket is "ASR-ified" (lowercased, punctuation
stripped) so the model doesn't lean solely on punctuation.

Splits: DailyDialog's own train/validation/test splits are disjoint sets of
dialogues, so no conversation leaks across splits. Both the complete utterance
and any truncation of it always land in the same split. Hard-case templates are
split 80/10/10 at the *template* level for the same reason.

Output: data/dataset/{train,validation,test}.jsonl
        fields: text, label (1=complete, 0=incomplete), source
"""

import json
import random
import re
import string
from pathlib import Path

from datasets import load_dataset

import hard_cases

random.seed(1337)

OUT_DIR = Path(__file__).parent / "dataset"

# Words that, when an utterance ends with them, guarantee the turn is incomplete.
CONTINUATION_WORDS = set("""
a an the and or but so because if when while that which who whose to of in on at
by for with from into over under about between during before after my your his
her its our their this these those very really quite than as not no do does did
is are was were be been being have has had will would can could shall should may
might must i'm it's he's she's we're they're i've we've you've i'll we'll you'll
""".split())

NAMES = ["john", "sarah", "michael", "emma", "david", "olivia", "james", "priya",
         "carlos", "yuki", "fatima", "liam"]
CITIES = ["Boston", "Denver", "Austin", "Seattle", "Chicago", "Miami",
          "Portland", "Atlanta"]
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine"]


# --------------------------------------------------------------------------- #
# Text cleaning (DailyDialog has spaced-out punctuation and curly quotes)
# --------------------------------------------------------------------------- #
def detokenize(text: str) -> str:
    t = text.strip()
    t = t.replace(" ’ ", "'").replace("’", "'")
    t = re.sub(r"\s+([,.!?;:%])", r"\1", t)
    t = re.sub(r"\s+'", "'", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def asr_ify(text: str) -> str:
    """Simulate punctuation-less, lowercased streaming STT output."""
    t = text.lower()
    t = t.replace("'", "'")
    t = re.sub(rf"[{re.escape(string.punctuation.replace(chr(39), ''))}]", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def strip_trailing_punct(words):
    """Remove punctuation glued to the last word so truncations look unfinished."""
    if words:
        words = words[:-1] + [words[-1].rstrip(string.punctuation)]
    return [w for w in words if w]


# --------------------------------------------------------------------------- #
# Synthetic truncation
# --------------------------------------------------------------------------- #
def truncate(utterance: str):
    """Return an incomplete prefix of `utterance`, or None if not possible.

    60%: cut right after a continuation word (guaranteed incomplete).
    40%: random mid-cut (weakly noisy, mostly incomplete) — kept so the model
         can't learn the shortcut "incomplete iff ends in a function word".
    """
    words = utterance.split()
    if len(words) < 5:
        return None

    cont_positions = [
        i for i in range(2, len(words) - 1)
        if words[i].lower().strip(string.punctuation) in CONTINUATION_WORDS
    ]
    if cont_positions and random.random() < 0.6:
        cut = random.choice(cont_positions) + 1
    else:
        cut = random.randint(3, len(words) - 2)

    prefix = strip_trailing_punct(words[:cut])
    if len(prefix) < 2:
        return None
    text = " ".join(prefix)
    # Reject prefixes that accidentally look sentence-final.
    if text[-1] in ".!?":
        return None
    return text


# --------------------------------------------------------------------------- #
# Hard-case expansion
# --------------------------------------------------------------------------- #
def expand(template: str) -> str:
    name = random.choice(NAMES)
    spelled = " ".join(name.upper())
    out = template
    while "{d}" in out:
        d = random.choice(DIGIT_WORDS) if random.random() < 0.5 else str(random.randint(0, 9))
        out = out.replace("{d}", d, 1)
    out = out.replace("{name}", name)
    out = out.replace("{city}", random.choice(CITIES))
    out = out.replace("{spelling}", spelled)
    out = out.replace("{partial_spelling}", " ".join(name.upper()[: max(2, len(name) // 2)]))
    return out


def split_templates(templates):
    t = templates[:]
    random.shuffle(t)
    n = len(t)
    return {"train": t[: int(n * 0.8)],
            "validation": t[int(n * 0.8): int(n * 0.9)],
            "test": t[int(n * 0.9):]}


# --------------------------------------------------------------------------- #
def build():
    rows = {"train": [], "validation": [], "test": []}

    # ---- DailyDialog ----
    for split in rows:
        ds = load_dataset("pixelsandpointers/better_daily_dialog", split=split)
        for ex in ds:
            utt = detokenize(ex["utterance"])
            if not (2 <= len(utt.split()) <= 60):
                continue
            rows[split].append({"text": utt, "label": 1, "source": "dailydialog"})
            trunc = truncate(utt)
            if trunc:
                rows[split].append({"text": trunc, "label": 0, "source": "dailydialog_trunc"})

    # ---- Hard cases (split at template level, expanded with variations) ----
    per_template = {"train": 40, "validation": 5, "test": 5}
    for label, templates in [(1, hard_cases.COMPLETE), (0, hard_cases.INCOMPLETE)]:
        tsplits = split_templates(templates)
        for split, temps in tsplits.items():
            for tmpl in temps:
                variants = {expand(tmpl) for _ in range(per_template[split])}
                for v in variants:
                    rows[split].append({"text": v, "label": label, "source": "hard_case"})

    # ---- ASR-ify a fraction (punctuation dropout) ----
    for split in rows:
        for r in rows[split]:
            if random.random() < 0.4:
                r["text"] = asr_ify(r["text"])

    # ---- Dedup (keep first label seen; drop cross-label conflicts entirely) ----
    for split in rows:
        seen, conflict = {}, set()
        for r in rows[split]:
            key = r["text"].lower()
            if key in seen and seen[key] != r["label"]:
                conflict.add(key)
            seen.setdefault(key, r["label"])
        deduped, emitted = [], set()
        for r in rows[split]:
            key = r["text"].lower()
            if key in conflict or key in emitted:
                continue
            emitted.add(key)
            deduped.append(r)
        random.shuffle(deduped)
        rows[split] = deduped

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, data in rows.items():
        path = OUT_DIR / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        n_pos = sum(r["label"] for r in data)
        by_src = {}
        for r in data:
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        print(f"{split}: {len(data)} rows | complete={n_pos} incomplete={len(data)-n_pos} | {by_src}")


if __name__ == "__main__":
    build()
