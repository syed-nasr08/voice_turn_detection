"""Evaluate the turn-detection model against baselines on the held-out test set.

Reports, for each system:
  - accuracy, per-class precision/recall/F1, macro-F1
  - ROC-AUC (model only)
  - precision/recall of COMPLETE at the high-confidence threshold (0.90),
    the number that governs "how often do we interrupt the user"
  - slice metrics by data source (real turns, truncations, hard cases)

Baselines:
  1. majority     — always predict COMPLETE
  2. heuristic    — COMPLETE iff text ends with terminal punctuation, or (when
                    no punctuation, i.e. raw STT) last word is not a
                    continuation word (conjunction/article/preposition/aux)

Usage: .venv/bin/python evaluation/evaluate.py [--model training/model_out]
"""

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from build_dataset import CONTINUATION_WORDS  # noqa: E402

HIGH_CONF = 0.90


def load_test():
    rows = [json.loads(l) for l in open(ROOT / "data" / "dataset" / "test.jsonl")]
    return rows


def heuristic_predict(text: str) -> int:
    t = text.strip()
    if t and t[-1] in ".!?":
        return 1
    last = t.split()[-1].lower().strip(string.punctuation) if t.split() else ""
    return 0 if last in CONTINUATION_WORDS else 1


@torch.no_grad()
def model_probs(rows, model_dir):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    for i in range(0, len(rows), 256):
        batch = [r["text"] for r in rows[i:i + 256]]
        enc = tok(batch, truncation=True, max_length=64, padding=True, return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
        probs.extend(p.cpu().tolist())
    return np.array(probs)


def report(name, y_true, y_pred, probs=None):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=["incomplete", "complete"], digits=4))
    print(f"accuracy={accuracy_score(y_true, y_pred):.4f}  macro-F1={f1_score(y_true, y_pred, average='macro'):.4f}")
    if probs is not None:
        print(f"ROC-AUC={roc_auc_score(y_true, probs):.4f}")
        hi = (probs >= HIGH_CONF).astype(int)
        print(f"@threshold {HIGH_CONF}: precision(COMPLETE)={precision_score(y_true, hi, zero_division=0):.4f} "
              f"recall(COMPLETE)={recall_score(y_true, hi):.4f} "
              f"(fires on {hi.mean():.1%} of inputs)")


def slice_report(rows, y_pred):
    print(f"\n--- accuracy by source ---")
    for src in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        yt = [rows[i]["label"] for i in idx]
        yp = [y_pred[i] for i in idx]
        print(f"{src:20s} n={len(idx):6d} acc={accuracy_score(yt, yp):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "training" / "model_out"))
    args = ap.parse_args()

    rows = load_test()
    y_true = np.array([r["label"] for r in rows])

    report("BASELINE 1: majority (always COMPLETE)", y_true, np.ones_like(y_true))

    heur = np.array([heuristic_predict(r["text"]) for r in rows])
    report("BASELINE 2: punctuation + function-word heuristic", y_true, heur)
    slice_report(rows, heur)

    probs = model_probs(rows, args.model)
    report(f"MODEL: {args.model}", y_true, (probs >= 0.5).astype(int), probs)
    slice_report(rows, (probs >= 0.5).astype(int))

    # worst errors, for qualitative inspection
    print("\n--- 10 most-confident mistakes ---")
    err = np.abs(probs - y_true)
    for i in np.argsort(-err)[:10]:
        print(f"p(complete)={probs[i]:.3f} label={y_true[i]} [{rows[i]['source']}] {rows[i]['text'][:90]!r}")


if __name__ == "__main__":
    main()
