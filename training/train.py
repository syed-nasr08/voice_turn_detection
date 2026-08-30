"""Fine-tune a small encoder for turn detection (complete vs incomplete).

Model: nreimers/MiniLM-L6-H384-uncased (22M params) — small enough for
single-digit-ms CPU inference after INT8 quantization, strong enough for
sentence-level classification.

Usage: .venv/bin/python training/train.py
Output: training/model_out/  (best checkpoint by validation macro-F1)
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset"
OUT = Path(__file__).resolve().parent / "model_out"
MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
MAX_LEN = 64
SEED = 1337


def load_split(name: str) -> Dataset:
    rows = [json.loads(l) for l in open(DATA / f"{name}.jsonl")]
    return Dataset.from_list(rows)


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "incomplete", 1: "complete"},
        label2id={"incomplete": 0, "complete": 1},
    )

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    train = load_split("train").map(tokenize, batched=True)
    val = load_split("validation").map(tokenize, batched=True)

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=256,
        learning_rate=3e-5,
        warmup_steps=600,  # ~10% of total steps (transformers 5.x dropped warmup_ratio)
        weight_decay=0.01,
        label_smoothing_factor=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        logging_steps=200,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train, eval_dataset=val,
        processing_class=tok, compute_metrics=metrics,
    )
    trainer.train()
    print("Best validation:", trainer.evaluate())

    trainer.save_model(str(OUT))
    tok.save_pretrained(str(OUT))
    print(f"Saved best model to {OUT}")


if __name__ == "__main__":
    main()
