"""Export the fine-tuned model to ONNX and INT8-quantize it for CPU serving.

Usage: .venv/bin/python training/export_onnx.py
Output: serving/model/model.onnx        (fp32)
        serving/model/model.int8.onnx   (dynamic INT8 — the one we serve)
        serving/model/tokenizer files
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "training" / "model_out"
OUT = ROOT / "serving" / "model"
MAX_LEN = 64


class Wrapper(torch.nn.Module):
    """Fix the input signature to (input_ids, attention_mask) -> probs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return torch.softmax(logits, dim=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    wrapped = Wrapper(model)

    enc = tok(["my phone number is five five five"], return_tensors="pt",
              padding="max_length", max_length=16, truncation=True)
    torch.onnx.export(
        wrapped,
        (enc["input_ids"], enc["attention_mask"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["probs"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "probs": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    quantize_dynamic(
        str(OUT / "model.onnx"),
        str(OUT / "model.int8.onnx"),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    tok.save_pretrained(str(OUT))

    # ---- parity check: torch vs fp32 onnx vs int8 onnx ----
    tests = [
        "My phone number is five five five",
        "What time do you close?",
        "I want to cancel my",
        "yeah that works for me",
    ]
    enc = tok(tests, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        ref = wrapped(enc["input_ids"], enc["attention_mask"]).numpy()

    feed = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
    for name in ["model.onnx", "model.int8.onnx"]:
        sess = ort.InferenceSession(str(OUT / name), providers=["CPUExecutionProvider"])
        out = sess.run(["probs"], feed)[0]
        drift = np.abs(out - ref).max()
        print(f"{name}: max |p - p_torch| = {drift:.4f}")
        for t, p in zip(tests, out[:, 1]):
            print(f"   p(complete)={p:.3f}  {t!r}")

    for f in OUT.glob("model*.onnx"):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
