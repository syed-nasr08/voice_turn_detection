"""Turn-detection inference service.

FastAPI + ONNX Runtime (INT8, CPU). One ONNX session per worker process.

Endpoints:
  POST /v1/turn   -> P(complete) + recommended silence timeout
  GET  /healthz   -> readiness (model loaded AND warmed up)
  GET  /livez     -> liveness
  GET  /metrics   -> Prometheus text format

Run:  uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 2
"""

import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from serving.policy import decide

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))
MODEL_FILE = os.environ.get("MODEL_FILE", "model.int8.onnx")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "en-minilm-l6-int8-v1")
MAX_LEN = 64

app = FastAPI(title="turn-detector", version=MODEL_VERSION)

_state = {"ready": False}
_metrics = {"requests": 0, "errors": 0, "latency_buckets": {}, "latency_sum": 0.0}
_BUCKETS = [1, 2, 5, 10, 25, 50, 100, 250, 1000]  # ms


class TurnRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=2000)
    language: Optional[str] = "en"     # advisory; logged, not used for routing
    context: Optional[str] = None      # reserved: agent's last utterance
    audio_b64: Optional[str] = None    # reserved: future audio model


class TurnResponse(BaseModel):
    probability_complete: float
    decision_hint: str
    recommended_timeout_ms: int
    model_version: str
    inference_ms: float


@app.on_event("startup")
def load_model():
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
    so.inter_op_num_threads = 1
    _state["session"] = ort.InferenceSession(
        str(MODEL_DIR / MODEL_FILE), sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(MAX_LEN)
    _state["tokenizer"] = tok
    for _ in range(10):  # warmup before reporting ready
        _predict("warm up the model with a sentence of plausible length okay")
    _state["ready"] = True


def _predict(text: str) -> float:
    enc = _state["tokenizer"].encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    probs = _state["session"].run(["probs"], {"input_ids": ids, "attention_mask": mask})[0]
    return float(probs[0, 1])


def _observe(ms: float):
    _metrics["requests"] += 1
    _metrics["latency_sum"] += ms
    for b in _BUCKETS:
        if ms <= b:
            _metrics["latency_buckets"][b] = _metrics["latency_buckets"].get(b, 0) + 1


@app.post("/v1/turn", response_model=TurnResponse)
def turn(req: TurnRequest):
    t0 = time.perf_counter()
    try:
        # Only the tail of a long transcript matters for endpointing.
        text = req.transcript.strip()
        if len(text.split()) > 48:
            text = " ".join(text.split()[-48:])
        p = _predict(text)
    except Exception:
        _metrics["errors"] += 1
        raise
    ms = (time.perf_counter() - t0) * 1000
    _observe(ms)
    return TurnResponse(
        probability_complete=round(p, 4),
        model_version=MODEL_VERSION,
        inference_ms=round(ms, 2),
        **decide(p),
    )


@app.get("/", include_in_schema=False)
def demo_page():
    """Interactive demo: live mic (browser STT) or type-along scoring."""
    return FileResponse(Path(__file__).parent / "demo.html", media_type="text/html")


@app.get("/healthz")
def healthz():
    return Response(status_code=200 if _state["ready"] else 503)


@app.get("/livez")
def livez():
    return {"status": "alive"}


@app.get("/metrics")
def metrics():
    lines = [
        "# TYPE turn_requests_total counter",
        f"turn_requests_total {_metrics['requests']}",
        "# TYPE turn_errors_total counter",
        f"turn_errors_total {_metrics['errors']}",
        "# TYPE turn_latency_ms histogram",
    ]
    cumulative = 0
    for b in _BUCKETS:
        cumulative = _metrics["latency_buckets"].get(b, 0)
        lines.append(f'turn_latency_ms_bucket{{le="{b}"}} {cumulative}')
    lines.append(f'turn_latency_ms_bucket{{le="+Inf"}} {_metrics["requests"]}')
    lines.append(f"turn_latency_ms_sum {_metrics['latency_sum']:.2f}")
    lines.append(f"turn_latency_ms_count {_metrics['requests']}")
    lines.append(f'turn_model_info{{version="{MODEL_VERSION}"}} 1')
    return Response("\n".join(lines) + "\n", media_type="text/plain")
