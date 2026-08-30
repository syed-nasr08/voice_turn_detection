"""Stress-test the turn-detector API: latency percentiles + throughput.

Usage:
  .venv/bin/python evaluation/latency_bench.py                 # sequential latency
  .venv/bin/python evaluation/latency_bench.py --concurrency 16 --requests 5000
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def load_payloads(n):
    rows = [json.loads(l) for l in open(ROOT / "data" / "dataset" / "test.jsonl")]
    random.seed(7)
    return [{"transcript": r["text"]} for r in random.sample(rows, min(n, len(rows)))]


async def worker(client, url, payloads, latencies, errors):
    for p in payloads:
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=p)
            r.raise_for_status()
        except Exception:
            errors.append(1)
            continue
        latencies.append((time.perf_counter() - t0) * 1000)


async def run(args):
    payloads = load_payloads(args.requests)
    chunks = [payloads[i::args.concurrency] for i in range(args.concurrency)]
    latencies, errors = [], []

    async with httpx.AsyncClient(timeout=5.0) as client:
        # warmup
        for p in payloads[:20]:
            await client.post(args.url, json=p)

        t0 = time.perf_counter()
        await asyncio.gather(*[
            worker(client, args.url, c, latencies, errors) for c in chunks
        ])
        wall = time.perf_counter() - t0

    latencies.sort()
    q = lambda p: latencies[min(len(latencies) - 1, int(p / 100 * len(latencies)))]
    print(f"concurrency={args.concurrency}  requests={len(latencies)}  errors={len(errors)}")
    print(f"wall={wall:.2f}s  throughput={len(latencies) / wall:.1f} req/s")
    print(f"latency ms: mean={statistics.mean(latencies):.2f}  p50={q(50):.2f}  "
          f"p90={q(90):.2f}  p99={q(99):.2f}  max={latencies[-1]:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/turn")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=1)
    asyncio.run(run(ap.parse_args()))
