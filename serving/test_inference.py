"""Smoke-test the running turn-detector API with realistic utterances.

Usage: .venv/bin/python serving/test_inference.py [--url http://127.0.0.1:8000]
"""

import argparse

import httpx

CASES = [
    # (transcript, expectation)
    ("What time do you close?", "complete"),
    ("Yeah that works for me.", "complete"),
    ("My phone number is five five five one two three four five six seven.", "complete"),
    ("okay sounds good bye", "complete"),
    ("My phone number is five five five", "INCOMPLETE"),
    ("I want to cancel my", "INCOMPLETE"),
    ("So what I was thinking is", "INCOMPLETE"),
    ("um let me check", "INCOMPLETE"),
    ("My email is john at", "INCOMPLETE"),
    ("I'd like a burger, fries, and", "INCOMPLETE"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    health = httpx.get(f"{args.url}/healthz")
    print(f"healthz: {health.status_code}\n")

    for text, expect in CASES:
        r = httpx.post(f"{args.url}/v1/turn", json={"transcript": text}).json()
        print(f"p(complete)={r['probability_complete']:.3f}  "
              f"hint={r['decision_hint']:<15} timeout={r['recommended_timeout_ms']:>5}ms  "
              f"({r['inference_ms']:.1f}ms)  expect={expect:<10}  {text!r}")


if __name__ == "__main__":
    main()
