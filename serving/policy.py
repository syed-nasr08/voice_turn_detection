"""Decision policy: convert P(complete) into an endpointing recommendation.

The model never ends a turn by itself — it recommends how much *additional
silence* the agent should require before treating the turn as over. The caller
(the agent's VAD loop) enforces the timeout and the hard ceiling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    # (min_probability, recommended_silence_ms, hint)
    bands: tuple = (
        (0.90, 160, "complete"),
        (0.60, 600, "maybe_complete"),
        (0.30, 1000, "maybe_complete"),
        (0.00, 1600, "wait"),
    )
    hard_ceiling_ms: int = 3000  # agent must respond after this much silence, always


CONFIG = PolicyConfig()


def decide(p_complete: float, config: PolicyConfig = CONFIG):
    for threshold, timeout_ms, hint in config.bands:
        if p_complete >= threshold:
            return {"decision_hint": hint, "recommended_timeout_ms": timeout_ms}
    return {"decision_hint": "wait", "recommended_timeout_ms": config.hard_ceiling_ms}
