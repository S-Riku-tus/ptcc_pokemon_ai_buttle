from __future__ import annotations

import os

import fallback_v3
from fallback_v3 import agent as _fallback_agent
from fallback_v3 import diag_reset as _fallback_reset
from fallback_v3 import diag_snapshot as _fallback_snapshot
from ml_runtime import HybridRanker
from policy_base import attack_table


_ATTACKS = {
    int(attack_id): {"damage": getattr(attack, "damage", 0)}
    for attack_id, attack in attack_table.items()
}
_RUNTIME = HybridRanker(
    attacks=_ATTACKS,
    threshold=float(os.environ.get("ALAKAZAM_ML_THRESHOLD", "0.55")),
)
_DIAG = fallback_v3._DIAG


def _choose(observation):
    """Internal entry implementation; public ``agent`` is deliberately last."""
    fallback = list(_fallback_agent(observation))
    if observation.get("select") is None:
        return fallback
    return _RUNTIME.choose(observation, fallback)


def diag_reset():
    _fallback_reset()
    _RUNTIME.reset()


def diag_snapshot():
    return {"fallback": _fallback_snapshot(), "ml": _RUNTIME.snapshot()}


# IMPORTANT: Kaggle's Python loader selects the last callable in main.py.
# Keep this function as the final callable definition in this file.
def agent(observation):
    return _choose(observation)
