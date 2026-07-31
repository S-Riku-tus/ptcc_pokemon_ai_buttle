from __future__ import annotations

import os

import fallback_policy
from fallback_policy import agent as _fallback_agent
from fallback_policy import diag_reset as _fallback_reset
from fallback_policy import diag_snapshot as _fallback_snapshot
from ml_runtime import HybridRanker
from policy_base import attack_table
from v29_runtime import HybridRanker as V29Ranker


_ATTACKS = {
    int(attack_id): {"damage": getattr(attack, "damage", 0)}
    for attack_id, attack in attack_table.items()
}
_RUNTIME = HybridRanker(
    attacks=_ATTACKS,
    threshold=float(
        os.environ.get(
            "ALAKAZAM_ML_V33_THRESHOLD",
            os.environ.get(
                "ALAKAZAM_ML_V32_THRESHOLD",
                os.environ.get("ALAKAZAM_ML_V31_THRESHOLD", "0.0"),
            ),
        )
    ),
)
_V29_RUNTIME = V29Ranker(
    attacks=_ATTACKS,
    threshold=float(os.environ.get("ALAKAZAM_ML_THRESHOLD", "0.20")),
)
_DIAG = fallback_policy._DIAG


def _choose(observation):
    """Internal entry implementation; public ``agent`` is deliberately last."""
    if observation.get("select") is None:
        return list(_fallback_agent(observation))
    recalled = _RUNTIME.recall(observation)
    if recalled is not None:
        # The intra-turn history has to see memory hits too, otherwise the
        # offer and pass-over counts drift away from the training corpus.
        _RUNTIME.note_decision(observation, recalled)
        return recalled
    deterministic = list(_fallback_agent(observation))
    baseline = _V29_RUNTIME.choose(observation, deterministic)
    action = _RUNTIME.choose(
        observation,
        baseline,
        deterministic,
        memory_checked=True,
    )
    _RUNTIME.note_decision(observation, action)
    return action


def diag_reset():
    _fallback_reset()
    _V29_RUNTIME.reset()
    _RUNTIME.reset()


def diag_snapshot():
    return {
        "fallback": _fallback_snapshot(),
        "v29": _V29_RUNTIME.snapshot(),
        "ml": _RUNTIME.snapshot(),
    }


# IMPORTANT: Kaggle's Python loader selects the last callable in main.py.
# Keep this function as the final callable definition in this file.
def agent(observation):
    return _choose(observation)
