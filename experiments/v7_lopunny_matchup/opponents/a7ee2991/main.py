"""Local Mega Lopunny ex / Mega Froslass ex imitation sparring agent.

This is an evaluation instrument, not a Kaggle candidate.  Its exact-list
teacher corpus is reconstructed from the opponent seat of 106 stored games
against the Grimmsnarl corpus.
"""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import agent as _fallback_agent
from tree_runtime import ImitationRuntime


MY_DECK = [
    3, 3, 3, 11, 11, 11, 11, 13,
    66, 66, 66, 174,
    305, 305, 305, 305,
    848, 848, 849, 849, 860, 860, 861, 861,
    1086, 1086, 1086, 1086,
    1087, 1087, 1087,
    1121, 1121, 1121, 1121,
    1122, 1122,
    1152, 1152, 1152, 1152,
    1174, 1174, 1174, 1182, 1182,
    1225, 1225, 1225,
    1227, 1227, 1227, 1227,
    1229, 1229, 1229, 1229,
    1264, 1264, 1264,
]

_RUNTIME = ImitationRuntime()


def diag_reset():
    _RUNTIME.reset()
    fallback_policy.diag_reset()


def diag_snapshot():
    return {
        "mode": os.environ.get("LOPUNNY_POLICY_MODE", "ml").lower(),
        "ml": _RUNTIME.snapshot(),
        "fallback": fallback_policy.diag_snapshot(),
    }


# Keep the public callable last: the competition loader chooses the last
# callable defined in main.py.
def agent(observation):
    if not isinstance(observation, dict):
        return []
    if observation.get("select") is None:
        _RUNTIME.reset()
        _fallback_agent(observation)
        return list(MY_DECK)
    if os.environ.get("LOPUNNY_POLICY_MODE", "ml").lower() == "rule":
        return list(_fallback_agent(observation))
    return _RUNTIME.choose(observation)
