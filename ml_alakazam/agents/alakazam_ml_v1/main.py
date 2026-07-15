from __future__ import annotations

import os

from fallback_v12 import agent as _fallback_agent
from fallback_v12 import diag_reset as _fallback_reset
from fallback_v12 import diag_snapshot as _fallback_snapshot
from fallback_v12 import (
    AlakazamPolicy, EFFECT_PREVENT_ENERGY, ENERGY_TYPES, FezMode,
    GLOBAL_EFFECT_PROTECTORS, TurnState, _is_ace_spec, _turn_boss_mark,
    _TURN_STATE, _validate_deck, card_table, get_card, my_deck,
)
from ml_runtime import HybridRanker
from policy_base import attack_table


_ATTACKS = {
    int(attack_id): {"damage": getattr(attack, "damage", 0)}
    for attack_id, attack in attack_table.items()
}
_RUNTIME = HybridRanker(
    attacks=_ATTACKS,
    threshold=float(os.environ.get("ALAKAZAM_ML_THRESHOLD", "0.65")),
)
_DIAG = __import__("fallback_v12")._DIAG


def agent(observation):
    fallback = list(_fallback_agent(observation))
    if observation.get("select") is None:
        return fallback
    return _RUNTIME.choose(observation, fallback)


def diag_reset():
    _fallback_reset()
    _RUNTIME.reset()


def diag_snapshot():
    return {"fallback": _fallback_snapshot(), "ml": _RUNTIME.snapshot()}
