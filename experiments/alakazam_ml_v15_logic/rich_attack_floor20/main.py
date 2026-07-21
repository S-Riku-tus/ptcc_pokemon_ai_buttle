"""Ablation: attack-turn Rich only, with a 20-card pre-draw runway."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14 = ROOT / "agents" / "alakazam" / "alakazam_ml_v14"
sys.path.insert(0, str(V14))
spec = importlib.util.spec_from_file_location("_v14_rich_attack_floor", V14 / "fallback_policy.py")
if spec is None or spec.loader is None:
    raise RuntimeError("could not load v14 policy")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
_original = P.AlakazamPolicy._enriching_attach_score


def _enriching_attach_score(self, pokemon, *, is_active=False):
    if (self.me.deckCount < 20
            or not any(option.type == P.OptionType.ATTACK
                       for option in (self.select.option or []))):
        return -1
    return _original(self, pokemon, is_active=is_active)


P.AlakazamPolicy._enriching_attach_score = _enriching_attach_score
fallback_policy = P
_DIAG = P._DIAG


def agent(obs_dict):
    return P.agent(obs_dict)
