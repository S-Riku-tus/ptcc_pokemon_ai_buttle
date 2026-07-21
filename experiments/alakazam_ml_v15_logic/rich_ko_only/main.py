"""Ablation: spend Rich only when its net +3 hand converts Powerful Hand to a KO."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14 = ROOT / "agents" / "alakazam" / "alakazam_ml_v14"
sys.path.insert(0, str(V14))
spec = importlib.util.spec_from_file_location("_v14_rich_ko", V14 / "fallback_policy.py")
if spec is None or spec.loader is None:
    raise RuntimeError("could not load v14 policy")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
_original = P.AlakazamPolicy._enriching_attach_score


def _enriching_attach_score(self, pokemon, *, is_active=False):
    active = self.me.active[0] if self.me.active else None
    opponent = self.opponent.active[0] if self.opponent.active else None
    if (active is None or active.id != P.C.ALAKAZAM or opponent is None
            or not self._can_attack(active) or self._effect_prevented(opponent)):
        return -1
    current = self._alakazam_damage(P.POWERFUL_HAND, opponent)
    projected = 20 * (self.me.handCount + 3)
    if not (current < opponent.hp <= projected):
        return -1
    return _original(self, pokemon, is_active=is_active)


P.AlakazamPolicy._enriching_attach_score = _enriching_attach_score
fallback_policy = P
_DIAG = P._DIAG


def agent(obs_dict):
    return P.agent(obs_dict)
