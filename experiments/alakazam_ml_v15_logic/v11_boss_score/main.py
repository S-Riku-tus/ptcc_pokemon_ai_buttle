"""Ablation: v14's protected escape, v11's ordinary Boss evaluation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14_DIR = ROOT / "agents" / "alakazam" / "alakazam_ml_v14"
V11_DIR = ROOT / "agents" / "alakazam" / "alakazam_ml_v11"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(V14_DIR))
P = _load("_v14_v11_boss_policy", V14_DIR / "fallback_policy.py")
sys.path.insert(0, str(V11_DIR))
V11 = _load("_v11_boss_reference_richdeck", V11_DIR / "fallback_policy.py")
_v14_boss = P.AlakazamPolicy._boss_target_score
_v11_boss = V11.AlakazamPolicy._boss_target_score


def _boss_target_score(self, target):
    if (self._articuno_escape_target(target)
            or self._boss_effect_lock_escape_ko(target)):
        return _v14_boss(self, target)
    return _v11_boss(self, target)


P.AlakazamPolicy._boss_target_score = _boss_target_score
fallback_policy = P
_DIAG = P._DIAG


def agent(obs_dict):
    return P.agent(obs_dict)
