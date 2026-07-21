"""Ablation: best deterministic candidate plus safe ML bench-role choice."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14_DIR = ROOT / "agents" / "alakazam" / "alakazam_ml_v14"
V11_DIR = ROOT / "agents" / "alakazam" / "alakazam_ml_v11"
sys.path.insert(0, str(V14_DIR))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _load("_v15_best_ml_fallback", V14_DIR / "fallback_policy.py")
sys.path.insert(0, str(V11_DIR))
V11 = _load("_v11_best_ml_reference", V11_DIR / "fallback_policy.py")
sys.path.insert(0, str(V14_DIR))
MLR = _load("_v15_best_ml_runtime", V14_DIR / "ml_runtime.py")

_v11_rich = V11.AlakazamPolicy._enriching_attach_score
_v14_boss = P.AlakazamPolicy._boss_target_score
_v11_boss = V11.AlakazamPolicy._boss_target_score


def _enriching_attach_score(self, pokemon, *, is_active=False):
    del is_active
    return _v11_rich(self, pokemon)


def _boss_target_score(self, target):
    if (self._articuno_escape_target(target)
            or self._boss_effect_lock_escape_ko(target)):
        return _v14_boss(self, target)
    return _v11_boss(self, target)


P.AlakazamPolicy._enriching_attach_score = _enriching_attach_score
P.AlakazamPolicy._boss_target_score = _boss_target_score

_original_scope = MLR._candidate_scope_reason


def _candidate_scope_reason(context, fallback_context):
    reason = _original_scope(context, fallback_context)
    if (reason == "preserve_fallback_bench_role"
            and context["card_id"] in MLR.ML_SAFE_BENCH_IDS
            and fallback_context["card_id"] in MLR.ML_SAFE_BENCH_IDS):
        return None
    return reason


MLR._candidate_scope_reason = _candidate_scope_reason
MLR.ML_ALLOWED_ACTIONS.add("evolve")
_RUNTIME = MLR.HybridRanker(threshold=0.55)
_RUNTIME.enable_override = True
_RUNTIME.threshold_override = 0.37
if _RUNTIME.model is not None:
    _RUNTIME.model["fallback_probability"] = 0.37
fallback_policy = P
_DIAG = P._DIAG


def agent(observation):
    fallback = list(P.agent(observation))
    if observation.get("select") is None:
        return fallback
    return _RUNTIME.choose(observation, fallback)
