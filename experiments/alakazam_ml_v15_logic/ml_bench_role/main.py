"""Ablation: let ML choose between safe Abra/Dunsparce bench roles."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14 = ROOT / "agents" / "alakazam" / "alakazam_ml_v14"
sys.path.insert(0, str(V14))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _load("_v14_ml_bench_fallback", V14 / "fallback_policy.py")
MLR = _load("_v14_ml_bench_runtime", V14 / "ml_runtime.py")
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
