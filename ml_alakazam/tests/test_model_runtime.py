from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_alakazam.src.distill_model import tree_score
from ml_alakazam.src.feature_engineering import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents" / "alakazam_ml_v1"


def _runtime_module():
    sys.path.insert(0, str(AGENT))
    try:
        spec = importlib.util.spec_from_file_location("ml_runtime_test", AGENT / "ml_runtime.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(AGENT))


def test_compact_model_matches_saved_predictions():
    model = json.loads((ROOT / "models" / "ranker_model.json").read_text())
    candidates = pd.read_parquet(ROOT / "data_processed" / "legal_candidate_dataset.parquet")
    test = candidates[candidates["split_time"] == "test"].head(100)
    actual = np.asarray([
        tree_score(row.tolist(), model)
        for row in test[FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
    ])
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ROOT / "models" / "ranker_model.txt"))
    expected = booster.predict(test[FEATURE_COLUMNS].fillna(0))
    assert np.max(np.abs(actual - expected)) < 1e-10


def test_tree_runtime_handles_nan_and_unknown_large_ids():
    model = json.loads((ROOT / "models" / "ranker_model.json").read_text())
    features = [0.0] * len(model["feature_names"])
    features[0] = float("nan")
    features[-1] = 99999999.0
    assert math.isfinite(tree_score(features, model))


def test_model_load_failure_uses_legal_fallback(monkeypatch):
    runtime_module = _runtime_module()
    monkeypatch.setattr(runtime_module, "_model_path", lambda: "definitely_missing_model.json")
    runtime = runtime_module.HybridRanker()
    observation = {
        "remainingOverageTime": 600,
        "current": {"yourIndex": 0, "players": [{}, {}]},
        "select": {"option": [{"type": 14}], "minCount": 1, "maxCount": 1, "context": 0},
    }
    assert runtime.choose(observation, [0]) == [0]
    assert runtime.snapshot()["model_loaded"] is False


def test_timeout_guard_uses_fallback():
    runtime_module = _runtime_module()
    runtime = runtime_module.HybridRanker()
    observation = {
        "remainingOverageTime": 1.0,
        "current": {"yourIndex": 0, "players": [{}, {}]},
        "select": {"option": [{"type": 14}], "minCount": 1, "maxCount": 1, "context": 0},
    }
    assert runtime.choose(observation, [0]) == [0]


def test_feature_order_matches_distilled_model():
    model = json.loads((ROOT / "models" / "ranker_model.json").read_text())
    assert model["feature_names"] == FEATURE_COLUMNS

