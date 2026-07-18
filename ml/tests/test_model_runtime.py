from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
lgb = pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")

from ml.core.distill import tree_score


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "ml" / "alakazam"
AGENT = ROOT / "agents" / "alakazam_ml_v8"


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


def _observation(options):
    return {
        "remainingOverageTime": 600,
        "current": {
            "yourIndex": 0, "firstPlayer": 0, "turn": 3, "turnActionCount": 1,
            "players": [
                {
                    "hand": [{"id": 1081}, {"id": 5}],
                    "handCount": 2, "deckCount": 30, "prize": [1] * 6,
                    "active": [{"id": 743, "hp": 140, "maxHp": 140, "energyCards": [{"id": 5}]}],
                    "bench": [{"id": 741, "hp": 60, "maxHp": 60}], "benchMax": 5,
                },
                {
                    "handCount": 6, "deckCount": 30, "prize": [1] * 6,
                    "active": [{"id": 999, "hp": 100, "maxHp": 100, "energyCards": [{"id": 19}]}],
                    "bench": [], "benchMax": 5,
                },
            ],
        },
        "select": {"type": 0, "context": 0, "option": options, "minCount": 1, "maxCount": 1},
    }


def test_compact_categorical_model_matches_lightgbm():
    model = json.loads((DATA_ROOT / "models" / "ranker_model.json").read_text())
    schema = json.loads((DATA_ROOT / "models" / "model_schema.json").read_text())
    rows = pd.read_csv(DATA_ROOT / "processed" / "dataset_rows.csv.gz", nrows=300)
    rows["action_type"] = rows["action_type"].astype(str).map(schema["action_type_map"]).fillna(-1)
    matrix = rows[model["feature_names"]].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(float)
    actual = np.asarray([tree_score(row.tolist(), model) for row in matrix])
    booster = lgb.Booster(model_file=str(DATA_ROOT / "models" / "ranker.txt"))
    expected = booster.predict(matrix)
    assert np.max(np.abs(actual - expected)) < 1e-12


def test_tree_runtime_handles_nan_and_unknown_large_ids():
    model = json.loads((DATA_ROOT / "models" / "ranker_model.json").read_text())
    features = [0.0] * len(model["feature_names"])
    features[0] = float("nan")
    features[-1] = 99999999.0
    assert math.isfinite(tree_score(features, model))


def test_model_load_failure_uses_legal_fallback(monkeypatch):
    runtime_module = _runtime_module()
    monkeypatch.setattr(runtime_module, "_model_path", lambda: "definitely_missing_model.json")
    runtime = runtime_module.HybridRanker()
    observation = _observation([{"type": 14}, {"type": 13, "attackId": 1}])
    assert runtime.choose(observation, [0]) == [0]
    assert runtime.snapshot()["model_loaded"] is False


def test_timeout_and_nested_selection_use_fallback():
    runtime_module = _runtime_module()
    runtime = runtime_module.HybridRanker()
    observation = _observation([{"type": 14}, {"type": 13, "attackId": 1}])
    observation["remainingOverageTime"] = 1
    assert runtime.choose(observation, [0]) == [0]
    observation["remainingOverageTime"] = 600
    observation["select"]["context"] = 7
    assert runtime.choose(observation, [0]) == [0]


def test_rule_only_fallback_action_blocks_model_override(monkeypatch):
    runtime_module = _runtime_module()
    runtime = runtime_module.HybridRanker(threshold=0.0)
    runtime.model = {
        "format": "lightgbm_tree_v2", "feature_names": [],
        "trees": [{"v": 0.0}], "temperature": 1.0,
        "fallback_probability": 0.0, "fallback_margin": 0.0,
        "action_type_map": {"hammer": 0, "attack": 1},
        "action_type_thresholds": {},
    }
    scores = iter([10.0, 0.0])
    monkeypatch.setattr(runtime_module, "_tree_score", lambda row, model: next(scores))
    observation = _observation([{"type": 7, "index": 0}, {"type": 13, "attackId": 1}])
    assert runtime.choose(observation, [1]) == [1]
    assert runtime.snapshot()["fallback"] == 1
    assert runtime.snapshot().get("model_selected", 0) == 0


def test_energy_requires_high_probability(monkeypatch):
    runtime_module = _runtime_module()
    runtime = runtime_module.HybridRanker(threshold=0.0)
    runtime.model = {
        "format": "lightgbm_tree_v2", "feature_names": [],
        "trees": [{"v": 0.0}], "temperature": 1.0,
        "fallback_probability": 0.0, "fallback_margin": 0.0,
        "action_type_map": {"energy": 0, "attack": 1},
        "action_type_thresholds": {"energy": 0.85},
    }
    scores = iter([1.0, 0.0])  # P ~= .73, below energy gate.
    monkeypatch.setattr(runtime_module, "_tree_score", lambda row, model: next(scores))
    observation = _observation([
        {"type": 8, "index": 1, "inPlayArea": 4, "inPlayIndex": 0},
        {"type": 13, "attackId": 1},
    ])
    assert runtime.choose(observation, [1]) == [1]


def test_feature_order_matches_distilled_model():
    model = json.loads((DATA_ROOT / "models" / "ranker_model.json").read_text())
    schema = json.loads((DATA_ROOT / "models" / "model_schema.json").read_text())
    assert model["feature_names"] == schema["feature_columns"]
    assert model["format"] == "lightgbm_tree_v2"
