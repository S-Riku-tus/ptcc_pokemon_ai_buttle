"""v3 elite action-prior runtime invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from ml_runtime import Ranker  # noqa: E402


def test_exported_action_model_is_complete_and_exact() -> None:
    model = json.loads(
        (AGENT_DIR / "action_model.json").read_text(encoding="utf-8")
    )
    assert model["format"] == "lightgbm_multiclass_tree_v1"
    assert model["blend_alpha"] == pytest.approx(0.10)
    assert model["export_probe_max_abs_error"] <= 1e-9
    assert len(model["feature_names"]) == 556
    assert len(model["classes"]) == 15
    assert {len(trees) for trees in model["class_trees"]} == {143}


def test_ranker_loads_action_prior_without_weakening_fallback() -> None:
    ranker = Ranker()
    snapshot = ranker.snapshot()
    assert snapshot["action_prior_loaded"] == 1
    assert snapshot["action_prior_load_error"] is None
    assert ranker.teacher_code == 16


def test_action_prior_can_reorder_action_families() -> None:
    ranker = object.__new__(Ranker)
    ranker.stats = {"action_prior_used": 0}
    ranker.action_model = {
        "feature_names": [],
        "classes": [0, 1],
        "class_trees": [[{"v": 0.0}], [{"v": 30.0}]],
        "blend_alpha": 0.10,
    }
    features = [{"action_type_id": 0}, {"action_type_id": 1}]
    # v2 prefers action 0, but a sufficiently strong elite prior for action 1
    # changes the family while leaving concrete selection to the base ranker.
    scores = ranker._apply_action_prior(features, [0, 1], [2.0, 1.0])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(2.0)
    assert scores[1] > scores[0]
    assert ranker.stats["action_prior_used"] == 1
