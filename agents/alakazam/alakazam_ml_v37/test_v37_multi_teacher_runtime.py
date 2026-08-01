from __future__ import annotations

import math

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from ml_runtime import HybridRanker


def _row(runtime: HybridRanker) -> list[float]:
    assert runtime.type_spec is not None
    return [0.0] * int(runtime.type_spec["rich_row_features"])


def test_v37_rmy_expert_artifact_matches_primary_feature_contract():
    runtime = HybridRanker()
    assert runtime.rmy_type_model is not None
    assert runtime.rmy_type_model["format"] == "v37_xgb_rmy_type_expert_v1"
    assert runtime.rmy_type_model["classes"] == runtime.type_model["classes"]
    assert runtime.rmy_type_model["cols"] == runtime.type_model["cols"]
    assert len(runtime.rmy_type_model["trees"]) == 1661
    assert math.isclose(float(runtime.rmy_type_model["weight"]), 0.15)


def test_v37_prediction_is_exact_probability_blend():
    runtime = HybridRanker()
    row = _row(runtime)
    primary = runtime._compact_type_probabilities(runtime.type_model, row)
    expert = runtime._compact_type_probabilities(runtime.rmy_type_model, row)
    assert primary is not None and expert is not None
    classes, p_primary = primary
    expert_classes, p_expert = expert
    assert classes == expert_classes
    expected = [0.85 * a + 0.15 * b for a, b in zip(p_primary, p_expert)]
    predicted_type, confidence = runtime._predict_action_type(row)
    best = max(range(len(expected)), key=expected.__getitem__)
    assert predicted_type == classes[best]
    assert math.isclose(confidence, expected[best], rel_tol=0.0, abs_tol=1e-12)


def test_v37_uses_selected_validation_threshold_and_reports_expert():
    runtime = HybridRanker()
    assert math.isclose(float(runtime.type_model["threshold"]), 0.45)
    snapshot = runtime.snapshot()
    assert snapshot["rmy_type_model_loaded"] is True
    assert snapshot["runtime_scope"].startswith("v37_rmy_blended")
