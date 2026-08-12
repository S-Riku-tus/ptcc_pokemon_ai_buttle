"""The v21 hard-state-weighted, unconditioned model artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_runtime  # noqa: E402


def artifact(name: str) -> dict:
    return json.loads((AGENT_DIR / name).read_text(encoding="utf-8"))


def test_model_artifact_is_the_recorded_hard_state_fit() -> None:
    model = artifact("ranker_model.json")
    metadata = artifact("metadata.json")
    digest = hashlib.sha256(
        (AGENT_DIR / "ranker_model.json").read_bytes()
    ).hexdigest()
    assert len(model["trees"]) == 1380
    assert len(model["feature_names"]) == 848
    # The six retreat-lock columns must be in the shipped model, not only in
    # the extractor, or the corpus upgrade never reached inference.
    for name in (
        "active_retreat_locked", "bench_marnie_body_count",
        "bench_locked_ready_attacker", "retreat_lock_risk",
        "candidate_funds_active_retreat", "candidate_leaves_retreat_locked",
    ):
        assert name in model["feature_names"], name
    assert "teacher_team_id" not in model["feature_names"]
    assert digest == metadata["ranker"]["sha256"]
    assert metadata["ranker"]["win_weight"] == 1.0
    assert metadata["ranker"]["hard_state_weight"] == 2.5
    assert metadata["ranker"]["hard_state_set"] == "v21"
    assert metadata["ranker"]["early_stopping_patience"] == 800
    assert metadata["ranker"]["eventual_result_used_for_weighting"] is False
    assert metadata["ranker"]["retrained"] is True


def test_only_one_coherent_policy_model_ships() -> None:
    assert not (AGENT_DIR / "ranker_model_v9.json").exists()
    ranker = ml_runtime.Ranker()
    assert not hasattr(ranker, "secondary_model")
    assert ranker.teacher_code is None
    assert ranker.default_teacher_code is None
    assert not hasattr(ranker, "route_teacher_codes")


def test_dead_public_teacher_route_is_removed() -> None:
    ranker = ml_runtime.Ranker()
    assert ranker.teacher_code is None
    assert not hasattr(ranker, "set_route")
    assert "route_teacher_changes" not in ranker.snapshot()


def test_legacy_class_escalation_is_retired() -> None:
    ranker = ml_runtime.Ranker()
    assert ml_runtime.ESCALATION_MODE == "off"
    assert ranker.escalation_code is None
