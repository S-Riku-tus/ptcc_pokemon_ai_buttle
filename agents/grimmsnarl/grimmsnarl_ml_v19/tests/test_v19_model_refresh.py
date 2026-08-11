"""The v19 retrained primary, retained v9 expert, and public-state gate."""

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


def test_both_model_artifacts_are_the_recorded_fits() -> None:
    primary = artifact("ranker_model.json")
    secondary = artifact("ranker_model_v9.json")
    metadata = artifact("metadata.json")
    primary_digest = hashlib.sha256(
        (AGENT_DIR / "ranker_model.json").read_bytes()
    ).hexdigest()
    secondary_digest = hashlib.sha256(
        (AGENT_DIR / "ranker_model_v9.json").read_bytes()
    ).hexdigest()
    assert len(primary["trees"]) == 413
    assert len(secondary["trees"]) == 1485
    assert len(primary["feature_names"]) == 822
    assert primary["feature_names"] == secondary["feature_names"]
    assert "teacher_team_id" not in primary["feature_names"]
    assert primary_digest == metadata["ranker"]["primary"]["sha256"]
    assert secondary_digest == metadata["ranker"]["secondary"]["sha256"]
    assert metadata["ranker"]["retrained"] is True


def test_gate_uses_one_complete_expert_on_each_side_of_turn_four() -> None:
    ranker = ml_runtime.Ranker()
    ranker.set_route("v8_default")
    assert ranker._select_expert({"current": {"turn": 4}}) is ranker.model
    assert ranker._select_expert({"current": {"turn": 5}}) is ranker.secondary_model

    ranker.set_route("v8_alakazam_guarded")
    assert ranker._select_expert({"current": {"turn": 4}}) is ranker.model
    assert ranker._select_expert({"current": {"turn": 5}}) is ranker.secondary_model

    ranker.set_route("v8_mirror")
    assert ranker._select_expert({"current": {"turn": 4}}) is ranker.model
    assert ranker._select_expert({"current": {"turn": 9}}) is ranker.secondary_model
    assert ranker.teacher_code is None
    assert ranker.default_teacher_code is None
    assert ranker.route_teacher_codes == {}


def test_unknown_route_uses_opening_primary_then_stable_secondary() -> None:
    ranker = ml_runtime.Ranker()
    ranker.set_route("future_unknown_matchup")
    assert ranker._select_expert({"current": {"turn": 1}}) is ranker.model
    assert ranker._select_expert({"current": {"turn": 6}}) is ranker.secondary_model


def test_legacy_class_escalation_is_retired() -> None:
    ranker = ml_runtime.Ranker()
    assert ml_runtime.ESCALATION_MODE == "off"
    assert ranker.escalation_code is None
