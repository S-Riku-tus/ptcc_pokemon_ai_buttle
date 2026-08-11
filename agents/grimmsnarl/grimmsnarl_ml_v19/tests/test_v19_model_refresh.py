"""The v19 win-weighted, unconditioned model artifact."""

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


def test_model_artifact_is_the_recorded_win_weighted_fit() -> None:
    model = artifact("ranker_model.json")
    metadata = artifact("metadata.json")
    digest = hashlib.sha256(
        (AGENT_DIR / "ranker_model.json").read_bytes()
    ).hexdigest()
    assert len(model["trees"]) == 575
    assert len(model["feature_names"]) == 822
    assert "teacher_team_id" not in model["feature_names"]
    assert digest == metadata["ranker"]["sha256"]
    assert metadata["ranker"]["win_weight"] == 4.0
    assert metadata["ranker"]["retrained"] is True


def test_only_one_coherent_policy_model_ships() -> None:
    assert not (AGENT_DIR / "ranker_model_v9.json").exists()
    ranker = ml_runtime.Ranker()
    assert not hasattr(ranker, "secondary_model")
    assert ranker.teacher_code is None
    assert ranker.default_teacher_code is None
    assert ranker.route_teacher_codes == {}


def test_public_route_does_not_change_the_unconditioned_model() -> None:
    ranker = ml_runtime.Ranker()
    ranker.set_route("v8_alakazam_guarded")
    assert ranker.active_route == "v8_alakazam_guarded"
    assert ranker.teacher_code is None
    ranker.set_route("v8_mirror")
    assert ranker.active_route == "v8_mirror"
    assert ranker.snapshot()["route_teacher_changes"] == 0


def test_legacy_class_escalation_is_retired() -> None:
    ranker = ml_runtime.Ranker()
    assert ml_runtime.ESCALATION_MODE == "off"
    assert ranker.escalation_code is None
