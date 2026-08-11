"""The v19 core model and its public matchup-conditioned teacher pin."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_runtime  # noqa: E402


def model() -> dict:
    return json.loads(
        (AGENT_DIR / "ranker_model.json").read_text(encoding="utf-8")
    )


def test_refreshed_model_artifact_is_the_recorded_fit() -> None:
    artifact = model()
    metadata = json.loads(
        (AGENT_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(
        (AGENT_DIR / "ranker_model.json").read_bytes()
    ).hexdigest()
    assert len(artifact["trees"]) == 454
    assert len(artifact["feature_names"]) == 823
    assert artifact["teacher_team_id"] == 16422241
    assert artifact["teacher_team_code"] == 9
    assert digest == metadata["ranker"]["sha256"]
    assert metadata["ranker"]["retrained"] is True


def test_model_embeds_the_public_route_teacher_map() -> None:
    artifact = model()
    assert artifact["route_teacher_codes"] == {
        "v8_default": 9,
        "v8_mirror": 9,
        "v8_wall_guarded": 9,
        "v8_alakazam_guarded": 20,
    }
    assert artifact["route_teacher_teams"] == {
        "v8_default": 16422241,
        "v8_mirror": 16422241,
        "v8_wall_guarded": 16422241,
        "v8_alakazam_guarded": 16561259,
    }


def test_ranker_changes_only_the_whole_argmax_teacher_pin() -> None:
    ranker = ml_runtime.Ranker()
    assert ranker.teacher_code == 9
    ranker.set_route("v8_alakazam_guarded")
    assert ranker.teacher_code == 20
    assert ranker.active_route == "v8_alakazam_guarded"
    ranker.set_route("v8_mirror")
    assert ranker.teacher_code == 9
    snapshot = ranker.snapshot()
    assert snapshot["route_teacher_changes"] == 1
    assert snapshot["route_teacher_codes"]["v8_alakazam_guarded"] == 20


def test_unknown_route_falls_back_to_the_trained_default() -> None:
    ranker = ml_runtime.Ranker()
    ranker.set_route("future_unknown_matchup")
    assert ranker.teacher_code == ranker.default_teacher_code == 9


def test_legacy_class_escalation_is_retired() -> None:
    ranker = ml_runtime.Ranker()
    assert ml_runtime.ESCALATION_MODE == "off"
    assert ranker.escalation_code is None
