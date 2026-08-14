from __future__ import annotations

import json
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
V8 = AGENT.parent / "grimmsnarl_ml_v8"


def test_v24_keeps_v22s_model_condition_and_runtime_bytes() -> None:
    source = (V8 / "ranker_model.json").read_bytes().strip()
    expected = source.replace(
        b'"teacher_team_id":16494330', b'"teacher_team_id":16371703'
    ).replace(
        b'"teacher_team_code":16', b'"teacher_team_code":0'
    )
    assert (AGENT / "ranker_model.json").read_bytes().strip() == expected

    for name in (
        "deck.csv",
        "fallback_policy.py",
        "ml_features.py",
        "ml_planner.py",
        "ml_runtime.py",
        "policy_base.py",
    ):
        assert (AGENT / name).read_bytes() == (V8 / name).read_bytes(), name


def test_metadata_records_v22_as_the_parent() -> None:
    metadata = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    model = json.loads((AGENT / "ranker_model.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "grimmsnarl_ml_v24"
    assert metadata["parent_agent"] == "grimmsnarl_ml_v22"
    assert metadata["deck_hash"] == "9714ab5c3996f6cc"
    assert metadata["deck_changed"] is False
    assert model["teacher_team_id"] == metadata["ranker"]["teacher_team_id"] == 16371703
    assert model["teacher_team_code"] == metadata["ranker"]["teacher_team_code"] == 0
    assert metadata["ranker"]["trees_changed_from_v22"] is False
