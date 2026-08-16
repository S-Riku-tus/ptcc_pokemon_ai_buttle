from __future__ import annotations

import json
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
V8 = AGENT.parent / "grimmsnarl_ml_v8"


def test_only_model_condition_changes_the_v8_policy_bytes() -> None:
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


def test_metadata_records_the_single_policy_change() -> None:
    metadata = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    model = json.loads((AGENT / "ranker_model.json").read_text(encoding="utf-8"))
    # vfinal inherits this bundle unchanged and adds only the search layer, so
    # the invariant is the ranker pin, not the directory name.
    assert metadata["name"] in ("grimmsnarl_ml_v22", "grimmsnarl_ml_vfinal")
    assert metadata["deck_hash"] == "9714ab5c3996f6cc"
    assert metadata["deck_changed"] is False
    assert model["teacher_team_id"] == metadata["ranker"]["teacher_team_id"] == 16371703
    assert model["teacher_team_code"] == metadata["ranker"]["teacher_team_code"] == 0
    assert metadata["ranker"]["trees_changed"] is False
