from __future__ import annotations

import json
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
V22 = AGENT.parent / "grimmsnarl_ml_v22"
V25 = AGENT.parent / "grimmsnarl_ml_v25"
V21 = AGENT.parent / "grimmsnarl_ml_v21"


def test_v22_remains_the_default_policy_bytes() -> None:
    for name in (
        "deck.csv",
        "fallback_policy.py",
        "ml_features.py",
        "ml_planner.py",
        "policy_base.py",
        "ranker_model.json",
    ):
        assert (AGENT / name).read_bytes() == (V22 / name).read_bytes(), name


def test_v25_is_only_a_separate_peer_candidate_model() -> None:
    assert (AGENT / "ranker_v25_model.json").read_bytes() == (
        V25 / "ranker_model.json"
    ).read_bytes()
    source = (AGENT / "main.py").read_text(encoding="utf-8")
    assert '_RANKER = Ranker("ranker_model.json")' in source
    assert '_PEER = Ranker("ranker_v25_model.json")' in source
    assert "is_mirror=_search_mirror(observation, route)" in source


def test_reactive_wall_guard_is_the_measured_v21_guard() -> None:
    assert (AGENT / "wall_break.py").read_bytes() == (
        V21 / "wall_break.py"
    ).read_bytes()


def test_metadata_records_the_safety_boundary() -> None:
    metadata = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "grimmsnarl_ml_v26"
    assert metadata["parent_agent"] == "grimmsnarl_ml_v22"
    assert metadata["deck_hash"] == "9714ab5c3996f6cc"
    assert metadata["deck_changed"] is False
    assert metadata["h2_search"]["determinizations"] == 3
    assert metadata["h2_search"]["minimum_turn"] == 5
    assert metadata["peer_candidate_ranker"]["default_policy"] is False
    assert metadata["value_model"]["test_auc"] >= 0.85
