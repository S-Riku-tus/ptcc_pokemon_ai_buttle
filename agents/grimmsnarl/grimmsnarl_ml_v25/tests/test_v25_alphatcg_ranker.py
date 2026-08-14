from __future__ import annotations

import hashlib
import json
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
V22 = AGENT.parent / "grimmsnarl_ml_v22"


def test_v25_pins_the_retrained_alphatcg_ranker() -> None:
    raw = (AGENT / "ranker_model.json").read_bytes()
    model = json.loads(raw)
    metadata = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["name"] == "grimmsnarl_ml_v25"
    assert metadata["parent_agent"] == "grimmsnarl_ml_v22"
    assert model["teacher_team_id"] == 16381823
    assert model["teacher_team_code"] == 3
    assert len(model["trees"]) == metadata["ranker"]["trees"] == 228
    assert len(model["feature_names"]) == metadata["ranker"]["features"] == 823
    assert hashlib.sha256(raw).hexdigest() == metadata["ranker"]["sha256"]


def test_v25_keeps_the_measured_v22_runtime_and_deck() -> None:
    for name in (
        "deck.csv",
        "fallback_policy.py",
        "ml_features.py",
        "ml_planner.py",
        "policy_base.py",
    ):
        assert (AGENT / name).read_bytes() == (V22 / name).read_bytes(), name

    metadata = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["deck_hash"] == "9714ab5c3996f6cc"
    assert metadata["deck_changed"] is False
    assert metadata["change_scope"]["v24_froslass_veto_included"] is False
    assert metadata["change_scope"]["explicit_search_included"] is False

    runtime = (AGENT / "ml_runtime.py").read_text(encoding="utf-8")
    assert 'ESCALATION_MODE = "off"' in runtime
