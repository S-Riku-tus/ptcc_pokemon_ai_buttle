from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v12_bundles_the_validation_selected_three_policies() -> None:
    primary = json.loads((ROOT / "ranker_model.json").read_text(encoding="utf-8"))
    fresh = json.loads((ROOT / "ranker_fresh0_model.json").read_text(encoding="utf-8"))
    old20 = json.loads((ROOT / "ranker_old20_model.json").read_text(encoding="utf-8"))
    assert len(primary["trees"]) == 1485
    assert fresh["teacher_team_id"] == 16422241
    assert fresh["teacher_team_code"] == 0
    assert old20["teacher_team_id"] == 16561259
    assert old20["teacher_team_code"] == 20


def test_v12_main_uses_unanimous_two_voter_gate() -> None:
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "fresh_index == old20_index" in source
    assert '"consensus_gate": dict(_GATE_STATS)' in source
