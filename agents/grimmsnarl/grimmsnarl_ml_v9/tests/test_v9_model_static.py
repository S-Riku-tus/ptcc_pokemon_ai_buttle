from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v9_is_a_deployable_unconditioned_current_policy() -> None:
    model = json.loads((ROOT / "ranker_model.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    assert "teacher_team_id" not in model["feature_names"]
    assert model.get("teacher_team_id") is None
    assert model.get("teacher_team_code") is None
    assert len(model["trees"]) == 1485
    assert metadata["version"] == "9.0.0"
