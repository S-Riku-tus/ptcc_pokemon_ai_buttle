from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v10_is_single_search_teacher_distillation() -> None:
    model = json.loads((ROOT / "ranker_model.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    assert "teacher_team_id" not in model["feature_names"]
    assert len(model["trees"]) == 505
    assert metadata["ranker"]["teacher_team_id"] == 16561259
    assert metadata["version"] == "10.0.0"
