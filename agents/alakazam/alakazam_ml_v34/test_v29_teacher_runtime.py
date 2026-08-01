"""Inherited safety regressions plus v32 model provenance."""

from __future__ import annotations

import json
from pathlib import Path

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from ml_runtime import (
    HybridRanker,
    _candidate_safety_reason,
    _feature_semantic_key,
)


HERE = Path(__file__).resolve().parent


def _context(
    action_type: str,
    *,
    card_id: int = -1,
    breaks_current_ko: bool = False,
) -> dict[str, object]:
    return {
        "action_type": action_type,
        "card_id": card_id,
        "breaks_current_ko": breaks_current_ko,
        "attack_lethal": False,
    }


def test_v34_model_records_the_refetched_yushin_corpus():
    model = json.loads((HERE / "ranker_model.json").read_text("utf-8"))

    assert model["runtime_scope"] == "v34_yushin_recent_corpus_ranker"
    assert model["teacher_trajectories"] == 1981
    assert model["training_decisions"] == 97511
    # Cohorts describe the whole 2,268-episode index; the deployed model is
    # fitted on the newest 87.5% of it, which is the trajectory count above.
    assert model["teacher_cohorts"] == {
        "yushin_20260717": 180,
        "yushin_current_top": 115,
        "yushin_20260726": 994,
        "yushin_20260801": 980,
    }
    assert model["ensemble_role"].startswith("large_leaf")
    assert not (HERE / "ranker_numeric_model.json").exists()
    assert len(model["feature_names"]) > 600
    assert "v29_ranker_score" in model["feature_names"]
    assert "recent_log_0_card_id" in model["feature_names"]


def test_v33_runtime_cleanly_omits_rejected_numeric_model():
    runtime = HybridRanker()
    snapshot = runtime.snapshot()

    assert snapshot["model_loaded"]
    assert not snapshot["numeric_model_loaded"]
    assert "numeric_model" not in snapshot["errors"]


def test_v29_semantic_label_collapses_equivalent_card_copies():
    left = {
        "option_type": 0,
        "candidate_card_id": 741,
        "candidate_attack_id": -1,
        "candidate_target_id": -1,
        "candidate_target_hp": -1,
        "candidate_target_max_hp": -1,
        "candidate_target_energy": -1,
        "candidate_target_special_energy": -1,
        "candidate_inplay_area": -1,
    }
    right = dict(left)
    right["option_index"] = 17

    assert _feature_semantic_key(left) == _feature_semantic_key(right)


def test_v29_safety_keeps_immediate_ko_material():
    reason = _candidate_safety_reason(
        _context("trainer", breaks_current_ko=True),
        _context("attack"),
        {},
        attack_is_available=True,
    )

    assert reason == "breaks_current_ko"


def test_v29_safety_never_ends_with_ready_alakazam_attack():
    reason = _candidate_safety_reason(
        _context("end"),
        _context("trainer"),
        {"has_ready_active_alakazam": 1},
        attack_is_available=True,
    )

    assert reason == "end_with_ready_attack"


def test_v29_safety_preserves_two_body_dudunsparce_floor():
    reason = _candidate_safety_reason(
        _context("ability", card_id=66),
        _context("trainer"),
        {"self_board_count": 2},
        attack_is_available=False,
    )

    assert reason == "dudunsparce_body_floor"
