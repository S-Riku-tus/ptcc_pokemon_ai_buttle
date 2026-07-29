"""Focused regressions for the v29 teacher-imitation runtime."""

from __future__ import annotations

import json
from pathlib import Path

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from ml_runtime import _candidate_safety_reason, _feature_semantic_key


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


def test_v29_model_records_frozen_teacher_corpus():
    model = json.loads((HERE / "ranker_model.json").read_text("utf-8"))

    assert model["runtime_scope"] == "v29_residual_main_policy"
    assert model["training_decisions"] == 18_336
    assert model["training_candidate_rows"] == 192_179
    assert model["teacher_ranks"] == [2, 3, 5, 6, 8]
    assert len(model["feature_names"]) == 422
    assert model["fallback_probability"] == 0.20


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
