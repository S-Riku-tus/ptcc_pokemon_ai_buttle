"""v35 regressions: the two guards that were narrowed, and the ones that stay.

v35 changes the runtime, not the model. The v31 shell answered 16.3% of scoped
holdout decisions with the v29 baseline instead of the ranker's pick, and the
audit in ``scripts/experiment_alakazam_v35_shell_audit.py`` showed every guard
was net negative against the teacher, costing 5.16 points of played agreement.
Two guards account for 83% of that: the lethal guard forced the Powerful Hand
the moment it could KO the opposing active, and the Boss guard forced Boss's
Orders whenever the deterministic policy wanted it.

Both are narrowed from a compulsion ("play this now") to a prohibition ("do
not throw this away"). These tests pin the narrowing on the cases that changed
and, just as importantly, pin the cases that did not -- a guard that quietly
stopped firing would take the safety property with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from ml_runtime import (  # noqa: E402
    HybridRanker,
    _v35_safety_reason,
)
from v29_runtime import _candidate_safety_reason  # noqa: E402


HERE = Path(__file__).resolve().parent


def context(action, *, card_id=-1, breaks_ko=False, lethal=False):
    return {
        "option": {},
        "action_type": action,
        "card_id": card_id,
        "target_id": -1,
        "breaks_current_ko": breaks_ko,
        "attack_lethal": lethal,
    }


def state(*, ready_alakazam=1, board=4):
    return {
        "has_ready_active_alakazam": ready_alakazam,
        "self_board_count": board,
    }


def reason(pick, baseline, *, attack=True, lethal_available=False,
           board_state=None):
    return _v35_safety_reason(
        pick,
        baseline,
        board_state if board_state is not None else state(),
        attack_is_available=attack,
        lethal_available=lethal_available,
    )


# --- the lethal guard, narrowed ------------------------------------------


def test_lethal_board_no_longer_forces_the_attack():
    # v34 returned "lethal_guard" here before the ranker was even scored.
    # Powerful Hand stays on the board while the teacher banks value, so a
    # non-terminal pick is allowed through.
    for action in ("evolve", "energy", "bench", "ability", "trainer"):
        assert reason(
            context(action), context("attack", lethal=True),
            lethal_available=True,
        ) is None


def test_lethal_board_still_refuses_to_end_the_turn():
    assert reason(
        context("end"), context("attack", lethal=True),
        attack=False, lethal_available=True,
    ) == "lethal_declined_by_end"


def test_lethal_board_still_refuses_a_non_lethal_swing():
    assert reason(
        context("attack", lethal=False), context("attack", lethal=True),
        lethal_available=True,
    ) == "lethal_declined_by_weak_attack"


def test_lethal_board_allows_the_lethal_attack_itself():
    assert reason(
        context("attack", lethal=True), context("attack", lethal=True),
        lethal_available=True,
    ) is None


def test_breaking_the_ko_is_still_refused_on_a_lethal_board():
    # This is the guard that makes waiting safe: Powerful Hand deals 20 per
    # card in hand, so any action that spends the hand below lethal has to be
    # refused, or narrowing the lethal guard would give the KO away.
    assert reason(
        context("trainer", breaks_ko=True), context("attack", lethal=True),
        lethal_available=True,
    ) == "breaks_current_ko"


# --- the Boss guard, narrowed --------------------------------------------


def test_boss_route_no_longer_blocks_every_other_action():
    for action in ("evolve", "energy", "ability", "trainer", "attack"):
        assert reason(context(action), context("boss")) is None


def test_boss_route_is_still_preserved_against_ending_the_turn():
    assert reason(
        context("end"), context("boss"), attack=False,
    ) == "preserve_fallback_boss_route"


# --- everything else is unchanged ----------------------------------------


def test_unmodelled_other_is_still_refused():
    assert reason(context("other"), context("trainer")) == "unmodelled_other"


def test_ending_a_turn_with_a_ready_attack_is_still_refused():
    assert reason(
        context("end"), context("trainer"), attack=True,
    ) == "end_with_ready_attack"


def test_dudunsparce_body_floor_is_still_refused():
    assert reason(
        context("ability", card_id=66), context("trainer"),
        board_state=state(board=2),
    ) == "dudunsparce_body_floor"


def test_dudunsparce_cycle_is_allowed_with_a_third_body():
    assert reason(
        context("ability", card_id=66), context("trainer"),
        board_state=state(board=3),
    ) is None


def test_v35_only_diverges_from_the_shared_guard_where_intended():
    """Anything outside the two narrowed cases must match v29's guard."""
    picks = [
        context(action, card_id=card, breaks_ko=broke)
        for action in ("evolve", "energy", "bench", "ability", "trainer",
                       "attack", "retreat", "other", "end")
        for card in (-1, 66)
        for broke in (False, True)
    ]
    baselines = [context(action) for action in ("trainer", "boss", "attack")]
    for pick in picks:
        for baseline in baselines:
            for attack_available in (False, True):
                for board in (2, 4):
                    shared = _candidate_safety_reason(
                        pick, baseline, state(board=board),
                        attack_is_available=attack_available,
                    )
                    narrowed = _v35_safety_reason(
                        pick, baseline, state(board=board),
                        attack_is_available=attack_available,
                        lethal_available=False,
                    )
                    if (
                        baseline["action_type"] == "boss"
                        and pick["action_type"] not in ("boss", "end")
                        and shared == "preserve_fallback_boss_route"
                    ):
                        assert narrowed != "preserve_fallback_boss_route"
                        continue
                    assert narrowed == shared


# --- deployment provenance ------------------------------------------------


def test_v35_defaults_to_the_narrowed_shell():
    runtime = HybridRanker()
    snapshot = runtime.snapshot()

    assert snapshot["shell"] == "v35"
    assert snapshot["runtime_scope"] == (
        "v35_narrowed_shell_plus_recent_corpus_ranker"
    )
    assert not snapshot["errors"]


def test_v34_shell_can_be_restored_for_a_b_runs():
    os.environ["ALAKAZAM_ML_V35_SHELL"] = "v34"
    try:
        runtime = HybridRanker()
        assert runtime.snapshot()["shell"] == "v34"
        assert runtime._safety_reason(
            context("evolve"), context("attack", lethal=True), state(),
            attack_is_available=True, lethal_available=True,
        ) == "lethal_guard"
    finally:
        del os.environ["ALAKAZAM_ML_V35_SHELL"]


def test_v35_ships_the_v34_ranker_unchanged():
    """v35 is a runtime-only change; the model must be byte-identical."""
    model = json.loads((HERE / "ranker_model.json").read_text("utf-8"))

    assert model["tree_count"] == 2050
    assert len(model["trees"]) == 2050
    assert model["runtime_scope"] == "v34_yushin_recent_corpus_ranker"
