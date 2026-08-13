from __future__ import annotations

import json
import sys
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
V22 = AGENT.parent / "grimmsnarl_ml_v22"
sys.path.insert(0, str(AGENT))

from ml_runtime import (  # noqa: E402
    OPENING_TEACHER_CODE,
    V22_TEACHER_CODE,
    Ranker,
)
from v23_strategy import StrategyPlanner, own_turn  # noqa: E402


def _observation(*, turn: int, first: int = 0, your: int = 0, opponent_ids=()):
    opponent = {
        "active": ([{"id": opponent_ids[0]}] if opponent_ids else []),
        "bench": [{"id": card_id} for card_id in opponent_ids[1:]],
    }
    players = [{"active": [], "bench": []}, opponent]
    if your == 1:
        players.reverse()
    return {
        "current": {
            "turn": turn,
            "firstPlayer": first,
            "yourIndex": your,
            "players": players,
        },
        "select": {"context": 7},
    }


def _row(**updates):
    row = {
        "attacker_body_count": 0,
        "backup_attacker_ready": 0,
        "marnie_body_count": 1,
        "field_646": 0,
        "field_647": 0,
        "ctx_completes_candy_route": 0,
        "morgrem_route_available": 0,
        "candidate_is_grimmsnarl": 0,
        "candidate_is_impidimp": 0,
        "candidate_is_morgrem": 0,
        "candidate_is_froslass": 0,
        "evolve_into_attacker": 0,
        "candy_into_attacker": 0,
        "triggers_punk_up": 0,
        "is_bench": 0,
    }
    row.update(updates)
    return row


def test_own_turn_is_seat_relative() -> None:
    assert own_turn({"turn": 0, "yourIndex": 0, "firstPlayer": 0}) == 0
    assert own_turn({"turn": 1, "yourIndex": 0, "firstPlayer": 0}) == 1
    assert own_turn({"turn": 3, "yourIndex": 0, "firstPlayer": 0}) == 2
    assert own_turn({"turn": 2, "yourIndex": 1, "firstPlayer": 0}) == 1
    assert own_turn({"turn": 4, "yourIndex": 1, "firstPlayer": 0}) == 2


def test_ranker_uses_v8_only_for_own_turns_one_and_two() -> None:
    ranker = object.__new__(Ranker)
    ranker.teacher_code = V22_TEACHER_CODE
    for turn in (1, 3):
        assert ranker._teacher_code_for(_observation(turn=turn)) == OPENING_TEACHER_CODE
    assert ranker._teacher_code_for(_observation(turn=0)) == V22_TEACHER_CODE
    assert ranker._teacher_code_for(_observation(turn=5)) == V22_TEACHER_CODE
    assert ranker._teacher_code_for(
        _observation(turn=4, your=1, first=0)
    ) == OPENING_TEACHER_CODE
    assert ranker._teacher_code_for(
        _observation(turn=6, your=1, first=0)
    ) == V22_TEACHER_CODE


def test_opening_search_takes_a_card_that_completes_candy_route() -> None:
    planner = StrategyPlanner()
    observation = _observation(turn=3)
    rows = [_row(), _row(ctx_completes_candy_route=1)]
    moved = planner.adjust(
        observation, observation["select"], 0, {0: 9.0, 1: 1.0}, rows
    )
    assert moved == 1
    assert planner.snapshot()["opening_complete_overrides"] == 1


def test_opening_search_takes_morgrem_bridge_over_extra_impidimp() -> None:
    planner = StrategyPlanner()
    observation = _observation(turn=3)
    rows = [
        _row(field_646=1, candidate_is_impidimp=1),
        _row(field_646=1, candidate_is_morgrem=1),
    ]
    moved = planner.adjust(
        observation, observation["select"], 0, {0: 9.0, 1: 1.0}, rows
    )
    assert moved == 1
    assert planner.snapshot()["opening_bridge_overrides"] == 1


def test_alakazam_search_builds_backup_before_optional_engine() -> None:
    planner = StrategyPlanner()
    observation = _observation(turn=7, opponent_ids=(743, 742))
    rows = [
        _row(
            attacker_body_count=1,
            field_646=1,
            marnie_body_count=2,
            candidate_is_froslass=1,
        ),
        _row(
            attacker_body_count=1,
            field_646=1,
            marnie_body_count=2,
            candidate_is_morgrem=1,
        ),
    ]
    moved = planner.adjust(
        observation, observation["select"], 0, {0: 9.0, 1: 1.0}, rows
    )
    assert moved == 1
    assert planner.snapshot()["alakazam_search_overrides"] == 1


def test_continuity_rule_does_not_touch_the_same_non_alakazam_board() -> None:
    planner = StrategyPlanner()
    observation = _observation(turn=7, opponent_ids=(648,))
    rows = [
        _row(attacker_body_count=1, field_646=1, candidate_is_froslass=1),
        _row(attacker_body_count=1, field_646=1, candidate_is_morgrem=1),
    ]
    assert planner.adjust(
        observation, observation["select"], 0, {0: 9.0, 1: 1.0}, rows
    ) == 0


def test_model_and_deck_are_frozen_from_v22() -> None:
    for name in (
        "deck.csv",
        "fallback_policy.py",
        "ml_features.py",
        "ml_planner.py",
        "policy_base.py",
        "ranker_model.json",
    ):
        assert (AGENT / name).read_bytes() == (V22 / name).read_bytes(), name

    metadata = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "grimmsnarl_ml_v23"
    assert metadata["parent_agent"] == "grimmsnarl_ml_v22"
    assert metadata["deck_changed"] is False
    assert metadata["ranker"]["trees_changed"] is False
