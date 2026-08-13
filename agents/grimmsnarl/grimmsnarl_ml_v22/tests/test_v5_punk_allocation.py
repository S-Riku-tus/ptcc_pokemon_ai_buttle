"""The v5 planner guard over the Punk Up target select.

The budget in ``fallback_policy.punk_search_budget`` is computed against one
promise - the body that just evolved ends the activation able to attack - so
these tests pin that the guard makes the promise hold and does nothing else.
Each also asserts the stand-down, because the measured teacher rates say the
ranker is already right here 100% of the time: the guard is insurance for a
tighter budget, and insurance that fires wider than its proof is a cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from ml_planner import Planner  # noqa: E402

TRIGGER_SERIAL = 87
BACKUP_SERIAL = 42
THIRD_SERIAL = 43


def _dark(count: int) -> list[dict[str, int]]:
    return [{"id": mf.DARK_ENERGY_ID} for _ in range(count)]


def _body(card_id: int, serial: int, energy: int) -> dict:
    return {
        "id": card_id, "serial": serial, "hp": 340.0, "maxHp": 340.0,
        "energies": _dark(energy),
    }


def _observation(active: dict, bench: list[dict]) -> dict:
    return {
        "current": {
            "turn": 5,
            "yourIndex": 0,
            "players": [
                {"active": [active], "bench": bench, "hand": [],
                 "prize": [{}] * 4},
                {"active": [{"id": 1, "hp": 100.0, "maxHp": 100.0}],
                 "bench": [], "prize": [{}] * 4},
            ],
        }
    }


def _select(bench_count: int, trigger_serial: int = TRIGGER_SERIAL) -> dict:
    options = [{"type": 3, "area": mf.AREA_ACTIVE, "index": 0,
                "playerIndex": 0}]
    options += [
        {"type": 3, "area": mf.AREA_BENCH, "index": slot, "playerIndex": 0}
        for slot in range(bench_count)
    ]
    return {
        "context": mf.CTX_ATTACH_FROM,
        "minCount": 1,
        "maxCount": 1,
        "effect": {"id": mf.GRIMMSNARL_EX_ID, "serial": trigger_serial,
                   "playerIndex": 0},
        "option": options,
    }


def _adjust(planner, observation, select, index, scores=None):
    return planner.adjust(observation, select, index, scores)


def test_energy_is_pulled_back_onto_the_trigger_that_cannot_attack_yet():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 0),
        [_body(mf.IMPIDIMP_ID, BACKUP_SERIAL, 0)],
    )
    # the ranker wants to fuel the Bench body first
    assert _adjust(planner, observation, _select(1), 1) == 0
    assert planner.stats["punk_alloc_trigger_overrides"] == 1


def test_it_stands_down_once_the_trigger_can_attack():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 2),
        [_body(mf.IMPIDIMP_ID, BACKUP_SERIAL, 0)],
    )
    assert _adjust(planner, observation, _select(1), 1) == 1
    assert planner.stats["punk_alloc_trigger_overrides"] == 0


def test_it_stands_down_when_the_trigger_is_not_on_the_menu():
    planner = Planner()
    observation = _observation(
        _body(mf.MORGREM_ID, BACKUP_SERIAL, 0),
        [_body(mf.IMPIDIMP_ID, THIRD_SERIAL, 0)],
    )
    assert _adjust(planner, observation, _select(1), 1) == 1
    assert planner.stats["punk_alloc_trigger_overrides"] == 0


def test_it_does_not_touch_a_ranker_that_already_fuels_the_trigger():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 1),
        [_body(mf.IMPIDIMP_ID, BACKUP_SERIAL, 0)],
    )
    assert _adjust(planner, observation, _select(1), 0) == 0
    assert planner.stats["punk_alloc_trigger_overrides"] == 0


def test_a_fifth_energy_never_goes_onto_a_body_already_holding_four():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 4),
        [_body(mf.IMPIDIMP_ID, BACKUP_SERIAL, 2)],
    )
    assert _adjust(planner, observation, _select(1), 0) == 1
    assert planner.stats["punk_alloc_stack_overrides"] == 1


def test_the_stack_cap_stands_down_when_every_body_is_as_full():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 4),
        [_body(mf.GRIMMSNARL_EX_ID, BACKUP_SERIAL, 4)],
    )
    assert _adjust(planner, observation, _select(1), 0) == 0
    assert planner.stats["punk_alloc_stack_overrides"] == 0


def test_the_trigger_rule_outranks_the_stack_cap():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 1),
        [_body(mf.GRIMMSNARL_EX_ID, BACKUP_SERIAL, 4)],
    )
    assert _adjust(planner, observation, _select(1), 1) == 0
    assert planner.stats["punk_alloc_trigger_overrides"] == 1
    assert planner.stats["punk_alloc_stack_overrides"] == 0


def test_the_ranker_keeps_its_ordering_inside_the_allowed_set():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 4),
        [_body(mf.IMPIDIMP_ID, BACKUP_SERIAL, 0),
         _body(mf.MORGREM_ID, THIRD_SERIAL, 1)],
    )
    scores = {0: 9.0, 1: 1.0, 2: 5.0}
    assert _adjust(planner, observation, _select(2), 0, scores) == 2


def test_another_ability_attach_select_is_left_alone():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 0),
        [_body(mf.IMPIDIMP_ID, BACKUP_SERIAL, 0)],
    )
    select = _select(1)
    select["effect"] = {"id": mf.MUNKIDORI_ID, "serial": 5, "playerIndex": 0}
    assert _adjust(planner, observation, select, 1) == 1
    assert planner.stats["punk_alloc_considered"] == 0


def test_a_single_candidate_select_is_left_alone():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 0), []
    )
    assert _adjust(planner, observation, _select(0), 0) == 0
    assert planner.stats["punk_alloc_considered"] == 0


def test_the_guard_never_raises_on_a_malformed_select():
    planner = Planner()
    observation = _observation(
        _body(mf.GRIMMSNARL_EX_ID, TRIGGER_SERIAL, 0), []
    )
    select = _select(1)
    select["option"] = [{"type": 3}, {"type": 3}]
    assert _adjust(planner, observation, select, 1) == 1
    assert planner.stats["errors"] == 0
