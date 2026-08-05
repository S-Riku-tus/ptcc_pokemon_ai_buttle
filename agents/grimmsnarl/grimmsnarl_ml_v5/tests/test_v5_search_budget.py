"""v5: the multi-pick searches take what the board wants, not what is allowed.

The counts asserted here are the ones the >= 1120 band actually plays over
3,710 same-60 replays, so a change that breaks one of these is a change away
from the measured elite policy, not away from an opinion.
"""

from __future__ import annotations

import sys
from pathlib import Path

from test_v4_static import (  # noqa: E402
    AreaType, Card, EnergyType, Pokemon, SelectContext, load_main,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

TRIGGER_SERIAL = 87


class FakeSelect:
    def __init__(self, effect=None, min_count=0, max_count=5):
        self.effect = effect
        self.minCount = min_count
        self.maxCount = max_count
        self.option = []


def _policy(main, board, *, effect_serial=TRIGGER_SERIAL, bench_max=5,
            context=None, effect_id=None):
    policy = main.GrimmsnarlPolicy.__new__(main.GrimmsnarlPolicy)
    policy.my_board = lambda: list(board)
    policy.me = type("P", (), {"bench": board[1:], "benchMax": bench_max})()
    effect = Card(main.C.GRIMMSNARL_EX)
    effect.serial = effect_serial
    policy.select = FakeSelect(effect=effect)
    policy.context = (
        SelectContext.ATTACH_TO if context is None else context
    )
    policy.effect_id = (
        main.C.GRIMMSNARL_EX if effect_id is None else effect_id
    )
    return policy


def _grim(main, serial, energy, appeared=True):
    body = Pokemon(main.C.GRIMMSNARL_EX,
                   energies=[EnergyType.DARKNESS] * energy)
    body.serial = serial
    body.appearThisTurn = appeared
    return body


def _base(main, card_id, serial, energy):
    body = Pokemon(card_id, energies=[EnergyType.DARKNESS] * energy)
    body.serial = serial
    body.appearThisTurn = False
    return body


# ----- Punk Up ---------------------------------------------------------------
def test_a_lone_fresh_grimmsnarl_takes_two_not_five():
    main = load_main()
    policy = _policy(main, [_grim(main, TRIGGER_SERIAL, 0)])
    assert policy.punk_search_budget(5) == 2


def test_one_hungry_backup_buys_one_more_card():
    main = load_main()
    policy = _policy(main, [
        _grim(main, TRIGGER_SERIAL, 0),
        _base(main, main.C.IMPIDIMP, 42, 0),
    ])
    assert policy.punk_search_budget(5) == 3


def test_two_hungry_backups_buy_two_more():
    main = load_main()
    policy = _policy(main, [
        _grim(main, TRIGGER_SERIAL, 0),
        _base(main, main.C.IMPIDIMP, 42, 0),
        _base(main, main.C.MORGREM, 43, 1),
    ])
    assert policy.punk_search_budget(5) == 4


def test_a_backup_already_able_to_attack_buys_nothing():
    main = load_main()
    policy = _policy(main, [
        _grim(main, TRIGGER_SERIAL, 0),
        _base(main, main.C.MORGREM, 43, 2),
    ])
    assert policy.punk_search_budget(5) == 2


def test_a_half_fuelled_trigger_still_takes_the_floor_of_two():
    main = load_main()
    policy = _policy(main, [_grim(main, TRIGGER_SERIAL, 1)])
    assert policy.punk_search_budget(5) == 2


def test_a_fuelled_trigger_with_nothing_to_feed_still_takes_two():
    # The elite band takes 2.38 on average here and 2 on 62% of the boards;
    # the floor is theirs, not ours.
    main = load_main()
    policy = _policy(main, [_grim(main, TRIGGER_SERIAL, 2)])
    assert policy.punk_search_budget(5) == 2


def test_the_budget_never_exceeds_what_the_select_offers():
    main = load_main()
    policy = _policy(main, [
        _grim(main, TRIGGER_SERIAL, 0),
        _base(main, main.C.IMPIDIMP, 42, 0),
        _base(main, main.C.MORGREM, 43, 0),
        _base(main, main.C.IMPIDIMP, 44, 0),
    ])
    assert policy.punk_search_budget(5) == 5
    assert policy.punk_search_budget(2) == 2
    assert policy.punk_search_budget(0) == 0


def test_the_trigger_is_identified_by_serial_not_by_energy():
    main = load_main()
    other = _grim(main, 99, 0, appeared=False)
    trigger = _grim(main, TRIGGER_SERIAL, 2)
    policy = _policy(main, [other, trigger])
    # trigger is full, the *other* Grimmsnarl is the hungry backup: 0 + 1 -> 2
    assert policy.punk_search_budget(5) == 2
    policy = _policy(main, [trigger, other], effect_serial=99)
    # now the hungry one is the trigger: deficit 2 + one hungry body 0 -> 2
    assert policy.punk_search_budget(5) == 2


def test_five_is_still_reachable_when_five_energy_are_wanted():
    main = load_main()
    policy = _policy(main, [
        _grim(main, TRIGGER_SERIAL, 0),
        _base(main, main.C.IMPIDIMP, 42, 0),
        _base(main, main.C.IMPIDIMP, 43, 0),
        _base(main, main.C.MORGREM, 44, 1),
    ])
    assert policy.punk_search_budget(5) == 5


def test_munkidori_is_not_a_punk_up_target_and_buys_nothing():
    main = load_main()
    policy = _policy(main, [
        _grim(main, TRIGGER_SERIAL, 0),
        _base(main, main.C.MUNKIDORI, 42, 0),
    ])
    assert policy.punk_search_budget(5) == 2


# ----- Buddy-Buddy Poffin ----------------------------------------------------
def _poffin_policy(main, board, bench_max=5):
    policy = _policy(main, board, bench_max=bench_max,
                     context=SelectContext.TO_BENCH,
                     effect_id=main.C.BUDDY_POFFIN)
    return policy


def test_poffin_takes_both_on_an_empty_board():
    main = load_main()
    policy = _poffin_policy(main, [_base(main, main.C.MUNKIDORI, 1, 0)])
    assert policy.poffin_budget(2) == 2


def test_poffin_takes_one_when_only_two_bench_slots_are_left():
    main = load_main()
    board = [_base(main, main.C.MUNKIDORI, 1, 0)] + [
        _base(main, main.C.SNORUNT, 10 + i, 0) for i in range(3)
    ]
    assert _poffin_policy(main, board).poffin_budget(2) == 1


def test_poffin_takes_one_once_the_marnie_line_is_two_bodies_deep():
    main = load_main()
    board = [
        _base(main, main.C.MUNKIDORI, 1, 0),
        _base(main, main.C.IMPIDIMP, 2, 0),
        _base(main, main.C.MORGREM, 3, 0),
    ]
    assert _poffin_policy(main, board).poffin_budget(2) == 1


def test_poffin_takes_none_with_no_bench_room():
    main = load_main()
    board = [_base(main, main.C.MUNKIDORI, 1, 0)] + [
        _base(main, main.C.SNORUNT, 10 + i, 0) for i in range(5)
    ]
    assert _poffin_policy(main, board).poffin_budget(2) == 0


# ----- the trim itself -------------------------------------------------------
def test_search_budget_trims_the_selection_and_counts_it():
    main = load_main()
    main.DIAG.clear()
    main.DIAG.update(main._fresh_diag())
    policy = _policy(main, [_grim(main, TRIGGER_SERIAL, 0)])
    assert policy.search_budget([0, 1, 2, 3, 4]) == [0, 1]
    assert main.DIAG["punk_search_offered"] == 5
    assert main.DIAG["punk_search_taken"] == 2
    assert main.DIAG["punk_search_trimmed"] == 1
    assert main.DIAG["punk_search_counts"][2] == 1


def test_search_budget_leaves_every_other_select_alone():
    main = load_main()
    policy = _policy(main, [_grim(main, TRIGGER_SERIAL, 0)],
                     context=SelectContext.MAIN, effect_id=None)
    assert policy.search_budget([0, 1, 2]) == [0, 1, 2]


def test_search_budget_respects_a_forced_minimum():
    main = load_main()
    policy = _policy(main, [_grim(main, TRIGGER_SERIAL, 2)])
    policy.select.minCount = 4
    assert policy.search_budget([0, 1, 2, 3, 4]) == [0, 1, 2, 3]
