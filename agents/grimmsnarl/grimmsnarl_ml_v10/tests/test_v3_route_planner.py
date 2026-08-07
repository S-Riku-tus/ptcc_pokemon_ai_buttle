"""The v3 route/heal arithmetic, pinned to the boards it was written for.

Every case here is a shape the 59-game v2 ladder run actually produced, so a
regression shows up as a failing assertion rather than as a rating drop three
hundred games later:

* 89678716 - Boss gusted a 20 HP Mega Lucario ex the free Bench-30 already
  killed, spent the 180 on it, and left the 340 HP Active untouched. The route
  table has to score that gust at 3 prizes and three other gusts at 4.
* 89723274 - Adrena-Brain moved counters off an Impidimp while the damaged
  Grimmsnarl ex was on offer.
* 89703351 / 89698181 - one heal leaves the attacker inside the opponent's
  range and two take it out, which is a per-turn budget question.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_features as F  # noqa: E402
from ml_planner import Planner  # noqa: E402

GRIMMSNARL_EX = 648
IMPIDIMP = 646
MUNKIDORI = 112
FROSLASS = 104
SNORUNT = 860
DARK_ENERGY = 7
MEGA_LUCARIO_EX = 678       # 440 HP Mega ex, 3 prizes
LUCARIO_SUPPORT = 676       # 110 HP single-prize body
RIOLU = 673                 # 80 HP single-prize body
LUNATONE = 675              # 110 HP single-prize body


def body(card_id: int, hp: int, max_hp: int | None = None,
         energies: int = 0) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": max_hp if max_hp is not None else hp,
        "energies": [DARK_ENERGY] * energies,
        "energyCards": [DARK_ENERGY] * energies,
        "tools": [],
        "preEvolution": [],
    }


def player(active: list[dict], bench: list[dict]) -> dict:
    return {
        "active": active, "bench": bench, "hand": [], "discard": [],
        "prize": [None] * 6, "deckCount": 30, "handCount": 5, "benchMax": 5,
    }


def state(me: dict, opponent: dict, turn: int = 8, stadium: int = -1) -> dict:
    current = {
        "turn": turn, "yourIndex": 0, "players": [me, opponent],
        "firstPlayer": 0,
    }
    if stadium >= 0:
        current["stadium"] = [{"id": stadium}]
    return current


def bench_select(context: int, count: int, owner: int = 1) -> dict:
    return {
        "context": context, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": 3, "area": 5, "index": index, "playerIndex": owner}
            for index in range(count)
        ],
    }


# ----- turn_routes -----------------------------------------------------------

def test_route_table_reproduces_the_89678716_board() -> None:
    """Gusting the 20 HP Mega takes 3; gusting anything else takes 4."""
    opponent = player(
        [body(MEGA_LUCARIO_EX, 340, 340)],
        [
            body(LUCARIO_SUPPORT, 110),
            body(MEGA_LUCARIO_EX, 20, 440),
            body(RIOLU, 80),
            body(LUNATONE, 80, 110),
        ],
    )
    me = player([body(GRIMMSNARL_EX, 320, 320, energies=2)], [])
    routes = F.turn_routes(state(me, opponent), opponent)

    assert routes["no_boss_active_prizes"] == 0     # 340 HP survives the 180
    assert routes["no_boss_snipe_prizes"] == 3      # Bench-30 kills the Mega
    totals = {entry["index"]: entry["total"] for entry in routes["per_target"]}
    assert totals == {0: 4, 1: 3, 2: 4, 3: 4}
    mega = routes["per_target"][1]
    assert mega["dies_to_snipe_alone"] is True
    assert mega["active_prizes"] == 3 and mega["snipe_prizes"] == 0


def test_route_counts_the_displaced_active_as_a_snipe_target() -> None:
    """Gusting benches their Active, so the Bench-30 can finish it there."""
    opponent = player([body(RIOLU, 20)], [body(LUCARIO_SUPPORT, 110)])
    me = player([body(GRIMMSNARL_EX, 320, 320, energies=2)], [])
    routes = F.turn_routes(state(me, opponent), opponent)
    entry = routes["per_target"][0]
    assert entry["active_prizes"] == 1              # 180 kills the 110
    assert entry["snipe_prizes"] == 1               # 30 finishes the old Active
    assert entry["total"] == 2


# ----- the Boss-target override ----------------------------------------------

def make_planner_board() -> tuple[dict, dict]:
    opponent = player(
        [body(MEGA_LUCARIO_EX, 340, 340)],
        [
            body(LUCARIO_SUPPORT, 110),
            body(MEGA_LUCARIO_EX, 20, 440),
            body(RIOLU, 80),
            body(LUNATONE, 80, 110),
        ],
    )
    me = player([body(GRIMMSNARL_EX, 320, 320, energies=2)], [])
    return me, opponent


def test_boss_override_leaves_the_free_kill_to_the_snipe() -> None:
    me, opponent = make_planner_board()
    observation = {"current": state(me, opponent)}
    select = bench_select(F.CTX_SWITCH, 4)
    planner = Planner()
    scores = {0: 0.1, 1: 5.0, 2: 0.9, 3: 0.2}

    assert planner.adjust(observation, select, 1, scores) == 2
    assert planner.stats["boss_route_overrides"] == 1


def test_boss_override_holds_when_the_target_survives_the_snipe() -> None:
    me, opponent = make_planner_board()
    observation = {"current": state(me, opponent)}
    select = bench_select(F.CTX_SWITCH, 4)
    planner = Planner()
    # Slot 3 takes 4 prizes and does not die to the Bench-30 on its own, so
    # there is nothing dominated about it and the ranker keeps its answer.
    assert planner.adjust(observation, select, 3, {}) == 3
    assert planner.stats["boss_route_overrides"] == 0


def test_boss_override_needs_a_fuelled_attacker_in_the_active_spot() -> None:
    me, opponent = make_planner_board()
    me = player([body(IMPIDIMP, 70)], [body(GRIMMSNARL_EX, 320, 320, 2)])
    observation = {"current": state(me, opponent)}
    planner = Planner()
    select = bench_select(F.CTX_SWITCH, 4)
    assert planner.adjust(observation, select, 1, {}) == 1
    assert planner.stats["boss_route_considered"] == 0


def test_boss_override_ignores_promoting_our_own_body() -> None:
    me, opponent = make_planner_board()
    observation = {"current": state(me, opponent)}
    select = bench_select(F.CTX_TO_ACTIVE, 2, owner=0)
    planner = Planner()
    assert planner.adjust(observation, select, 1, {}) == 1
    assert planner.stats["boss_route_considered"] == 0


# ----- Adrena-Brain arithmetic ----------------------------------------------

def test_movable_counters_caps_at_three_and_at_damage_present() -> None:
    assert F.movable_counters(body(GRIMMSNARL_EX, 320, 320)) == 0
    assert F.movable_counters(body(GRIMMSNARL_EX, 310, 320)) == 1
    assert F.movable_counters(body(GRIMMSNARL_EX, 140, 320)) == 3


def test_heals_needed_counts_moves_not_damage() -> None:
    assert F.heals_needed(320, 180) == 0        # already survives
    assert F.heals_needed(140, 180) == 2        # 170 still dies, 200 lives
    assert F.heals_needed(180, 180) == 1
    assert F.heals_needed(60, 320) == 9         # capped, unsavable


def heal_select(count: int) -> dict:
    return {
        "context": F.CTX_REMOVE_DAMAGE_COUNTER, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": 3, "area": 4 if index == 0 else 5, "index": 0 if index == 0
             else index - 1, "playerIndex": 0}
            for index in range(count)
        ],
    }


def threat_board(active_hp: int, munkidori: int) -> tuple[dict, dict]:
    """Our attacker at ``active_hp`` in the mirror, facing Shadow Bullet's 180.

    The mirror is where this matters most - v2 went 9-9 there against 60.6% for
    the top five pilots - and 180 into a 320 HP body is exactly the threshold
    case: at 140 HP one Adrena-Brain move reaches 170 and still dies, two reach
    200 and live.
    """
    me = player(
        [body(GRIMMSNARL_EX, active_hp, 320, energies=2)],
        [body(IMPIDIMP, 40, 70)] + [
            body(MUNKIDORI, 110, 110, energies=1) for _ in range(munkidori)
        ],
    )
    opponent = player([body(GRIMMSNARL_EX, 320, 320, energies=2)], [])
    return me, opponent


def test_heal_override_saves_the_attacker_across_two_munkidori() -> None:
    me, opponent = threat_board(140, munkidori=2)
    observation = {"current": state(me, opponent)}
    select = heal_select(2)          # slot 0 Grimmsnarl, slot 1 Impidimp
    planner = Planner()
    planner.note(observation, {"context": 0, "option": []}, 0)

    assert F.incoming_damage(F._in_play(opponent), me["active"][0]) == 180.0
    assert F.heals_needed(140, 180) == 2
    assert planner.adjust(observation, select, 1, {0: 0.1, 1: 9.0}) == 0
    assert planner.stats["heal_overrides"] == 1


def test_heal_override_stands_down_when_attacker_cannot_be_saved() -> None:
    me, opponent = threat_board(140, munkidori=1)
    observation = {"current": state(me, opponent)}
    planner = Planner()
    # One move reaches 170 against 200: the attacker dies either way, so the
    # ranker's answer is not dominated and the planner keeps it.
    assert planner.adjust(observation, heal_select(2), 1, {}) == 1
    assert planner.stats["heal_overrides"] == 0


def test_heal_override_stands_down_when_ranker_already_saves() -> None:
    me, opponent = threat_board(180, munkidori=1)
    observation = {"current": state(me, opponent)}
    planner = Planner()
    assert planner.adjust(observation, heal_select(2), 0, {}) == 0
    assert planner.stats["heal_overrides"] == 0


def test_heal_budget_shrinks_as_munkidori_fire_within_one_turn() -> None:
    me, opponent = threat_board(140, munkidori=2)
    current = state(me, opponent)
    observation = {"current": current}
    main = {
        "context": F.MAIN_CONTEXT, "minCount": 1, "maxCount": 1,
        "option": [{"type": 10, "area": 5, "index": 1}, {"type": 14}],
    }
    planner = Planner()
    planner.note(observation, main, 0)
    assert planner.heals_available(current) == 2      # this one plus one more
    planner.note(observation, main, 0)
    assert planner.heals_available(current) == 1      # the second is in flight
    planner.note({"current": state(me, opponent, turn=9)}, main, 0)
    assert planner.heals_available(current) == 2      # new turn, budget resets


# ----- Freezing Shroud ledger -----------------------------------------------

def test_shroud_targets_count_ability_bodies_but_not_froslass() -> None:
    cards = [
        body(GRIMMSNARL_EX, 320), body(MUNKIDORI, 110), body(FROSLASS, 90),
        body(SNORUNT, 70), body(IMPIDIMP, 70),
    ]
    ids = {int(c["id"]) for c in F.shroud_targets(cards)}
    assert ids == {GRIMMSNARL_EX, MUNKIDORI}
    assert FROSLASS not in F.ABILITY_POKEMON_IDS
    # v2 could only recognise three card ids; the generated set is the pool.
    assert len(F.ABILITY_POKEMON_IDS) > 100


def test_froslass_guard_refuses_only_a_conceded_prize() -> None:
    me = player(
        [body(GRIMMSNARL_EX, 320, 320, energies=2)],
        [body(SNORUNT, 70), body(MUNKIDORI, 10, 110, energies=1)],
    )
    opponent = player([body(MEGA_LUCARIO_EX, 440, 440)], [])
    me["hand"] = [body(FROSLASS, 90)]
    observation = {"current": state(me, opponent)}
    select = {
        "context": F.MAIN_CONTEXT, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": 9, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
            {"type": 14},
        ],
    }
    planner = Planner()
    assert planner.adjust(observation, select, 0, {1: 0.5}) == 1
    assert planner.stats["froslass_overrides"] == 1

    # Same board, but their Munkidori dies to the same checkup: no longer a
    # one-sided loss, so the judgement goes back to the ranker.
    opponent["bench"] = [body(MUNKIDORI, 10, 110)]
    planner = Planner()
    assert planner.adjust(observation, select, 0, {1: 0.5}) == 0
    assert planner.stats["froslass_overrides"] == 0
