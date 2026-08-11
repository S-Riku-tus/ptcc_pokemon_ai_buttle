"""Golden states for the v16 wall-break route.

Every board here is one the v15 ladder logs produced.  The extreme case is
episode 91663479: Cornerstone Mask Ogerpon ex Active, an empty opposing Bench
for all 24 turns, 21 Shadow Bullets, 0 prizes and a deck-out - a state where
Grimmsnarl ex can never take a prize and Marnie's Morgrem can.

Option shapes are the engine's, as in ``test_v15_attack_access``: a MAIN energy
attachment is ``type: 8`` with ``area``/``index`` into the hand and
``inPlayArea``/``inPlayIndex`` onto the board, retreat is ``type: 12``, and the
promotion after a retreat is a ``type: 3`` card select over our own Bench.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from wall_break import (  # noqa: E402
    ABILITY_BLOCKER_IDS,
    BREAKER_IDS,
    DARK_ATTACKS,
    EX_BLOCKER_IDS,
    WallBreakGuard,
)

DARK = mf.DARK_ENERGY_ID
GRIM = mf.GRIMMSNARL_EX_ID
MORGREM = mf.MORGREM_ID
IMPIDIMP = mf.IMPIDIMP_ID
SNORUNT = mf.SNORUNT_ID
MUNKIDORI = mf.MUNKIDORI_ID
FROSLASS = mf.FROSLASS_ID
BOSS = mf.BOSS_ID
PAD = mf.POKE_PAD_ID
SHADOW = mf.SHADOW_BULLET_ID
CORKSCREW_MORGREM = 936
CORKSCREW_IMPIDIMP = 935

CRUSTLE = 345           # prevents all damage from Pokemon ex
OGERPON = 117           # prevents all damage from Pokemon with an Ability

END = {"type": 14}
RETREAT = {"type": 12}


def poke(
    card_id: int, energy: int = 0, hp: int = 100, serial: int = 0
) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": hp,
        "serial": serial,
        "energies": [DARK] * energy,
        "energyCards": [{"id": DARK}] * energy,
    }


def attach(hand_index: int, *, bench_slot: int | None = None) -> dict:
    return {
        "type": 8,
        "area": mf.AREA_HAND,
        "index": hand_index,
        "inPlayArea": mf.AREA_ACTIVE if bench_slot is None else mf.AREA_BENCH,
        "inPlayIndex": 0 if bench_slot is None else bench_slot,
    }


def evolve(hand_index: int, *, bench_slot: int | None = None) -> dict:
    return {
        "type": 9,
        "area": mf.AREA_HAND,
        "index": hand_index,
        "inPlayArea": mf.AREA_ACTIVE if bench_slot is None else mf.AREA_BENCH,
        "inPlayIndex": 0 if bench_slot is None else bench_slot,
    }


def play(hand_index: int) -> dict:
    return {"type": 7, "index": hand_index}


def attack(attack_id: int) -> dict:
    return {"type": 13, "attackId": attack_id}


def observation(
    *,
    active: dict,
    bench: list[dict],
    hand: list[dict],
    options: list[dict],
    opponent_active: dict,
    opponent_bench: list[dict] | None = None,
    context: int = mf.MAIN_CONTEXT,
    energy_attached: bool = False,
    retreated: bool = False,
    supporter_played: bool = False,
    deck_count: int = 30,
    max_count: int = 1,
    turn: int = 5,
) -> dict:
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "retreated": retreated,
            "energyAttached": energy_attached,
            "supporterPlayed": supporter_played,
            "players": [
                {
                    "active": [active],
                    "bench": list(bench),
                    "hand": list(hand),
                    "discard": [],
                    "prize": [{}] * 4,
                    "deckCount": deck_count,
                    "handCount": len(hand),
                    "asleep": False,
                    "paralyzed": False,
                },
                {
                    "active": [opponent_active],
                    "bench": list(opponent_bench or []),
                    "discard": [],
                    "prize": [{}] * 4,
                    "deckCount": 30,
                },
            ],
            "stadium": [],
        },
        "select": {
            "context": context,
            "minCount": 1,
            "maxCount": max_count,
            "option": list(options),
        },
    }


def walled(
    *,
    active: dict | None = None,
    bench: list[dict] | None = None,
    hand: list[dict] | None = None,
    options: list[dict],
    wall: int = OGERPON,
    wall_hp: int = 210,
    **kwargs,
) -> dict:
    """Episode 91663479's shape: an immune Active over an empty Bench."""
    return observation(
        active=active if active is not None else poke(GRIM, 2, 320, serial=1),
        bench=(
            bench if bench is not None else [poke(MORGREM, 2, 100, 2)]
        ),
        hand=hand if hand is not None else [{"id": PAD}, {"id": DARK}],
        options=options,
        opponent_active=poke(wall, hp=wall_hp, serial=90),
        **kwargs,
    )


# ----- the premise ----------------------------------------------------------
def test_dark_attacks_match_the_card_database() -> None:
    from cg.api import all_attack, all_card_data

    cards = {card.cardId: card for card in all_card_data()}
    attacks = {a.attackId: a for a in all_attack()}
    for card_id, (attack_id, need, damage) in DARK_ATTACKS.items():
        data = attacks[attack_id]
        assert attack_id in (cards[card_id].attacks or [])
        assert float(data.damage) == damage
        assert len(data.energies or []) == need
        # Every cost symbol must be payable by Basic Darkness: {D} itself, or
        # {C}, which any Energy pays.
        assert all(
            int(symbol) in (DARK, 0) for symbol in (data.energies or [])
        )


def test_the_bodies_left_out_can_never_attack_on_this_deck() -> None:
    """Froslass, Munkidori and Snorunt need Energy this deck does not run."""
    from cg.api import all_attack, all_card_data

    cards = {card.cardId: card for card in all_card_data()}
    attacks = {a.attackId: a for a in all_attack()}
    for card_id in (FROSLASS, MUNKIDORI, SNORUNT):
        assert card_id not in DARK_ATTACKS
        for attack_id in (cards[card_id].attacks or []):
            energies = attacks[attack_id].energies or []
            payable = all(int(symbol) in (DARK, 0) for symbol in energies)
            damage = float(attacks[attack_id].damage or 0)
            assert not (payable and damage > 0), (
                f"{card_id} can pay for {attack_id} after all"
            )


def test_blocker_split_partitions_the_merged_feature_set() -> None:
    assert EX_BLOCKER_IDS | ABILITY_BLOCKER_IDS == mf.EX_DAMAGE_BLOCKER_IDS
    assert not EX_BLOCKER_IDS & ABILITY_BLOCKER_IDS
    # The breakers are exactly the bodies neither Ability stops.
    assert BREAKER_IDS == {MORGREM, IMPIDIMP}
    assert not BREAKER_IDS & mf.ABILITY_HOLDER_IDS
    assert GRIM not in BREAKER_IDS


# ----- BREAK ----------------------------------------------------------------
def test_dead_shadow_bullet_becomes_a_retreat_to_the_breaker() -> None:
    guard = WallBreakGuard()
    obs = walled(options=[attack(SHADOW), RETREAT, END])
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["retreat_forced"] == 1
    assert guard.stats["dead_shadow_replaced"] == 1
    assert guard.stats["dead_swing_turns"] == 1


def test_the_promotion_after_that_retreat_takes_the_breaker() -> None:
    guard = WallBreakGuard()
    obs = walled(
        bench=[poke(MUNKIDORI, 1, 110, serial=3), poke(MORGREM, 2, 100, 2)],
        options=[attack(SHADOW), RETREAT, END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1

    promote = walled(
        bench=[poke(MUNKIDORI, 1, 110, serial=3), poke(MORGREM, 2, 100, 2)],
        context=mf.CTX_SWITCH,
        options=[
            {"type": 3, "area": mf.AREA_BENCH, "index": 0, "playerIndex": 0},
            {"type": 3, "area": mf.AREA_BENCH, "index": 1, "playerIndex": 0},
        ],
    )
    assert guard.adjust(promote, promote["select"], 0, [0]) == 1
    assert guard.stats["promote_forced"] == 1


def test_an_unfuelled_breaker_is_fuelled_instead_of_swinging() -> None:
    guard = WallBreakGuard()
    obs = walled(
        bench=[poke(MORGREM, 0, 100, serial=2)],
        options=[attack(SHADOW), attach(1), attach(1, bench_slot=0), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 2
    assert guard.stats["fuel_attach_forced"] == 1


def test_an_attachment_aimed_elsewhere_is_redirected_to_the_breaker() -> None:
    guard = WallBreakGuard()
    obs = walled(
        bench=[poke(MORGREM, 0, 100, serial=2), poke(MUNKIDORI, 0, 110, 3)],
        options=[attach(1, bench_slot=1), attach(1, bench_slot=0), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["attachment_redirected"] == 1


def test_an_active_breaker_attacks_instead_of_ending() -> None:
    guard = WallBreakGuard()
    obs = walled(
        active=poke(MORGREM, 2, 100, serial=2),
        bench=[poke(GRIM, 2, 320, serial=1)],
        options=[END, attack(CORKSCREW_MORGREM)],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["attack_forced"] == 1
    assert guard.stats["end_replaced"] == 1


def test_a_breaker_already_swinging_is_left_alone() -> None:
    guard = WallBreakGuard()
    obs = walled(
        active=poke(MORGREM, 2, 100, serial=2),
        bench=[poke(GRIM, 2, 320, serial=1)],
        options=[attack(CORKSCREW_MORGREM), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["route_compatible_kept"] == 1


def test_setup_plays_are_untouched_because_main_repeats() -> None:
    """Only a closing action or the attachment can lose the route."""
    guard = WallBreakGuard()
    obs = walled(options=[play(0), attack(SHADOW), RETREAT, END])
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["retreat_forced"] == 0


# ----- where it must stand down --------------------------------------------
def test_a_damageable_active_is_v15_exactly() -> None:
    guard = WallBreakGuard()
    obs = observation(
        active=poke(GRIM, 2, 320, serial=1),
        bench=[poke(MORGREM, 2, 100, serial=2)],
        hand=[{"id": PAD}, {"id": DARK}],
        options=[attack(SHADOW), RETREAT, END],
        opponent_active=poke(1, hp=300, serial=90),
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["dead_swing_decisions"] == 0


def test_a_bench_thirty_that_takes_a_prize_keeps_the_swing() -> None:
    guard = WallBreakGuard()
    obs = walled(
        options=[attack(SHADOW), RETREAT, END],
        opponent_bench=[poke(2, hp=30, serial=91)],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["dead_swing_decisions"] == 0


def test_a_playable_boss_that_takes_a_prize_is_left_to_the_ranker() -> None:
    guard = WallBreakGuard()
    obs = walled(
        hand=[{"id": BOSS}, {"id": DARK}],
        options=[attack(SHADOW), play(0), RETREAT, END],
        opponent_bench=[poke(2, hp=100, serial=91)],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["boss_prize_deferred"] >= 1


def test_a_spent_supporter_no_longer_defers_to_boss() -> None:
    guard = WallBreakGuard()
    obs = walled(
        hand=[{"id": BOSS}, {"id": DARK}],
        options=[attack(SHADOW), RETREAT, END],
        opponent_bench=[poke(2, hp=100, serial=91)],
        supporter_played=True,
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["retreat_forced"] == 1


def test_a_route_that_cannot_finish_before_the_deck_does_is_refused() -> None:
    guard = WallBreakGuard()
    obs = walled(
        bench=[poke(IMPIDIMP, 1, 70, serial=2)],  # 10 a swing into 210 HP
        options=[attack(SHADOW), RETREAT, END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["breaker_too_slow"] == 1


def test_no_breaker_in_play_leaves_the_swing_alone() -> None:
    guard = WallBreakGuard()
    obs = walled(
        bench=[poke(MUNKIDORI, 1, 110, serial=3)],
        options=[attack(SHADOW), RETREAT, END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["no_breaker_in_play"] == 1


def test_crustle_walls_the_ex_but_not_the_morgrem() -> None:
    guard = WallBreakGuard()
    obs = walled(
        wall=CRUSTLE, wall_hp=150, options=[attack(SHADOW), RETREAT, END]
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["retreat_forced"] == 1


# ----- the sacrifice gate ---------------------------------------------------
def dying_breaker(**kwargs) -> dict:
    """The breaker is knocked out by their next swing."""
    return walled(
        options=[attack(SHADOW), RETREAT, END],
        opponent_bench=[poke(GRIM, 2, 320, serial=91)],
        **kwargs,
    )


def test_a_breaker_that_dies_next_turn_waits_for_the_wall_to_prove_permanent(
) -> None:
    guard = WallBreakGuard()
    obs = dying_breaker(turn=5)
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["breaker_would_be_sacrificed"] == 1

    # Second consecutive own turn in the same dead state: nothing else is left.
    later = dying_breaker(turn=7)
    guard.note(later)
    assert guard.adjust(later, later["select"], 0, [0]) == 1
    assert guard.stats["retreat_forced"] == 1


# ----- PRESERVE -------------------------------------------------------------
def test_the_last_breaker_is_not_evolved_away_under_a_wall() -> None:
    guard = WallBreakGuard()
    obs = walled(
        active=poke(GRIM, 2, 320, serial=1),
        bench=[poke(MORGREM, 2, 100, serial=2)],
        hand=[{"id": GRIM}, {"id": DARK}],
        options=[evolve(0, bench_slot=0), attack(SHADOW), RETREAT, END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [2]) == 2
    assert guard.stats["last_breaker_evolve_refused"] == 1


def test_a_second_breaker_makes_the_evolution_free_again() -> None:
    guard = WallBreakGuard()
    obs = walled(
        active=poke(GRIM, 2, 320, serial=1),
        bench=[poke(MORGREM, 2, 100, serial=2), poke(MORGREM, 2, 100, 4)],
        hand=[{"id": GRIM}, {"id": DARK}],
        options=[evolve(0, bench_slot=0), attack(SHADOW), RETREAT, END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["last_breaker_evolve_refused"] == 0


def test_without_a_fuelled_attacker_the_evolution_still_happens() -> None:
    """Punk Up is how the attacker gets paid for; never refuse that."""
    guard = WallBreakGuard()
    obs = walled(
        active=poke(SNORUNT, 0, 70, serial=1),
        bench=[poke(MORGREM, 2, 100, serial=2)],
        hand=[{"id": GRIM}, {"id": DARK}],
        options=[evolve(0, bench_slot=0), attack(SHADOW), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["last_breaker_evolve_refused"] == 0


# ----- invariants -----------------------------------------------------------
def test_multi_pick_selects_are_never_touched() -> None:
    guard = WallBreakGuard()
    obs = walled(options=[attack(SHADOW), RETREAT, END], max_count=2)
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0


def test_a_new_game_resets_the_dead_turn_counter() -> None:
    guard = WallBreakGuard()
    first = dying_breaker(turn=9)
    guard.note(first)
    guard.adjust(first, first["select"], 0, [0])
    guard.note(dying_breaker(turn=11))
    assert guard._dead_turns == 1
    guard.note(dying_breaker(turn=1))
    assert guard._dead_turns == 0
    assert guard.stats["new_game_detected"] == 1


def test_a_malformed_board_returns_the_callers_index() -> None:
    guard = WallBreakGuard()
    broken = {"current": {"players": []}, "select": {"context": 0}}
    assert guard.adjust(broken, {"option": [END], "maxCount": 1}, 0, [0]) == 0
