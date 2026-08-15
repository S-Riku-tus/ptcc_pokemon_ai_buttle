"""v21 preserves the *fastest* route to a wall, not merely the last body.

v16-v20 refused a Grimmsnarl ex evolution only when exactly one Morgrem or
Impidimp was in play.  Episode 92168220 turn 13 is the hole: Crustle Active,
two Morgrem on our board, only one of them holding the two Darkness that
Corkscrew Punch needs.  ``len(breakers) == 2`` disarmed the guard, the fuelled
Morgrem became a Grimmsnarl ex that could not damage Crustle, and the game
ended on nine dead Shadow Bullets, zero prizes and a deck-out.

These tests pin both halves of the replacement: the fastest route is preserved
even when other breakers exist, and everything v17 already preserved still is.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from wall_break import WallBreakGuard  # noqa: E402

DARK = mf.DARK_ENERGY_ID
GRIM = mf.GRIMMSNARL_EX_ID
MORGREM = mf.MORGREM_ID
IMPIDIMP = mf.IMPIDIMP_ID
CRUSTLE = 345
SHADOW = mf.SHADOW_BULLET_ID


def poke(card_id: int, energy: int, hp: int, serial: int) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": hp,
        "serial": serial,
        "energies": [DARK] * energy,
        "energyCards": [{"id": DARK}] * energy,
    }


def evolve(hand_index: int, bench_slot: int) -> dict:
    return {
        "type": 9,
        "area": mf.AREA_HAND,
        "index": hand_index,
        "inPlayArea": mf.AREA_BENCH,
        "inPlayIndex": bench_slot,
    }


def board(bench: list[dict], options: list[dict]) -> dict:
    """Episode 92168220's shape: Crustle Active, a fuelled Grimmsnarl of ours."""
    return {
        "current": {
            "turn": 13,
            "yourIndex": 0,
            "players": [
                {
                    "active": [poke(GRIM, 2, 320, 1)],
                    "bench": bench,
                    "hand": [{"id": GRIM}],
                    "discard": [],
                    "prize": [{}] * 5,
                    "deckCount": 20,
                    "asleep": False,
                    "paralyzed": False,
                },
                {
                    "active": [poke(CRUSTLE, 0, 160, 90)],
                    "bench": [poke(CRUSTLE, 0, 160, 91)],
                    "discard": [],
                    "prize": [{}] * 4,
                    "deckCount": 20,
                },
            ],
            "stadium": [],
            "retreated": False,
            "energyAttached": False,
            "supporterPlayed": False,
        },
        "select": {
            "context": mf.MAIN_CONTEXT,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
    }


def test_fuelled_morgrem_is_preserved_over_an_empty_second_morgrem() -> None:
    """The 92168220 board: refuse the evolution onto the only fuelled route."""
    guard = WallBreakGuard()
    obs = board(
        [poke(MORGREM, 2, 100, 24), poke(MORGREM, 0, 100, 23)],
        [evolve(0, 0), {"type": 13, "attackId": SHADOW}, {"type": 14}],
    )
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 0, [0]) != 0
    assert guard.stats["last_breaker_evolve_refused"] == 1
    assert guard.stats["last_breaker_fastest_route"] == 1
    assert guard.stats["last_breaker_only_body"] == 0


def test_the_slower_morgrem_may_still_be_evolved() -> None:
    """Consuming the body that was not going to break the wall is fine."""
    guard = WallBreakGuard()
    obs = board(
        [poke(MORGREM, 2, 100, 24), poke(MORGREM, 0, 100, 23)],
        [evolve(0, 1), {"type": 13, "attackId": SHADOW}, {"type": 14}],
    )
    guard.note(obs)

    guard.adjust(obs, obs["select"], 0, [0])
    assert guard.stats["last_breaker_evolve_refused"] == 0
    assert guard.stats["last_breaker_fastest_route"] == 0


def test_equal_routes_are_not_preserved() -> None:
    """Two interchangeable fuelled Morgrem: spending one keeps the route."""
    guard = WallBreakGuard()
    obs = board(
        [poke(MORGREM, 2, 100, 24), poke(MORGREM, 2, 100, 23)],
        [evolve(0, 0), {"type": 13, "attackId": SHADOW}, {"type": 14}],
    )
    guard.note(obs)

    guard.adjust(obs, obs["select"], 0, [0])
    assert guard.stats["last_breaker_evolve_refused"] == 0


def test_single_slow_impidimp_is_still_preserved() -> None:
    """v17's invariant: the count disjunct keeps a too-slow lone breaker."""
    guard = WallBreakGuard()
    obs = board(
        [poke(IMPIDIMP, 0, 70, 2)],
        [evolve(0, 0), {"type": 13, "attackId": SHADOW}, {"type": 14}],
    )
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 0, [0]) != 0
    assert guard.stats["last_breaker_evolve_refused"] == 1
    assert guard.stats["last_breaker_only_body"] == 1


def test_no_ready_grimmsnarl_means_no_preserve() -> None:
    """Declining must never cost the attacker itself."""
    guard = WallBreakGuard()
    obs = board(
        [poke(MORGREM, 2, 100, 24), poke(MORGREM, 0, 100, 23)],
        [evolve(0, 0), {"type": 14}],
    )
    obs["current"]["players"][0]["active"] = [poke(MORGREM, 0, 100, 1)]
    guard.note(obs)

    guard.adjust(obs, obs["select"], 0, [0])
    assert guard.stats["last_breaker_evolve_refused"] == 0


def test_no_wall_means_the_guard_never_looks() -> None:
    """A damageable Active is not a dead swing, so PRESERVE stays out."""
    guard = WallBreakGuard()
    obs = board(
        [poke(MORGREM, 2, 100, 24), poke(MORGREM, 0, 100, 23)],
        [evolve(0, 0), {"type": 13, "attackId": SHADOW}, {"type": 14}],
    )
    obs["current"]["players"][1]["active"] = [poke(IMPIDIMP, 0, 70, 90)]
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["dead_swing_decisions"] == 0
