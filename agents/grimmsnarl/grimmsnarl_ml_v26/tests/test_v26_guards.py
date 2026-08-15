from __future__ import annotations

import sys
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

import ml_features as mf  # noqa: E402
from deck_clock import DeckClockGuard  # noqa: E402
from wall_trajectory import WallTrajectoryGuard  # noqa: E402


DARK = mf.DARK_ENERGY_ID


def poke(card_id: int, energy: int = 0, serial: int = 1) -> dict:
    return {
        "id": card_id,
        "hp": 100,
        "maxHp": 100,
        "serial": serial,
        "energies": [DARK] * energy,
        "energyCards": [{"id": DARK}] * energy,
    }


def attach(hand: int, bench: int) -> dict:
    return {
        "type": 8,
        "area": mf.AREA_HAND,
        "index": hand,
        "inPlayArea": mf.AREA_BENCH,
        "inPlayIndex": bench,
    }


def play(hand: int) -> dict:
    return {"type": 7, "area": mf.AREA_HAND, "index": hand}


def evolve(hand: int, bench: int) -> dict:
    return {
        "type": 9,
        "area": mf.AREA_HAND,
        "index": hand,
        "inPlayArea": mf.AREA_BENCH,
        "inPlayIndex": bench,
    }


def observation(*, active: dict, bench: list[dict], hand: list[dict],
                options: list[dict], context: int = mf.MAIN_CONTEXT,
                effect: dict | None = None, turn: int = 12,
                deck: int = 20, own_prizes: int = 3,
                opponent_prizes: int = 5) -> dict:
    select = {
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "option": options,
    }
    if effect is not None:
        select["effect"] = effect
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "players": [
                {
                    "active": [active],
                    "bench": bench,
                    "hand": hand,
                    "handCount": len(hand),
                    "discard": [],
                    "prize": [{}] * own_prizes,
                    "deckCount": deck,
                },
                {
                    "active": [poke(345, serial=90)],
                    "bench": [],
                    "discard": [],
                    "prize": [{}] * opponent_prizes,
                    "deckCount": 20,
                },
            ],
            "stadium": [],
        },
        "select": select,
    }


def armed(guard: WallTrajectoryGuard, obs: dict) -> None:
    guard.note(obs, wall_known=True)


def test_punk_up_concentrates_spare_energy_on_one_morgrem() -> None:
    guard = WallTrajectoryGuard()
    obs = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[poke(mf.MORGREM_ID, 1, 2), poke(mf.MUNKIDORI_ID, 1, 3)],
        hand=[],
        context=mf.CTX_ATTACH_FROM,
        effect={"id": mf.GRIMMSNARL_EX_ID, "serial": 1},
        options=[
            {"type": 3, "area": mf.AREA_BENCH, "index": 1, "playerIndex": 0},
            {"type": 3, "area": mf.AREA_BENCH, "index": 0, "playerIndex": 0},
        ],
    )
    armed(guard, obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["punk_concentrated"] == 1


def test_manual_attachment_completes_morgrem_without_starving_grimmsnarl() -> None:
    guard = WallTrajectoryGuard()
    obs = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[poke(mf.MORGREM_ID, 1, 2), poke(mf.MUNKIDORI_ID, 1, 3)],
        hand=[{"id": DARK}],
        options=[attach(0, 1), attach(0, 0), {"type": 14}],
    )
    armed(guard, obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["manual_attachment_completed"] == 1


def test_manual_attachment_never_starves_adrena_brain() -> None:
    guard = WallTrajectoryGuard()
    obs = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[poke(mf.MORGREM_ID, 1, 2), poke(mf.MUNKIDORI_ID, 0, 3)],
        hand=[{"id": DARK}],
        options=[attach(0, 1), attach(0, 0), {"type": 14}],
    )
    armed(guard, obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["munkidori_activation_preserved"] == 1


def test_last_breaker_is_preserved_before_wall_reaches_active() -> None:
    guard = WallTrajectoryGuard()
    obs = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[poke(mf.MORGREM_ID, 1, 2)],
        hand=[{"id": mf.GRIMMSNARL_EX_ID}],
        options=[evolve(0, 0), {"type": 14}],
    )
    armed(guard, obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["last_breaker_preserved"] == 1


def test_last_bench_slot_prefers_missing_froslass_route_to_third_munkidori() -> None:
    guard = WallTrajectoryGuard()
    obs = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[
            poke(mf.MUNKIDORI_ID, 1, 2),
            poke(mf.MUNKIDORI_ID, 1, 3),
            poke(mf.IMPIDIMP_ID, 0, 4),
            poke(mf.GRIMMSNARL_EX_ID, 2, 5),
        ],
        hand=[{"id": mf.MUNKIDORI_ID}, {"id": mf.SNORUNT_ID}],
        options=[play(0), play(1), {"type": 14}],
    )
    armed(guard, obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["last_bench_route_reserved"] == 1


def test_non_wall_match_is_byte_for_byte_noop() -> None:
    guard = WallTrajectoryGuard()
    obs = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[poke(mf.MORGREM_ID, 1, 2), poke(mf.MUNKIDORI_ID, 0, 3)],
        hand=[{"id": DARK}],
        options=[attach(0, 1), attach(0, 0), {"type": 14}],
    )
    guard.note(obs, wall_known=False)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["skip_non_wall"] == 1


def test_deck_clock_vetoes_only_a_lethal_optional_refill() -> None:
    guard = DeckClockGuard()
    obs = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[],
        hand=[{"id": mf.LILLIE_ID}, {"id": DARK}, {"id": mf.BOSS_ID}],
        deck=3,
        options=[play(0), {"type": 13, "attackId": mf.SHADOW_BULLET_ID}, {"type": 14}],
    )
    assert guard.adjust(obs, obs["select"], 0, {0: 3.0, 1: 2.0, 2: 1.0}) == 1
    assert guard.stats["lillie_deckout_vetoed"] == 1

    safe = observation(
        active=poke(mf.GRIMMSNARL_EX_ID, 2, 1),
        bench=[],
        hand=[{"id": mf.LILLIE_ID}, {"id": DARK}, {"id": mf.BOSS_ID}],
        deck=12,
        options=[play(0), {"type": 13, "attackId": mf.SHADOW_BULLET_ID}],
    )
    assert guard.adjust(safe, safe["select"], 0, {0: 3.0, 1: 2.0}) == 0
