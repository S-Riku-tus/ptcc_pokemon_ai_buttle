"""v17 completes v16's last-wall-breaker PRESERVE invariant."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from wall_break import WallBreakGuard  # noqa: E402

DARK = mf.DARK_ENERGY_ID
GRIM = mf.GRIMMSNARL_EX_ID
IMPIDIMP = mf.IMPIDIMP_ID
OGERPON = 117
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


def evolve(hand_index: int) -> dict:
    return {
        "type": 9,
        "area": mf.AREA_HAND,
        "index": hand_index,
        "inPlayArea": mf.AREA_BENCH,
        "inPlayIndex": 0,
    }


def board(options: list[dict], hand: list[dict] | None = None) -> dict:
    hand = list(hand or [{"id": GRIM}])
    return {
        "current": {
            "turn": 9,
            "yourIndex": 0,
            "players": [
                {
                    "active": [poke(GRIM, 2, 320, 1)],
                    "bench": [poke(IMPIDIMP, 0, 70, 2)],
                    "hand": hand,
                    "discard": [],
                    "prize": [{}] * 4,
                    "deckCount": 20,
                    "asleep": False,
                    "paralyzed": False,
                },
                {
                    "active": [poke(OGERPON, 0, 210, 90)],
                    "bench": [],
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


def punk_board(*, trigger_energy: int = 2, wall: int = OGERPON) -> dict:
    obs = board([])
    obs["current"]["players"][0]["active"] = [
        poke(GRIM, trigger_energy, 320, 1)
    ]
    obs["current"]["players"][0]["bench"] = [
        poke(mf.MORGREM_ID, 1, 100, 2),
        poke(IMPIDIMP, 0, 70, 3),
    ]
    obs["current"]["players"][1]["active"] = [
        poke(wall, 0, 210, 90)
    ]
    obs["select"] = {
        "context": mf.CTX_ATTACH_FROM,
        "minCount": 1,
        "maxCount": 1,
        "effect": {"id": GRIM, "serial": 1, "playerIndex": 0},
        "option": [
            {"type": 3, "area": mf.AREA_ACTIVE, "index": 0,
             "playerIndex": 0},
            {"type": 3, "area": mf.AREA_BENCH, "index": 0,
             "playerIndex": 0},
            {"type": 3, "area": mf.AREA_BENCH, "index": 1,
             "playerIndex": 0},
        ],
    }
    return obs


def test_too_slow_impidimp_is_preserved_by_the_free_shadow() -> None:
    guard = WallBreakGuard()
    obs = board([
        evolve(0),
        {"type": 13, "attackId": SHADOW},
        {"type": 14},
    ])
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["last_breaker_evolve_refused"] == 1
    assert guard.stats["last_breaker_preserve_shadow"] == 1
    assert guard.stats["last_breaker_evolve_kept"] == 0


def test_end_is_the_safe_fallback_when_shadow_is_not_offered() -> None:
    guard = WallBreakGuard()
    obs = board([evolve(0), {"type": 14}])
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["last_breaker_preserve_end"] == 1


def test_an_interchangeable_second_evolve_does_not_bypass_preserve() -> None:
    guard = WallBreakGuard()
    obs = board(
        [
            evolve(0),
            evolve(1),
            {"type": 13, "attackId": SHADOW},
            {"type": 14},
        ],
        hand=[{"id": GRIM}, {"id": GRIM}],
    )
    guard.note(obs)

    # The fallback points at another copy of the same evolution.  It still
    # consumes the same last Impidimp, so v17 must reach the closing fallback.
    assert guard.adjust(obs, obs["select"], 0, [1]) == 2
    assert guard.stats["last_breaker_preserve_shadow"] == 1


def test_no_legal_alternative_keeps_the_original_choice() -> None:
    guard = WallBreakGuard()
    obs = board([evolve(0)])
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["last_breaker_evolve_kept"] == 1


def test_punk_up_finishes_the_route_viable_morgrem() -> None:
    guard = WallBreakGuard()
    obs = punk_board()
    guard.note(obs)

    # v16 selected the Impidimp on the one missed stored decision.  The
    # Morgrem already has one Energy, so this Energy removes the final manual
    # attachment from the four-swing wall route.
    assert guard.adjust(obs, obs["select"], 2, [2]) == 1
    assert guard.stats["punk_breaker_forced"] == 1


def test_punk_up_keeps_an_already_correct_morgrem_target() -> None:
    guard = WallBreakGuard()
    obs = punk_board()
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 1, [1]) == 1
    assert guard.stats["punk_breaker_kept"] == 1


def test_punk_up_never_steals_the_triggering_grimmsnarls_second_energy(
) -> None:
    guard = WallBreakGuard()
    obs = punk_board(trigger_energy=1)
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["punk_breaker_considered"] == 0


def test_punk_up_target_is_unchanged_when_shadow_can_damage_the_active() -> None:
    guard = WallBreakGuard()
    obs = punk_board(wall=999)
    guard.note(obs)

    assert guard.adjust(obs, obs["select"], 2, [2]) == 2
    assert guard.stats["punk_breaker_considered"] == 0


def test_v20_changes_only_the_files_declared_against_v17() -> None:
    metadata = json.loads(
        (AGENT_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    v17 = AGENT_DIR.parent / "grimmsnarl_ml_v17"
    assert v17.is_dir()

    for name in metadata["files_identical_to_v17"]:
        ours = hashlib.sha256((AGENT_DIR / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((v17 / name).read_bytes()).hexdigest()
        assert ours == theirs, name

    for name in metadata["files_changed_from_v17"]:
        if not (v17 / name).exists():
            assert name in {
                "mirror_prize.py", "ranker_model_v9.json",
                "horizon_prize.py",
            }
            continue
        ours = hashlib.sha256((AGENT_DIR / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((v17 / name).read_bytes()).hexdigest()
        assert ours != theirs, name
