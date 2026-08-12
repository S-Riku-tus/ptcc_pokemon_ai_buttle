"""v20's bounded two-turn Prize tie-breaker."""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from horizon_prize import HorizonPrizePlanner  # noqa: E402


def pokemon(
    card_id: int,
    hp: int,
    *,
    serial: int,
    energy: int = 0,
) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": 320 if card_id == mf.GRIMMSNARL_EX_ID else 110,
        "serial": serial,
        "energies": [mf.DARK_ENERGY_ID] * energy,
        "energyCards": [
            {"id": mf.DARK_ENERGY_ID, "serial": 1000 + serial + slot}
            for slot in range(energy)
        ],
        "preEvolution": [],
    }


def board(bench: list[dict], *, prizes: int = 3) -> dict:
    return {
        "turn": 7,
        "yourIndex": 0,
        "players": [
            {
                "active": [pokemon(
                    mf.GRIMMSNARL_EX_ID, 320, serial=1, energy=2
                )],
                "bench": [pokemon(
                    mf.MUNKIDORI_ID, 80, serial=2, energy=1
                )],
                "hand": [],
                "discard": [],
                "prize": [{}] * prizes,
            },
            {
                "active": [pokemon(
                    mf.GRIMMSNARL_EX_ID, 320, serial=90, energy=2
                )],
                "bench": bench,
                "hand": [],
                "discard": [],
                "prize": [{}] * prizes,
            },
        ],
        "stadium": [],
    }


def prime(planner: HorizonPrizePlanner) -> None:
    planner.set_mirror(True)
    observation = {
        "current": board([]),
        "select": {
            "context": mf.MAIN_CONTEXT,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 13, "attackId": mf.SHADOW_BULLET_ID},
                {"type": 14},
            ],
        },
    }
    planner.note(observation)
    assert planner.adjust(
        observation, observation["select"], 0, {0: 1.0, 1: 0.0}
    ) == 0


def target_observation(bench: list[dict], *, prizes: int = 3) -> dict:
    return {
        "current": board(bench, prizes=prizes),
        "select": {
            "context": mf.CTX_DAMAGE,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "effect": {"id": mf.GRIMMSNARL_EX_ID},
            "option": [
                {
                    "type": 3,
                    "playerIndex": 1,
                    "area": mf.AREA_BENCH,
                    "index": slot,
                }
                for slot in range(len(bench))
            ],
        },
    }


def boss_observation(bench: list[dict], *, prizes: int = 3) -> dict:
    return {
        "current": board(bench, prizes=prizes),
        "select": {
            "context": mf.CTX_SWITCH,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {
                    "type": 3,
                    "playerIndex": 1,
                    "area": mf.AREA_BENCH,
                    "index": slot,
                }
                for slot in range(len(bench))
            ],
        },
    }


def test_redirects_only_a_near_tie_to_a_larger_two_turn_ceiling() -> None:
    planner = HorizonPrizePlanner()
    prime(planner)
    observation = target_observation([
        pokemon(mf.MUNKIDORI_ID, 100, serial=91),
        pokemon(mf.GRIMMSNARL_EX_ID, 60, serial=92),
    ])
    # 30 now + 30 next Shadow finishes the two-Prize target.  The first
    # target remains outside that conservative ceiling.
    moved = planner.adjust(
        observation, observation["select"], 0, {0: 1.0, 1: 0.95}
    )
    assert moved == 1
    assert planner.stats["overrides"] == 1


def test_large_ranker_margin_is_never_overridden() -> None:
    planner = HorizonPrizePlanner()
    prime(planner)
    observation = target_observation([
        pokemon(mf.MUNKIDORI_ID, 100, serial=91),
        pokemon(mf.GRIMMSNARL_EX_ID, 60, serial=92),
    ])
    assert planner.adjust(
        observation, observation["select"], 0, {0: 1.0, 1: 0.5}
    ) == 0
    assert planner.stats["rejected_score_gap"] == 1


def test_stands_down_before_shadow_outside_mirror_and_before_endgame() -> None:
    observation = target_observation([
        pokemon(mf.MUNKIDORI_ID, 100, serial=91),
        pokemon(mf.GRIMMSNARL_EX_ID, 60, serial=92),
    ])
    planner = HorizonPrizePlanner()
    planner.set_mirror(True)
    assert planner.adjust(
        observation, observation["select"], 0, {0: 1.0, 1: 0.95}
    ) == 0

    prime(planner)
    planner.set_mirror(False)
    assert planner.adjust(
        observation, observation["select"], 0, {0: 1.0, 1: 0.95}
    ) == 0

    late = HorizonPrizePlanner()
    prime(late)
    early_observation = target_observation([
        pokemon(mf.MUNKIDORI_ID, 100, serial=91),
        pokemon(mf.GRIMMSNARL_EX_ID, 60, serial=92),
    ], prizes=4)
    assert late.adjust(
        early_observation,
        early_observation["select"],
        0,
        {0: 1.0, 1: 0.95},
    ) == 0


def test_equal_prize_ceiling_does_not_turn_damage_into_a_preference() -> None:
    planner = HorizonPrizePlanner()
    prime(planner)
    observation = target_observation([
        pokemon(mf.MUNKIDORI_ID, 60, serial=91),
        pokemon(mf.MUNKIDORI_ID, 60, serial=92),
    ])
    assert planner.adjust(
        observation, observation["select"], 0, {0: 1.0, 1: 0.95}
    ) == 0


def test_boss_target_uses_the_same_strict_two_turn_prize_gate() -> None:
    planner = HorizonPrizePlanner()
    prime(planner)
    observation = boss_observation([
        pokemon(mf.MUNKIDORI_ID, 200, serial=91),
        pokemon(mf.GRIMMSNARL_EX_ID, 320, serial=92),
    ])
    # Both bodies survive the first 180.  A second Shadow reaches both, so
    # the near-tied two-Prize target has the strictly larger ceiling.
    assert planner.adjust(
        observation, observation["select"], 0, {0: 1.0, 1: 0.95}
    ) == 1
    assert planner.stats["boss_target_overrides"] == 1
