"""v18's post-Shadow maximum-immediate-Prize invariant."""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
import policy_router  # noqa: E402
from mirror_prize import MirrorPrizeGuard  # noqa: E402


def pokemon(card_id: int, hp: int, *, serial: int) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": 320 if card_id == mf.GRIMMSNARL_EX_ID else 110,
        "serial": serial,
        "energies": [],
    }


def current(
    bench: list[dict], *, turn: int = 7, stadium: int | None = None
) -> dict:
    value = {
        "turn": turn,
        "yourIndex": 0,
        "players": [
            {
                "active": [pokemon(mf.GRIMMSNARL_EX_ID, 320, serial=1)],
                "bench": [pokemon(mf.MUNKIDORI_ID, 110, serial=2)],
                "hand": [],
                "discard": [],
                "prize": [{}] * 4,
            },
            {
                "active": [pokemon(mf.GRIMMSNARL_EX_ID, 320, serial=90)],
                "bench": bench,
                "hand": [],
                "discard": [],
                "prize": [{}] * 4,
            },
        ],
        "stadium": [],
    }
    if stadium is not None:
        value["stadium"] = [{"id": stadium}]
    return value


def target_option(index: int, *, area: int = mf.AREA_BENCH) -> dict:
    return {"type": 3, "playerIndex": 1, "area": area, "index": index}


def shadow_observation(*, turn: int = 6) -> dict:
    return {
        "current": current([], turn=turn),
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


def target_observation(
    bench: list[dict],
    *,
    context: int = mf.CTX_DAMAGE_COUNTER,
    turn: int = 7,
    stadium: int | None = None,
) -> dict:
    effect = (
        mf.MUNKIDORI_ID
        if context == mf.CTX_DAMAGE_COUNTER
        else mf.GRIMMSNARL_EX_ID
    )
    return {
        "current": current(bench, turn=turn, stadium=stadium),
        "select": {
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 3,
            "effect": {"id": effect},
            "option": [target_option(index) for index in range(len(bench))],
        },
    }


def primed_guard() -> MirrorPrizeGuard:
    guard = MirrorPrizeGuard()
    guard.set_mirror(True)
    observation = shadow_observation()
    guard.note(observation)
    assert guard.adjust(observation, observation["select"], 0) == 0
    return guard


def test_adrena_redirects_one_prize_ko_to_two_prize_ko() -> None:
    guard = primed_guard()
    observation = target_observation([
        pokemon(mf.GRIMMSNARL_EX_ID, 20, serial=91),
        pokemon(mf.MUNKIDORI_ID, 20, serial=92),
    ])
    assert guard.adjust(observation, observation["select"], 1) == 0
    assert guard.stats["overrides"] == 1
    assert guard.stats["adrena_overrides"] == 1


def test_shadow_bench_damage_uses_the_same_prize_invariant() -> None:
    guard = primed_guard()
    observation = target_observation(
        [
            pokemon(mf.FROSLASS_ID, 20, serial=91),
            pokemon(mf.GRIMMSNARL_EX_ID, 30, serial=92),
        ],
        context=mf.CTX_DAMAGE,
    )
    assert guard.adjust(observation, observation["select"], 0) == 1
    assert guard.stats["shadow_bench_overrides"] == 1


def test_already_maximum_prize_target_is_untouched() -> None:
    guard = primed_guard()
    observation = target_observation([
        pokemon(mf.GRIMMSNARL_EX_ID, 20, serial=91),
        pokemon(mf.MUNKIDORI_ID, 20, serial=92),
    ])
    assert guard.adjust(observation, observation["select"], 0) == 0
    assert guard.stats["already_max_prizes"] == 1
    assert guard.stats["overrides"] == 0


def test_nonlethal_damaged_grim_is_not_a_preference() -> None:
    guard = primed_guard()
    observation = target_observation([
        pokemon(mf.GRIMMSNARL_EX_ID, 40, serial=91),
        pokemon(mf.MUNKIDORI_ID, 20, serial=92),
    ])
    # The one-Prize Munkidori is the only immediate KO.  The guard must not
    # turn the observed correlation into a broad Grimmsnarl preference.
    assert guard.adjust(observation, observation["select"], 1) == 1


def test_stands_down_before_shadow_and_outside_the_mirror() -> None:
    observation = target_observation([
        pokemon(mf.GRIMMSNARL_EX_ID, 20, serial=91),
        pokemon(mf.MUNKIDORI_ID, 20, serial=92),
    ])
    before = MirrorPrizeGuard()
    before.set_mirror(True)
    assert before.adjust(observation, observation["select"], 1) == 1

    outside = primed_guard()
    outside.set_mirror(False)
    assert outside.adjust(observation, observation["select"], 1) == 1


def test_battle_cage_prevents_a_false_damage_counter_ko() -> None:
    guard = primed_guard()
    observation = target_observation(
        [
            pokemon(mf.GRIMMSNARL_EX_ID, 20, serial=91),
            pokemon(mf.MUNKIDORI_ID, 20, serial=92),
        ],
        stadium=mf.BATTLE_CAGE_ID,
    )
    assert guard.adjust(observation, observation["select"], 1) == 1
    assert guard.stats["immediate_ko_prompts"] == 0


def test_turn_decrease_clears_the_first_shadow_gate() -> None:
    guard = primed_guard()
    observation = target_observation(
        [
            pokemon(mf.GRIMMSNARL_EX_ID, 20, serial=91),
            pokemon(mf.MUNKIDORI_ID, 20, serial=92),
        ],
        turn=1,
    )
    guard.note(observation)
    assert guard.adjust(observation, observation["select"], 1) == 1
    assert guard.stats["new_game_detected"] == 1


def public_board(card_id: int, *, turn: int) -> dict:
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {
                    "active": [pokemon(card_id, 100, serial=99)],
                    "bench": [],
                    "discard": [],
                },
            ],
            "stadium": [],
        }
    }


def test_router_does_not_leak_a_mirror_into_the_next_episode() -> None:
    router = policy_router.PolicyRouter()
    assert router.choose(
        public_board(mf.GRIMMSNARL_EX_ID, turn=12)
    ) == policy_router.MIRROR
    assert router.choose(public_board(999, turn=1)) == policy_router.DEFAULT
    assert router.snapshot()["new_game_detected"] == 1


def test_router_remains_sticky_inside_one_episode() -> None:
    router = policy_router.PolicyRouter()
    assert router.choose(
        public_board(mf.GRIMMSNARL_EX_ID, turn=4)
    ) == policy_router.MIRROR
    assert router.choose(public_board(999, turn=5)) == policy_router.MIRROR
    assert router.snapshot()["new_game_detected"] == 0
