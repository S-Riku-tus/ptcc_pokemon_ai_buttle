"""Golden states for the v15 attack-access invariant.

Each test is a board the v14 ladder logs produced (or the state v14 must keep
behaving on), written as the engine writes it: MAIN energy attachment is a
``type: 8`` option carrying ``area``/``index`` into the hand and
``inPlayArea``/``inPlayIndex`` onto the board, retreat is ``type: 12``, and
promotion after a retreat is a ``type: 3`` card select over our own Bench in
context 3.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from attack_access import AttackAccessGuard, RETREAT_COST  # noqa: E402

DARK = mf.DARK_ENERGY_ID
GRIM = mf.GRIMMSNARL_EX_ID
MORGREM = mf.MORGREM_ID
IMPIDIMP = mf.IMPIDIMP_ID
SNORUNT = mf.SNORUNT_ID
MUNKIDORI = mf.MUNKIDORI_ID
CANDY = mf.RARE_CANDY_ID
PAD = mf.POKE_PAD_ID
SHADOW = mf.SHADOW_BULLET_ID
FILCH = 934
CORKSCREW_MORGREM = 936
CRUSTLE = 345

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
    }


def attach(hand_index: int, *, bench_slot: int | None = None) -> dict:
    """A MAIN energy attachment from hand onto the Active or a Bench slot."""
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
    opponent_active: dict | None = None,
    opponent_bench: list[dict] | None = None,
    context: int = mf.MAIN_CONTEXT,
    energy_attached: bool = False,
    retreated: bool = False,
    asleep: bool = False,
    paralyzed: bool = False,
    max_count: int = 1,
    turn: int = 5,
) -> dict:
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "retreated": retreated,
            "energyAttached": energy_attached,
            "supporterPlayed": False,
            "players": [
                {
                    "active": [active],
                    "bench": list(bench),
                    "hand": list(hand),
                    "discard": [],
                    "prize": [{}] * 4,
                    "deckCount": 30,
                    "handCount": len(hand),
                    "asleep": asleep,
                    "paralyzed": paralyzed,
                },
                {
                    "active": [
                        opponent_active
                        if opponent_active is not None
                        else poke(1, hp=300, serial=90)
                    ],
                    "bench": list(
                        opponent_bench or [poke(2, hp=100, serial=91)]
                    ),
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


def trapped_snorunt(
    *,
    active_energy: int = 0,
    options: list[dict] | None = None,
    **kwargs,
) -> dict:
    """Episode 91548124's state: a finished attacker nobody can reach.

    Active Snorunt with no Energy, a Basic Darkness in hand, the manual
    attachment unused and a Shadow-Bullet-ready Grimmsnarl ex on the Bench.
    """
    return observation(
        active=poke(SNORUNT, energy=active_energy, hp=70, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[{"id": PAD}, {"id": DARK}],
        options=options
        if options is not None
        else [play(0), attach(1), attach(1, bench_slot=0), END],
        **kwargs,
    )


# ----- the route ------------------------------------------------------------
def test_retreat_costs_match_the_card_database() -> None:
    from cg.api import all_card_data

    table = {card.cardId: card for card in all_card_data()}
    for card_id, cost in RETREAT_COST.items():
        assert int(table[card_id].retreatCost) == cost
    # The premise of the one-attachment escape route.
    assert all(
        cost <= 1 for card_id, cost in RETREAT_COST.items() if card_id != GRIM
    )


def test_escape_attachment_is_forced_over_optional_setup() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt()
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["escape_attach_forced"] == 1
    assert guard.stats["trapped_turns"] == 1
    assert guard.stats["trapped_turns_worth"] == 1


def test_escape_attachment_beats_attaching_to_the_bench_attacker() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt()
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 2, [2]) == 1
    assert guard.stats["escape_attach_forced"] == 1


def test_the_route_step_itself_is_left_alone() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt()
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 1, [1]) == 1
    assert guard.stats["escape_attach_forced"] == 0
    assert guard.stats["route_compatible_kept"] == 1


def test_retreat_is_forced_once_paid_and_promotes_the_attacker() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt(
        active_energy=1, options=[play(0), RETREAT, END], energy_attached=True
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["retreat_forced"] == 1

    promote = observation(
        active=poke(SNORUNT, energy=1, hp=70, serial=1),
        bench=[
            poke(SNORUNT, hp=70, serial=3),
            poke(GRIM, energy=2, hp=340, serial=2),
        ],
        hand=[],
        context=mf.CTX_SWITCH,
        retreated=True,
        options=[
            {"type": 3, "area": mf.AREA_BENCH, "index": 0, "playerIndex": 0},
            {"type": 3, "area": mf.AREA_BENCH, "index": 1, "playerIndex": 0},
        ],
    )
    guard.note(promote)  # same turn, so the pending promotion survives
    assert guard.adjust(promote, promote["select"], 0, [0]) == 1
    assert guard.stats["promote_forced"] == 1


def test_active_grimmsnarl_one_energy_short_is_fuelled() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(GRIM, energy=1, hp=340, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[{"id": PAD}, {"id": DARK}],
        options=[play(0), attach(1), attach(1, bench_slot=0), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["enable_attach_forced"] == 1


def test_a_non_shadow_attack_loses_to_an_open_shadow_route() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(MORGREM, energy=2, hp=100, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[],
        options=[attack(CORKSCREW_MORGREM), RETREAT, END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["attack_overridden_for_route"] == 1
    assert guard.stats["retreat_forced"] == 1


# ----- routes that must not be invented -------------------------------------
def test_no_route_when_the_manual_attachment_is_already_spent() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt(energy_attached=True)
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["escape_attach_forced"] == 0
    assert guard.stats["no_route_available"] == 1


def test_no_route_when_we_already_retreated() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt(retreated=True)
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["escape_attach_forced"] == 0


def test_no_route_for_a_sleeping_or_paralysed_active() -> None:
    for condition in ("asleep", "paralyzed"):
        guard = AttackAccessGuard()
        obs = trapped_snorunt(**{condition: True})
        guard.note(obs)
        assert guard.adjust(obs, obs["select"], 0, [0]) == 0
        assert guard.stats["escape_attach_forced"] == 0


def test_no_route_without_a_darkness_in_hand() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(SNORUNT, hp=70, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[{"id": PAD}],
        options=[play(0), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["no_route_available"] == 1


def test_a_two_energy_escape_is_not_attempted() -> None:
    """A Grimmsnarl ex Active at 0 Energy cannot retreat on one card."""
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(GRIM, energy=0, hp=340, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[{"id": PAD}, {"id": DARK}],
        options=[play(0), attach(1), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0


def test_an_active_evolution_route_is_preferred_to_the_escape() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(MORGREM, hp=100, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[{"id": GRIM}, {"id": DARK}],
        options=[evolve(0), attach(1), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["route_compatible_kept"] == 1


def test_the_rare_candy_route_is_preferred_to_the_escape() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(IMPIDIMP, hp=70, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[{"id": CANDY}, {"id": GRIM}, {"id": DARK}],
        options=[play(0), attach(2), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["route_compatible_kept"] == 1


def test_multi_pick_selects_are_never_touched() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt(max_count=2)
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["main_decisions"] == 0


def test_promotion_never_fires_outside_our_own_route() -> None:
    guard = AttackAccessGuard()
    promote = observation(
        active=poke(SNORUNT, hp=70, serial=1),
        bench=[
            poke(SNORUNT, hp=70, serial=3),
            poke(GRIM, energy=2, hp=340, serial=2),
        ],
        hand=[],
        context=mf.CTX_TO_ACTIVE,
        options=[
            {"type": 3, "area": mf.AREA_BENCH, "index": 0, "playerIndex": 0},
            {"type": 3, "area": mf.AREA_BENCH, "index": 1, "playerIndex": 0},
        ],
    )
    guard.note(promote)
    assert guard.adjust(promote, promote["select"], 0, [0]) == 0
    assert guard.stats["promote_forced"] == 0


# ----- conversion -----------------------------------------------------------
def test_a_turn_never_ends_with_a_worthwhile_shadow_bullet_unspent() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(GRIM, energy=2, hp=340, serial=1),
        bench=[],
        hand=[{"id": PAD}],
        options=[play(0), attack(SHADOW), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 2, [2]) == 1
    assert guard.stats["end_replaced_by_shadow"] == 1
    assert guard.stats["played_shadow_bullets"] == 1


def test_setup_before_a_shadow_bullet_is_still_allowed() -> None:
    """Nothing but ATTACK and END closes a turn, so setup is not suppressed."""
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(GRIM, energy=2, hp=340, serial=1),
        bench=[],
        hand=[{"id": PAD}],
        options=[play(0), attack(SHADOW), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["end_replaced_by_shadow"] == 0


def test_the_bridge_attack_replaces_an_idle_end() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(IMPIDIMP, energy=1, hp=70, serial=1),
        bench=[poke(GRIM, hp=340, serial=2)],
        hand=[],
        options=[attack(FILCH), END],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 1, [0]) == 0
    assert guard.stats["end_replaced_by_bridge"] == 1
    assert guard.stats["played_other_attacks"] == 1


def test_the_bridge_attack_is_v8s_judgement_not_ours() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(IMPIDIMP, energy=1, hp=70, serial=1),
        bench=[poke(GRIM, hp=340, serial=2)],
        hand=[],
        options=[attack(FILCH), END],
    )
    guard.note(obs)
    # v8 ends too (e.g. a nearly empty deck makes Filch a loss): keep the END.
    assert guard.adjust(obs, obs["select"], 1, [1]) == 1
    assert guard.stats["end_replaced_by_bridge"] == 0
    assert guard.stats["played_ends"] == 1


# ----- the wall keeps v14's behaviour ---------------------------------------
def test_a_valueless_wall_does_not_open_the_route() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt(
        opponent_active=poke(CRUSTLE, hp=150, serial=90),
        opponent_bench=[poke(2, hp=100, serial=91)],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 0
    assert guard.stats["escape_attach_forced"] == 0
    assert guard.stats["trapped_turns"] == 1
    assert guard.stats["trapped_turns_worth"] == 0


def test_a_valueless_wall_keeps_the_wall_guards_end() -> None:
    guard = AttackAccessGuard()
    obs = observation(
        active=poke(GRIM, energy=2, hp=340, serial=1),
        bench=[],
        hand=[],
        options=[attack(SHADOW), END],
        opponent_active=poke(CRUSTLE, hp=150, serial=90),
        opponent_bench=[poke(2, hp=100, serial=91)],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 1, [1]) == 1
    assert guard.stats["end_replaced_by_shadow"] == 0
    assert guard.stats["ends_with_ready_attacker"] == 1


def test_a_wall_with_a_bench_prize_does_open_the_route() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt(
        opponent_active=poke(CRUSTLE, hp=150, serial=90),
        opponent_bench=[poke(2, hp=20, serial=91)],
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    assert guard.stats["escape_attach_forced"] == 1


# ----- bookkeeping ----------------------------------------------------------
def test_a_new_game_in_a_reused_process_drops_the_pending_route() -> None:
    guard = AttackAccessGuard()
    obs = trapped_snorunt(
        active_energy=1, options=[play(0), RETREAT, END], energy_attached=True
    )
    guard.note(obs)
    assert guard.adjust(obs, obs["select"], 0, [0]) == 1
    fresh = observation(
        active=poke(SNORUNT, hp=70, serial=1),
        bench=[poke(GRIM, energy=2, hp=340, serial=2)],
        hand=[],
        context=mf.CTX_SWITCH,
        turn=1,
        options=[
            {"type": 3, "area": mf.AREA_BENCH, "index": 0, "playerIndex": 0},
        ],
    )
    guard.note(fresh)
    assert guard.stats["new_game_detected"] == 1
    assert guard.adjust(fresh, fresh["select"], 0, [0]) == 0
    assert guard.stats["promote_forced"] == 0


def test_every_failure_returns_the_callers_index() -> None:
    guard = AttackAccessGuard()
    for broken in ({}, {"select": None}, {"current": {"players": []}}):
        select = {"context": 0, "minCount": 1, "maxCount": 1, "option": [END]}
        assert guard.adjust(broken, select, 0, [0]) == 0


# ----- artifacts ------------------------------------------------------------
def test_v19_keeps_v18s_deck_and_attack_access_against_v15() -> None:
    """v15's safety argument survives the intentional v19 model refresh.

    ``attack_access.py`` is on the identical list, so every state in this file
    still describes the shipped module and the v15 invariant is untouched.
    """
    metadata = json.loads(
        (AGENT_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    source = (AGENT_DIR / "main.py").read_text(encoding="utf-8")
    assert metadata["name"] == "grimmsnarl_ml_v19"
    assert metadata["deck_hash"] == "9714ab5c3996f6cc"
    assert metadata["deck_changed"] is False
    assert metadata["ranker"]["model_changed"] is True
    assert metadata["router"]["policy_switches"] == 0
    assert metadata["router"]["teacher_pin_switches"] == 0
    assert metadata["router"]["model_expert_switches"] == 0
    assert "AttackAccessGuard" in source
    assert "WallBreakGuard" in source
    assert "WallSafetyGuard" in source
    assert "Residual" in source
    assert not (AGENT_DIR / "ranker_model_v9.json").exists()

    v15 = AGENT_DIR.parent / "grimmsnarl_ml_v15"
    if not v15.is_dir():  # v15 may be archived away later
        return
    for name in metadata["files_identical_to_v15"]:
        ours = hashlib.sha256((AGENT_DIR / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((v15 / name).read_bytes()).hexdigest()
        assert ours == theirs, name
    changed = set(metadata["files_changed_from_v15"])
    assert changed == {
        "main.py", "ml_runtime.py", "wall_break.py", "policy_router.py",
        "mirror_prize.py", "ranker_model.json", "metadata.json",
    }
    assert not (v15 / "wall_break.py").exists()
    assert not (v15 / "mirror_prize.py").exists()
    for name in changed - {"wall_break.py", "mirror_prize.py"}:
        ours = hashlib.sha256((AGENT_DIR / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((v15 / name).read_bytes()).hexdigest()
        assert ours != theirs, name
    assert (
        hashlib.sha256(
            (AGENT_DIR / "ranker_model.json").read_bytes()
        ).hexdigest() == metadata["ranker"]["sha256"]
    )
