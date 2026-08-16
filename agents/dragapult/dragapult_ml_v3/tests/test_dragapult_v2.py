from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
AGENT = ROOT / "agents" / "dragapult" / "dragapult_ml_v3"
for path in (AGENT, ROOT / "agents" / "_base", ROOT / "vendor", ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from ml.core.replay_io import deck_hash  # noqa: E402
from cg.api import (  # noqa: E402
    AreaType,
    Card,
    OptionType,
    SelectContext,
    SelectType,
    to_observation_class,
)


def fresh(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def current(hand=None):
    return {
        "yourIndex": 0,
        "firstPlayer": 0,
        "turn": 3,
        "turnActionCount": 1,
        "energyAttached": False,
        "retreated": False,
        "stadiumPlayed": False,
        "supporterPlayed": False,
        "stadium": [],
        "players": [
            {
                "active": [{"id": 119, "hp": 70, "maxHp": 70, "energies": []}],
                "bench": [],
                "hand": list(hand or []),
                "handCount": len(hand or []),
                "deckCount": 40,
                "discard": [],
                "prize": [None] * 6,
            },
            {
                "active": [{"id": 119, "hp": 70, "maxHp": 70, "energies": []}],
                "bench": [],
                "hand": None,
                "handCount": 5,
                "deckCount": 40,
                "discard": [],
                "prize": [None] * 6,
            },
        ],
    }


def poke(card_id, *, serial, energies=None, hp=None):
    max_hp = {112: 110, 119: 70, 120: 90, 121: 320, 235: 30}.get(card_id, 100)
    return {
        "id": card_id,
        "serial": serial,
        "playerIndex": 0,
        "hp": max_hp if hp is None else hp,
        "maxHp": max_hp,
        "appearThisTurn": False,
        "energies": list(energies or []),
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def policy_observation(*, active, bench=None, hand=None, select=None):
    def player(*, active_cards, bench_cards, hand_cards, hand_count):
        return {
            "active": active_cards,
            "bench": bench_cards,
            "benchMax": 5,
            "deckCount": 40,
            "discard": [],
            "prize": [None] * 6,
            "handCount": hand_count,
            "hand": hand_cards,
            "poisoned": False,
            "burned": False,
            "asleep": False,
            "paralyzed": False,
            "confused": False,
        }

    raw = {
        "select": select,
        "logs": [],
        "current": {
            "turn": 3,
            "turnActionCount": 1,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                player(
                    active_cards=[active],
                    bench_cards=list(bench or []),
                    hand_cards=list(hand or []),
                    hand_count=len(hand or []),
                ),
                player(
                    active_cards=[poke(119, serial=90)],
                    bench_cards=[],
                    hand_cards=None,
                    hand_count=5,
                ),
            ],
        },
    }
    return to_observation_class(raw)


def main_select(options):
    return {
        "type": SelectType.MAIN,
        "context": SelectContext.MAIN,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": options,
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


def play_observation(*, optional=False):
    hand = [
        {"id": 119, "serial": 1, "playerIndex": 0},
        {"id": 235, "serial": 2, "playerIndex": 0},
    ]
    return {
        "step": 5,
        "remainingOverageTime": 590,
        "logs": [],
        "current": current(hand),
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 0 if optional else 1,
            "maxCount": 1,
            "option": [
                {"type": 7, "index": 0},
                {"type": 7, "index": 1},
            ],
        },
    }


def test_exact_deck_hash_and_counts():
    fallback = fresh("fallback_policy")
    assert len(fallback.MY_DECK) == 60
    assert deck_hash(fallback.MY_DECK) == "202ee2cec6cbe8b4"
    assert Counter(fallback.MY_DECK)[119] == 4
    assert Counter(fallback.MY_DECK)[121] == 3


def test_features_ignore_opponent_private_hand_and_reject_label_columns():
    features = fresh("ml_features")
    base = current([])
    first = features.state_features(base)
    base["players"][1]["hand"] = [
        {"id": 121}, {"id": 121}, {"id": 1080},
    ]
    second = features.state_features(base)
    assert first == second
    features.assert_no_leakage(list(first))
    try:
        features.assert_no_leakage(["turn", "final_reward"])
    except ValueError:
        pass
    else:
        raise AssertionError("final_reward must be rejected")


def test_runtime_falls_back_on_optional_and_unseen_candidate(tmp_path):
    runtime = fresh("ml_runtime")
    model = {
        "feature_names": ["option_type", "candidate_card_id", "teacher_team_id"],
        "trees": [{"v": 0.0}],
        "teacher_team_id": 16380946,
        "teacher_team_code": 0,
        "routed_contexts": [0],
        "runtime_support": {
            "select_context": [0],
            "option_type": [7],
            "candidate_card_id": [119],
            "candidate_attack_id": [-1],
        },
    }
    path = tmp_path / "ranker.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    ranker = runtime.Ranker(str(path))
    assert ranker.choose(play_observation(optional=True)) is None
    assert ranker.snapshot()["optional_fallback"] == 1
    assert ranker.choose(play_observation(optional=False)) is None
    assert ranker.snapshot()["ood_fallback"] == 1


def test_runtime_scores_supported_mandatory_decision(tmp_path):
    runtime = fresh("ml_runtime")
    model = {
        "feature_names": ["candidate_card_id", "teacher_team_id"],
        "trees": [{
            "f": 0, "t": 150.0, "d": "<=", "x": True,
            "l": {"v": 1.0}, "r": {"v": 0.0},
        }],
        "teacher_team_id": 16380946,
        "teacher_team_code": 0,
        "routed_contexts": [0],
        "runtime_support": {
            "select_context": [0],
            "option_type": [7],
            "candidate_card_id": [119, 235],
            "candidate_attack_id": [-1],
        },
    }
    path = tmp_path / "ranker.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    ranker = runtime.Ranker(str(path))
    assert ranker.choose(play_observation()) == 0
    ranker.commit(0)
    assert ranker.snapshot()["ranker_used"] == 1


def test_typed_energy_features_distinguish_fire_from_psychic():
    features = fresh("ml_features")
    base = current([
        {"id": 2, "serial": 2, "playerIndex": 0},
        {"id": 5, "serial": 3, "playerIndex": 0},
    ])
    option = {"type": 8, "index": 0, "inPlayArea": 4, "inPlayIndex": 0}
    select = {"context": 0, "option": [option], "minCount": 1, "maxCount": 1}

    base["players"][0]["active"][0]["energies"] = [2]
    fire_state = features.option_features(base, select, option)
    base["players"][0]["active"][0]["energies"] = [5]
    psychic_state = features.option_features(base, select, option)

    assert fire_state["candidate_target_fire"] == 1
    assert fire_state["candidate_target_psychic"] == 0
    assert psychic_state["candidate_target_fire"] == 0
    assert psychic_state["candidate_target_psychic"] == 1
    assert fire_state["candidate_attach_duplicate_color"] == 1
    assert psychic_state["candidate_attach_completes_colors"] == 1


def test_board_width_features_describe_a_real_backup_route():
    features = fresh("ml_features")
    base = current([{"id": 121, "serial": 3, "playerIndex": 0}])
    base["players"][0]["active"] = [poke(121, serial=10, energies=[2, 5])]
    base["players"][0]["bench"] = [poke(119, serial=11, energies=[2])]
    state = features.state_features(base)

    assert state["route_bodies"] == 2
    assert state["backup_route_bodies"] == 1
    assert state["bench_line_bodies"] == 1
    assert state["bench_best_route_eta"] == 3
    assert state["line_body_deficit_two"] == 0
    assert state["line_body_deficit_three"] == 1
    assert state["needs_drakloak_piece"] == 1
    assert state["needs_dragapult_piece"] == 0


def test_search_features_separate_board_development_from_energy_resources():
    features = fresh("ml_features")
    hand = [
        {"id": 1086, "serial": 1, "playerIndex": 0},
        {"id": 1198, "serial": 2, "playerIndex": 0},
    ]
    base = current(hand)
    select = {"context": 0, "minCount": 1, "maxCount": 1}
    poffin = features.option_features(base, select, {"type": 7, "index": 0})
    crispin = features.option_features(base, select, {"type": 7, "index": 1})

    assert poffin["candidate_is_board_search"] == 1
    assert poffin["candidate_board_search_urgent"] == 1
    assert poffin["candidate_resources_existing_route"] == 0
    assert crispin["candidate_is_board_search"] == 0
    assert crispin["candidate_resources_existing_route"] == 1


def test_fallback_attaches_missing_color_and_finishes_pult_before_munkidori():
    fallback = fresh("fallback_policy")
    hand = [
        {"id": 2, "serial": 1, "playerIndex": 0},
        {"id": 5, "serial": 2, "playerIndex": 0},
        {"id": 7, "serial": 3, "playerIndex": 0},
    ]
    active = poke(119, serial=10, energies=[2])
    munkidori = poke(112, serial=11)
    options = [
        {"type": OptionType.ATTACH, "area": AreaType.HAND, "index": 0,
         "inPlayArea": AreaType.ACTIVE, "inPlayIndex": 0},
        {"type": OptionType.ATTACH, "area": AreaType.HAND, "index": 1,
         "inPlayArea": AreaType.ACTIVE, "inPlayIndex": 0},
        {"type": OptionType.ATTACH, "area": AreaType.HAND, "index": 2,
         "inPlayArea": AreaType.BENCH, "inPlayIndex": 0},
    ]
    obs = policy_observation(
        active=active, bench=[munkidori], hand=hand, select=main_select(options)
    )
    policy = fallback.DragapultPolicy(obs)
    scores = [policy.score(option) for option in policy.select.option]

    assert scores[0] < 0  # duplicate Fire is forbidden
    assert scores[1] > scores[2]  # missing Psychic beats Munkidori's Dark
    assert policy.choose() == [1]


def test_route_search_prefers_immediate_drakloak_over_unusable_dragapult():
    fallback = fresh("fallback_policy")
    dragapult = {"id": 121, "serial": 2, "playerIndex": 0}
    obs = policy_observation(
        active=poke(119, serial=10),
        hand=[dragapult],
        select=main_select([{"type": OptionType.END}]),
    )
    policy = fallback.DragapultPolicy(obs)

    assert policy.needs_evolution_piece()
    assert policy.score_to_hand(Card(id=120, serial=3, playerIndex=0)) > policy.score_to_hand(
        obs.current.players[0].hand[0]
    )


def test_ability_options_resolve_from_area_index():
    """ABILITY carries area/index; reading inPlayArea/inPlayIndex resolves None.

    v1 read the wrong pair in both the feature module and the rule policy, so
    ``is_recon_directive`` and ``is_adrena_brain`` were constant zero over the
    whole corpus and every ability scored the same constant.
    """
    features = fresh("ml_features")
    base = current([])
    base["players"][0]["bench"] = [
        {"id": 120, "hp": 90, "maxHp": 90, "energies": []},
        {"id": 112, "hp": 80, "maxHp": 110, "energies": [7]},
    ]
    select = {"context": 0, "option": [], "minCount": 1, "maxCount": 1}

    drakloak = features.option_features(
        base, select, {"type": 10, "area": 5, "index": 0}
    )
    munkidori = features.option_features(
        base, select, {"type": 10, "area": 5, "index": 1}
    )
    assert drakloak["is_recon_directive"] == 1
    assert drakloak["is_adrena_brain"] == 0
    assert munkidori["is_adrena_brain"] == 1
    assert munkidori["candidate_ability_movable"] == 3


def test_spread_features_respect_the_remaining_counter_budget():
    """Six counters kill a 60 HP body; two do not."""
    features = fresh("ml_features")
    base = current([])
    base["players"][1]["bench"] = [{"id": 112, "hp": 60, "maxHp": 110, "energies": []}]
    option = {"type": 3, "area": 5, "index": 0, "playerIndex": 1}

    full = features.option_features(
        base, {"context": 14, "option": [option], "remainDamageCounter": 6}, option
    )
    nearly_spent = features.option_features(
        base, {"context": 14, "option": [option], "remainDamageCounter": 2}, option
    )
    assert full["candidate_dies_to_spread"] == 1
    assert nearly_spent["candidate_dies_to_spread"] == 0
    assert full["candidate_counters_to_ko"] == 6


def test_environment_timer_is_not_a_feature():
    """remainingOverageTime describes the pilot's machine, not the game."""
    features = fresh("ml_features")
    observation = play_observation()
    assert "remaining_time" not in features.observation_features(observation)


def test_guard_only_replaces_a_dominated_duplicate_attachment():
    main = fresh("main")
    current_state = {
        "yourIndex": 0,
        "players": [
            {
                "hand": [{"id": 2}, {"id": 5}],
                "active": [{"id": 121, "energies": [2]}],
                "bench": [{"id": 120, "energies": [2]}],
            },
            {},
        ],
    }
    options = [
        # duplicate Fire onto a Dragapult that already holds Fire
        {"type": 8, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
        # the Psychic that turns Phantom Dive on
        {"type": 8, "index": 1, "inPlayArea": 4, "inPlayIndex": 0},
    ]
    observation = {
        "current": current_state,
        "select": {"context": 0, "minCount": 1, "maxCount": 1, "option": options},
    }
    assert main._guarded_index(observation, 0) == 1
    # The completing attachment itself is never overridden.
    assert main._guarded_index(observation, 1) is None

    # With no completing alternative on offer the guard stays out of the way.
    only_duplicate = {
        "current": current_state,
        "select": {
            "context": 0, "minCount": 1, "maxCount": 1,
            "option": [options[0], {"type": 14}],
        },
    }
    assert main._guarded_index(only_duplicate, 0) is None


def test_guard_blocks_only_zero_energy_active_evolution_without_attachment():
    main = fresh("main")
    state = current([{"id": 121, "serial": 1, "playerIndex": 0}])
    state["players"][0]["active"] = [poke(120, serial=10)]
    evolve = {"type": 9, "index": 0, "inPlayArea": 4, "inPlayIndex": 0}
    observation = {
        "current": state,
        "select": {
            "context": 0, "minCount": 1, "maxCount": 1,
            "option": [evolve, {"type": 14}],
        },
    }
    assert main._is_dead_active_evolution(observation, 0)

    # One existing Energy enables Jet Headbutt, so this is a strategic choice
    # left to the learned policy rather than seized by the mechanical guard.
    state["players"][0]["active"][0]["energies"] = [2]
    assert not main._is_dead_active_evolution(observation, 0)

    # With zero attached, a currently legal manual attachment also makes the
    # evolved body an attacker this turn and must keep the evolution legal.
    state["players"][0]["active"][0]["energies"] = []
    state["players"][0]["hand"].append({"id": 5, "serial": 2, "playerIndex": 0})
    observation["select"]["option"].insert(
        1, {"type": 8, "index": 1, "inPlayArea": 4, "inPlayIndex": 0}
    )
    assert not main._is_dead_active_evolution(observation, 0)


def test_guard_never_blocks_bench_evolution():
    main = fresh("main")
    state = current([{"id": 121, "serial": 1, "playerIndex": 0}])
    state["players"][0]["active"] = [poke(112, serial=10)]
    state["players"][0]["bench"] = [poke(120, serial=11)]
    observation = {
        "current": state,
        "select": {
            "context": 0, "minCount": 1, "maxCount": 1,
            "option": [
                {"type": 9, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
                {"type": 14},
            ],
        },
    }
    assert not main._is_dead_active_evolution(observation, 0)


def test_guard_allows_active_evolution_that_survives_ready_phantom_dive():
    """Held-out episode 93140945: evolution converts a KO into survival."""
    main = fresh("main")
    state = current([{"id": 121, "serial": 1, "playerIndex": 0}])
    state["players"][0]["active"] = [poke(120, serial=10, hp=60)]
    state["players"][1]["active"] = [
        {**poke(121, serial=20, energies=[5, 2], hp=300), "playerIndex": 1}
    ]
    observation = {
        "current": state,
        "select": {
            "context": 0, "minCount": 1, "maxCount": 1,
            "option": [
                {"type": 9, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
                {"type": 14},
            ],
        },
    }
    assert main._survives_ready_phantom_dive(
        observation, state["players"][0]["active"][0]
    )
    assert not main._is_dead_active_evolution(observation, 0)

    # Without both Phantom Dive colours this is not the evidenced exception.
    state["players"][1]["active"][0]["energies"] = [5]
    assert main._is_dead_active_evolution(observation, 0)


def test_discard_table_is_teacher_derived_and_protects_the_route():
    fallback = fresh("fallback_policy")
    assert fallback.TEACHER_DISCARD_RATE, "generated discard table is empty"
    # Budew is the teachers' first pitch; Unfair Stamp is their last.
    assert fallback.TEACHER_DISCARD_RATE[235] > fallback.TEACHER_DISCARD_RATE[1080]

    obs = policy_observation(
        active=poke(120, serial=10, energies=[2]),
        hand=[{"id": 5, "serial": 1, "playerIndex": 0}],
        select={
            "type": SelectType.CARD,
            "context": SelectContext.DISCARD,
            "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": [{"type": OptionType.CARD, "area": AreaType.HAND,
                        "index": 0, "playerIndex": 0}],
            "deck": None, "contextCard": None, "effect": None,
        },
    )
    policy = fallback.DragapultPolicy(obs)
    # The only Psychic while the route still needs it must not be the payment.
    assert policy.score_discard(obs.current.players[0].hand[0]) < 0
