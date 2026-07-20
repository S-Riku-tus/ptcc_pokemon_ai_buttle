from __future__ import annotations

from ml_features import option_features, state_features


def card(card_id, **extra):
    return {"id": card_id, **extra}


def observation(*, hand, active, bench=(), deck_count=30, opp_hp=100):
    return {
        "yourIndex": 0,
        "turn": 5,
        "energyAttached": False,
        "players": [
            {
                "active": [active],
                "bench": list(bench),
                "benchMax": 5,
                "hand": list(hand),
                "handCount": len(hand),
                "discard": [],
                "deckCount": deck_count,
                "prize": [None] * 6,
            },
            {
                "active": [card(9000, hp=opp_hp, maxHp=opp_hp, energies=[])],
                "bench": [],
                "handCount": 6,
                "discard": [],
                "deckCount": 30,
                "prize": [None] * 6,
            },
        ],
    }


def test_v11_ml_features_model_rare_candy_draw_and_immediate_ko():
    current = observation(
        hand=[card(1079), card(743), card(742), card(305), card(741)],
        active=card(741, hp=50, maxHp=50, energies=[5], energyCards=[card(5)]),
        opp_hp=100,
    )
    select = {"type": 0, "context": 0}
    option = {"type": 7, "index": 0}
    features = option_features(current, select, option)
    assert features["rare_candy_route_available"] == 1
    assert features["candidate_evolution_draw_count"] == 3
    assert features["candidate_net_hand_delta"] == 1
    assert features["rare_candy_projected_damage"] == 120
    assert features["rare_candy_immediate_ko_estimate"] == 1


def test_v11_ml_features_expose_bench_kadabra_candy_out():
    current = observation(
        hand=[card(742), card(743), card(305)],
        active=card(741, hp=50, maxHp=50, energies=[]),
        bench=[card(741, hp=50, maxHp=50, energies=[])],
    )
    select = {"type": 0, "context": 0}
    option = {
        "type": 9,
        "index": 0,
        "inPlayArea": 5,
        "inPlayIndex": 0,
    }
    features = option_features(current, select, option)
    assert features["candidate_evolution_draw_count"] == 2
    assert features["candidate_net_hand_delta"] == 1
    assert features["kadabra_draws_toward_candy_for_active_abra"] == 1


def test_v11_ml_features_expose_low_deck_energy_hit_probability():
    current = observation(
        hand=[card(743), card(742), card(1079)],
        active=card(743, hp=140, maxHp=140, energies=[], energyCards=[]),
        deck_count=6,
    )
    features = state_features(current)
    assert features["active_alakazam_unpowered"] == 1
    assert features["emergency_energy_draw_state"] == 1
    assert features["psychic_hit_probability_draw3"] > 0.40


def test_v11_ml_features_mark_dunsparce_setup_choice_over_abra():
    current = observation(
        hand=[card(741), card(305), card(743)],
        active=card(741, hp=50, maxHp=50, energies=[]),
    )
    select = {"type": 1, "context": 1}
    option = {"type": 3, "area": 2, "index": 1}
    features = option_features(current, select, option)
    assert features["candidate_card_id"] == 305
    assert features["setup_dunsparce_over_abra"] == 1
