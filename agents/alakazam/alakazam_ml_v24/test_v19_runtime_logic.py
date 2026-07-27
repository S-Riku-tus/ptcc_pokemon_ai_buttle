from __future__ import annotations

from collections import Counter

import test_v11_runtime_logic as h


def _stadium_ready(obj):
    obj.state.stadiumPlayed = False
    obj.stadium_id = 0
    return obj


def test_v19_nighttime_mine_requires_a_real_stadium_role():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    quiet_opponent = h.Pokemon(9000)
    obj = _stadium_ready(
        h.bare_policy(policy, active=active, opp_active=quiet_opponent)
    )

    assert obj._score_play_trainer(h.Card(policy.C.NIGHTTIME_MINE)) < 0


def test_v19_nighttime_mine_stops_a_payable_tera_attack():
    policy = h.load_policy()
    tera_data = policy.card_table[678]
    tera_data.tera = True
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    opponent = h.Pokemon(678, energies=[h.EnergyType.COLORLESS], playerIndex=1)
    obj = _stadium_ready(
        h.bare_policy(policy, hand_count=12, active=active, opp_active=opponent)
    )

    assert obj._nighttime_mine_tax_stops_active()
    assert obj._score_play_trainer(h.Card(policy.C.NIGHTTIME_MINE)) == 14500


def test_v19_nighttime_mine_never_spends_the_current_ko_card():
    policy = h.load_policy()
    tera_data = policy.card_table[678]
    tera_data.tera = True
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    opponent = h.Pokemon(
        678,
        hp=200,
        maxHp=340,
        energies=[h.EnergyType.COLORLESS],
        playerIndex=1,
    )
    obj = _stadium_ready(
        h.bare_policy(policy, hand_count=10, active=active, opp_active=opponent)
    )

    assert obj._score_play_trainer(h.Card(policy.C.NIGHTTIME_MINE)) < 0


def test_v19_lana_values_actual_three_card_recovery():
    policy = h.load_policy()
    obj = h.bare_policy(policy)
    obj.discard = Counter({
        policy.C.ABRA: 1,
        policy.C.KADABRA: 1,
        policy.C.PSYCHIC_ENERGY: 1,
    })

    assert obj._lana_recoverable_count() == 3
    assert obj._score_play_trainer(h.Card(policy.C.LANA_AID)) == 13200


def test_v19_lana_excludes_rule_box_and_special_energy_cards():
    policy = h.load_policy()
    obj = h.bare_policy(policy)
    obj.discard = Counter({
        policy.C.FEZANDIPITI_EX: 1,
        policy.C.TELEPATH_ENERGY: 2,
        policy.C.ENRICHING_ENERGY: 1,
    })

    assert obj._lana_recoverable_count() == 0
    assert obj._score_play_trainer(h.Card(policy.C.LANA_AID)) < 0


def test_v19_lana_scales_with_available_targets():
    policy = h.load_policy()
    obj = h.bare_policy(policy)
    obj.discard = Counter({policy.C.ABRA: 1})
    assert obj._score_play_trainer(h.Card(policy.C.LANA_AID)) == 4500

    obj.discard[policy.C.KADABRA] = 1
    assert obj._score_play_trainer(h.Card(policy.C.LANA_AID)) == 10500
