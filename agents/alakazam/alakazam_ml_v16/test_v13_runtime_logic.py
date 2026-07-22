from __future__ import annotations

import types
from collections import Counter
from types import SimpleNamespace

import test_v11_runtime_logic as h


def _real_effect_method(policy, obj):
    obj._effect_prevented = types.MethodType(policy.AlakazamPolicy._effect_prevented, obj)
    return obj


def test_v13_deck_replaces_max_rod_with_enriching_energy_only():
    policy = h.load_policy()
    counts = Counter(policy.my_deck)
    assert len(policy.my_deck) == 60
    assert counts[policy.C.ENRICHING_ENERGY] == 1
    assert counts[1110] == 0
    assert counts[policy.C.PSYCHIC_ENERGY] == 2
    assert counts[policy.C.TELEPATH_ENERGY] == 4


def test_v13_enriching_attach_has_net_plus_three_hand_delta():
    policy = h.load_policy()
    obj = h.bare_policy(policy)
    option = SimpleNamespace(index=0)
    old_get_card = policy.get_card
    policy.get_card = lambda *args, **kwargs: h.Card(policy.C.ENRICHING_ENERGY)
    try:
        assert obj._hand_delta(h.OptionType.ATTACH, option) == 3
    finally:
        policy.get_card = old_get_card


def test_v15_enriching_does_not_overprioritize_immediate_dudunsparce_recycle():
    policy = h.load_policy()
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70)
    obj = h.bare_policy(
        policy,
        hand_count=7,
        active=h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC]),
        bench=[dunsparce],
        opp_active=h.Pokemon(9000, hp=260, maxHp=260, playerIndex=1),
    )
    obj.hand[policy.C.DUDUNSPARCE] = 1
    obj._deck_spend_ok = lambda *args, **kwargs: True
    assert obj._enrich_cycle_ready(dunsparce)
    assert obj._enriching_attach_score(dunsparce) == 6650


def test_v13_articuno_scope_protects_basic_rocket_but_not_evolution():
    policy = h.load_policy()
    tarountula = h.Pokemon(400, hp=60, maxHp=60, playerIndex=1)
    spidops = h.Pokemon(401, hp=140, maxHp=140, playerIndex=1)
    articuno = h.Pokemon(policy.ROCKET_ARTICUNO_ID, hp=110, maxHp=110, playerIndex=1)
    obj = h.bare_policy(policy, opp_active=tarountula, opp_bench=[articuno, spidops])
    _real_effect_method(policy, obj)
    assert obj._effect_prevented(tarountula)
    assert obj._effect_prevented(articuno)
    assert not obj._effect_prevented(spidops)
    assert obj._articuno_escape_target(spidops)


def test_v16_powerful_hand_into_mist_is_rejected_as_zero_progress():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                     energies=[h.EnergyType.PSYCHIC])
    target = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1,
                       energyCards=[h.Card(policy.C.MIST_ENERGY)])
    obj = h.bare_policy(policy, hand_count=12, active=mine, opp_active=target)
    _real_effect_method(policy, obj)
    assert obj._alakazam_damage(policy.POWERFUL_HAND, target) == 0
    assert obj._score_attack(SimpleNamespace(attackId=policy.POWERFUL_HAND)) < 0


def test_v13_boss_does_not_abandon_grimmsnarl_ex_ko_for_morgrem():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                     energies=[h.EnergyType.PSYCHIC])
    grimmsnarl = h.Pokemon(policy.GRIMMSNARL_EX_ID, hp=320, maxHp=320, playerIndex=1)
    morgrem = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1)
    obj = h.bare_policy(policy, hand_count=19, active=mine,
                        opp_active=grimmsnarl, opp_bench=[morgrem])
    assert obj._active_best_dmg(grimmsnarl) >= grimmsnarl.hp
    assert obj._boss_damage_after_spend(morgrem) >= morgrem.hp
    assert obj._boss_target_score(morgrem) < 0


def test_v13_boss_escapes_articuno_lock_to_unprotected_spidops():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                     energies=[h.EnergyType.PSYCHIC])
    tarountula = h.Pokemon(400, hp=60, maxHp=60, playerIndex=1)
    articuno = h.Pokemon(policy.ROCKET_ARTICUNO_ID, hp=110, maxHp=110, playerIndex=1)
    spidops = h.Pokemon(401, hp=120, maxHp=140, playerIndex=1)
    obj = h.bare_policy(policy, hand_count=8, active=mine,
                        opp_active=tarountula, opp_bench=[articuno, spidops])
    _real_effect_method(policy, obj)
    assert obj._articuno_active_lock()
    assert obj._boss_target_score(articuno) < 0
    assert obj._boss_target_score(spidops) > 0


def test_v13_mist_prior_decays_after_all_four_public_copies_seen():
    policy = h.load_policy()
    crustle = h.Pokemon(345, playerIndex=1)
    obj = h.bare_policy(policy, opp_active=h.Pokemon(9000, playerIndex=1),
                        opp_bench=[crustle])
    policy._V9_STATE["mist_seen_serials"] = {("serial", i) for i in range(4)}
    obj.hand[policy.C.ENHANCED_HAMMER] = 1
    obj.discard[policy.C.ENHANCED_HAMMER] = 3
    assert obj._mist_seen_count() == 4
    assert obj._mist_probability() == 0.0
    assert not obj._should_reserve_last_hammer()


def test_v13_mist_prior_reserves_last_hammer_after_one_seen_in_crustle():
    policy = h.load_policy()
    crustle = h.Pokemon(345, playerIndex=1)
    obj = h.bare_policy(policy, opp_active=h.Pokemon(9000, playerIndex=1),
                        opp_bench=[crustle])
    policy._V9_STATE["mist_seen_serials"] = {("serial", 77)}
    obj.hand[policy.C.ENHANCED_HAMMER] = 1
    obj.discard[policy.C.ENHANCED_HAMMER] = 3
    assert obj._mist_probability() >= 0.70
    assert obj._should_reserve_last_hammer()


def test_v13_articuno_all_protected_board_unlocks_fez_breaker_mode():
    policy = h.load_policy()
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210,
                    energies=[h.EnergyType.COLORLESS])
    tarountula = h.Pokemon(400, hp=60, maxHp=60, playerIndex=1)
    articuno = h.Pokemon(policy.ROCKET_ARTICUNO_ID, hp=110, maxHp=110, playerIndex=1)
    obj = h.bare_policy(policy, active=fez, opp_active=tarountula, opp_bench=[articuno])
    _real_effect_method(policy, obj)
    assert obj._articuno_breaker_required()
    assert obj._fez_mode(fez) == "ALTERNATE_ATTACKER"
    assert obj._fez_attach_score(fez, True, h.Card(policy.C.PSYCHIC_ENERGY)) > 0


def test_v15_boss_does_not_treat_projected_draw_ko_as_already_secured():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                     energies=[h.EnergyType.PSYCHIC])
    grimmsnarl = h.Pokemon(policy.GRIMMSNARL_EX_ID, hp=320, maxHp=320, playerIndex=1)
    morgrem = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1)
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140)
    kadabra = h.Pokemon(policy.C.KADABRA, hp=100, maxHp=100)
    obj = h.bare_policy(policy, hand_count=15, active=mine,
                        bench=[dudun, kadabra],
                        opp_active=grimmsnarl, opp_bench=[morgrem])
    obj.hand[policy.C.ALAKAZAM] = 1
    # Current 300 misses, but Boss(-1) + Dudunsparce(+3) + Alakazam evolve(+2)
    # reaches 19 cards / 380 damage without using a Supporter draw.
    assert obj._active_best_dmg(grimmsnarl) == 300
    assert obj._boss_active_reachable_damage(grimmsnarl) >= grimmsnarl.hp
    assert obj._boss_target_score(morgrem) > 0
