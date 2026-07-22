from __future__ import annotations

from types import SimpleNamespace

import test_v11_runtime_logic as h
from test_v13_runtime_logic import _real_effect_method


def test_v16_end_beats_protected_powerful_hand_fallback():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    target = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1,
                       energyCards=[h.Card(policy.C.MIST_ENERGY)])
    obj = h.bare_policy(policy, active=mine, opp_active=target)
    _real_effect_method(policy, obj)
    attack = SimpleNamespace(type=h.OptionType.ATTACK, attackId=policy.POWERFUL_HAND)
    end = SimpleNamespace(type=h.OptionType.END)
    obj.context = h.SelectContext.MAIN
    obj.select = SimpleNamespace(contextCard=None, option=[attack, end])

    assert obj._score_attack(attack) < 0
    assert obj._score(end) > obj._score_attack(attack)


def test_v14_psychic_attachment_beats_rich_cycle_when_it_enables_attack():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM)
    dunsparce = h.Pokemon(policy.C.DUNSPARCE)
    obj = h.bare_policy(policy, active=mine, bench=[dunsparce],
                        opp_active=h.Pokemon(9000, playerIndex=1))
    psychic = h.Card(policy.C.PSYCHIC_ENERGY)
    obj.me.hand = [psychic, h.Card(policy.C.ENRICHING_ENERGY)]
    obj.hand[policy.C.PSYCHIC_ENERGY] = 1
    obj.hand[policy.C.ENRICHING_ENERGY] = 1
    obj.state.energyAttached = False
    obj._deck_spend_ok = lambda *args, **kwargs: True

    assert obj._active_alakazam_can_be_fueled()
    assert obj._enriching_attach_score(dunsparce) < 0
    obj.select = SimpleNamespace(contextCard=psychic, option=[])
    assert obj._score_attach_target(mine, is_active=True) == 28000


def test_v14_boss_escapes_mist_lock_for_low_value_bench_ko():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    locked = h.Pokemon(879, hp=140, maxHp=140, playerIndex=1,
                       energyCards=[h.Card(policy.C.MIST_ENERGY)])
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70, playerIndex=1)
    obj = h.bare_policy(policy, hand_count=9, active=mine,
                        opp_active=locked, opp_bench=[dunsparce])
    _real_effect_method(policy, obj)

    damage = obj._boss_damage_after_spend(dunsparce)
    assert obj._boss_effect_lock_escape_ko(dunsparce, damage)
    assert obj._boss_target_score(dunsparce) > 0
    assert obj._score_play_trainer(h.Card(policy.C.BOSS_ORDERS)) > 0


def test_v14_seen_mist_reserves_all_hammers_from_non_mist_targets():
    policy = h.load_policy()
    telepath_target = h.Pokemon(
        878,
        energies=[h.EnergyType.PSYCHIC],
        energyCards=[h.Card(policy.C.TELEPATH_ENERGY)],
        playerIndex=1,
    )
    crustle = h.Pokemon(345, playerIndex=1)
    obj = h.bare_policy(policy, opp_active=telepath_target, opp_bench=[crustle])
    policy._V9_STATE["mist_seen_serials"] = {("serial", 77)}
    obj.hand[policy.C.ENHANCED_HAMMER] = 3

    assert obj._mist_probability() >= 0.70
    assert obj._should_reserve_hammer_for_seen_mist()
    assert obj._score_play_trainer(h.Card(policy.C.ENHANCED_HAMMER)) < 0


def test_v14_attached_mist_releases_reservation_and_plays_hammer():
    policy = h.load_policy()
    mist_target = h.Pokemon(
        879,
        energies=[h.EnergyType.COLORLESS],
        energyCards=[h.Card(policy.C.MIST_ENERGY)],
        playerIndex=1,
    )
    obj = h.bare_policy(policy, opp_active=mist_target)
    policy._V9_STATE["mist_seen_serials"] = {("serial", 77)}
    obj.hand[policy.C.ENHANCED_HAMMER] = 3

    assert not obj._should_reserve_hammer_for_seen_mist()
    assert obj._score_play_trainer(h.Card(policy.C.ENHANCED_HAMMER)) == 26000


def test_v14_rich_draw_stop_has_defined_deterministic_backup_eta():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    backup = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    opponent = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    obj = h.bare_policy(policy, hand_count=8, active=active, bench=[backup],
                        opp_active=opponent)

    assert obj._backup_eta() == 0
    assert not obj._enrich_draw_needed()
