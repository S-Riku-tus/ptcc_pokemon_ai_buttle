from __future__ import annotations

from types import SimpleNamespace as NS

import test_v11_runtime_logic as h
from test_v13_runtime_logic import _real_effect_method


def _main(obj, options=()):
    obj.context = h.SelectContext.MAIN
    obj.select = NS(context=h.SelectContext.MAIN, contextCard=None, effect=None,
                    option=list(options), minCount=1, maxCount=1)
    obj.state.energyAttached = getattr(obj.state, "energyAttached", False)
    obj.state.retreated = getattr(obj.state, "retreated", False)
    return obj


def test_v16_deck_and_ranker_are_byte_identical_to_v15():
    here = h.Path(__file__).resolve().parent
    parent = here.parent / "alakazam_ml_v15"
    assert (here / "deck.csv").read_bytes() == (parent / "deck.csv").read_bytes()
    assert (here / "ranker_model.json").read_bytes() == (
        parent / "ranker_model.json"
    ).read_bytes()


def test_v16_cubchoo_lock_retreats_to_ready_alakazam():
    policy = h.load_policy()
    policy.diag_reset()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC], serial=25)
    backup = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC], serial=27)
    cubchoo = h.Pokemon(506, energies=[h.EnergyType.COLORLESS], playerIndex=1, serial=77)
    retreat = NS(type=h.OptionType.RETREAT)
    end = NS(type=h.OptionType.END)
    obj = _main(h.bare_policy(policy, active=active, bench=[backup], opp_active=cubchoo),
                [retreat, end])
    obj.state.turn = 5
    obj.state.yourIndex = 0
    obj.state.players = [obj.me, obj.opponent]
    obs = NS(current=obj.state, logs=[NS(type=15, playerIndex=1, attackId=716)])

    policy._remember_attack_disable(obs)
    assert obj._attack_disabled_on_active()
    assert obj._score_retreat() == 45000
    assert obj.choose() == [0]


def test_v16_attack_lock_memory_clears_after_active_changes():
    policy = h.load_policy()
    policy.diag_reset()
    first = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC], serial=25)
    second = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC], serial=27)
    obj = _main(h.bare_policy(policy, active=first, bench=[second],
                              opp_active=h.Pokemon(506, playerIndex=1)))
    obj.state.turn = 7
    obj.state.yourIndex = 0
    obj.state.players = [obj.me, obj.opponent]
    policy._V9_STATE.update({"attack_disabled_key": (25, policy.C.ALAKAZAM),
                             "attack_disabled_turn": 7})
    obj.me.active = [second]
    policy._remember_attack_disable(NS(current=obj.state, logs=[]))
    assert policy._V9_STATE["attack_disabled_key"] is None


def test_v16_hammer_never_hits_active_that_post_hammer_attack_kos():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    energy = h.Card(policy.C.TELEPATH_ENERGY)
    target = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1,
                       energies=[h.EnergyType.PSYCHIC], energyCards=[energy])
    obj = _main(h.bare_policy(policy, hand_count=8, active=mine, opp_active=target))
    assert obj._hammer_target_is_doomed(energy, target, h.AreaType.ACTIVE)
    assert obj._hammer_target_score(energy, target, h.AreaType.ACTIVE) < 0
    assert obj._score_play_trainer(h.Card(policy.C.ENHANCED_HAMMER)) < 0


def test_v16_hammer_preserves_exact_powerful_hand_ko():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    target = h.Pokemon(9000, hp=160, maxHp=160, playerIndex=1,
                       energies=[h.EnergyType.PSYCHIC],
                       energyCards=[h.Card(policy.C.TELEPATH_ENERGY)])
    obj = _main(h.bare_policy(policy, hand_count=8, active=mine, opp_active=target))
    assert obj._lethal_now()
    assert obj._hammer_damage_after_spend(target) == 140
    assert obj._score_play_trainer(h.Card(policy.C.ENHANCED_HAMMER)) < 0


def test_v16_terminal_active_win_blocks_nonwinning_boss():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    active_ex = h.Pokemon(policy.GRIMMSNARL_EX_ID, hp=200, maxHp=320, playerIndex=1)
    normal = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1)
    obj = _main(h.bare_policy(policy, hand_count=10, active=mine,
                              opp_active=active_ex, opp_bench=[normal]))
    obj.me.prize = [h.Card(1), h.Card(2)]
    assert obj._active_win_reachable_without_gust()
    assert obj._boss_target_score(normal) < 0


def test_v16_boss_prefers_koable_ex_over_single_prizer():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    active = h.Pokemon(9000, hp=300, maxHp=300, playerIndex=1)
    ex = h.Pokemon(policy.GRIMMSNARL_EX_ID, hp=200, maxHp=320, playerIndex=1)
    normal = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1)
    obj = _main(h.bare_policy(policy, hand_count=11, active=mine,
                              opp_active=active, opp_bench=[normal, ex]))
    obj.me.prize = [h.Card(1), h.Card(2)]
    assert obj._boss_target_score(ex) > obj._boss_target_score(normal)


def test_v16_abra_switch_prefers_dunsparce_and_protects_koable_kadabra():
    policy = h.load_policy()
    grim = h.Pokemon(policy.GRIMMSNARL_EX_ID,
                     energies=[h.EnergyType.COLORLESS, h.EnergyType.COLORLESS],
                     playerIndex=1)
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70)
    abra = h.Pokemon(policy.C.ABRA, hp=50, maxHp=50)
    kadabra = h.Pokemon(policy.C.KADABRA, hp=80, maxHp=80)
    obj = h.bare_policy(policy, bench=[dunsparce, abra, kadabra], opp_active=grim)
    obj.context = h.SelectContext.SWITCH
    obj.select = NS(contextCard=None, effect=h.Card(policy.C.ABRA), option=[])
    obj.hand[policy.C.ALAKAZAM] = 1
    assert obj._abra_switch_score(dunsparce) > obj._abra_switch_score(abra)
    assert obj._abra_switch_score(abra) > obj._abra_switch_score(kadabra)


def test_v16_fezandipiti_is_only_an_early_teleport_wall():
    policy = h.load_policy()
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    abra = h.Pokemon(policy.C.ABRA, hp=50, maxHp=50)
    obj = h.bare_policy(policy, opp_active=h.Pokemon(9000, playerIndex=1))
    assert len(obj.opponent.prize) == 6
    assert obj._abra_switch_score(fez) > obj._abra_switch_score(abra)
    obj.opponent.prize = [h.Card(1), h.Card(2), h.Card(3), h.Card(4)]
    assert obj._abra_switch_score(fez) < obj._abra_switch_score(abra)


def test_v16_all_protected_targets_concentrate_energy_on_fezandipiti():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, energies=[h.EnergyType.COLORLESS])
    mine = h.Pokemon(policy.C.ABRA)
    locked = h.Pokemon(9000, playerIndex=1,
                       energyCards=[h.Card(policy.C.MIST_ENERGY)])
    obj = _main(h.bare_policy(policy, active=active, bench=[fez, mine], opp_active=locked))
    _real_effect_method(policy, obj)
    obj.select.effect = h.Card(policy.C.PSYCHIC_ENERGY)
    assert obj._effect_breaker_required()
    assert obj._score_attach_target(mine, is_active=False) < 0
    assert obj._score_attach_target(fez, is_active=False) > 0


def test_v16_grimmsnarl_safe_board_caps_extra_low_hp_basics():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    backup = h.Pokemon(policy.C.ABRA)
    grim = h.Pokemon(policy.GRIMMSNARL_EX_ID, hp=300, maxHp=330,
                     energies=[h.EnergyType.COLORLESS, h.EnergyType.COLORLESS],
                     playerIndex=1)
    munkidori = h.Pokemon(112, energies=[h.EnergyType.DARKNESS], playerIndex=1)
    obj = _main(h.bare_policy(policy, active=active, bench=[backup], opp_active=grim,
                              opp_bench=[munkidori]))
    assert obj._grimmsnarl_matchup()
    assert obj._score_play_poke(h.Card(policy.C.ABRA)) < 0
    assert obj._score_play_poke(h.Card(policy.C.DUNSPARCE)) > 0
    obj.field[policy.C.DUNSPARCE] = 1
    assert obj._score_play_poke(h.Card(policy.C.DUNSPARCE)) < 0


def test_v16_sufficient_damage_and_backup_stop_optional_search():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    backup = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    target = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    attack = NS(type=h.OptionType.ATTACK, attackId=policy.POWERFUL_HAND)
    obj = _main(h.bare_policy(policy, hand_count=8, active=active,
                              bench=[backup], opp_active=target), [attack])
    obj.me.prize = [1]
    assert obj._ko_and_continuity_locked()
    assert obj._score_play_trainer(h.Card(policy.C.POKE_PAD)) < 0
    assert obj._score_play_trainer(h.Card(policy.C.HILDA)) < 0


def test_v16_backup_eta_counts_evolution_age_and_energy():
    policy = h.load_policy()
    fresh = h.Pokemon(policy.C.ABRA)
    fresh.appearThisTurn = True
    obj = h.bare_policy(policy, bench=[fresh], opp_active=h.Pokemon(9000, playerIndex=1))
    obj.hand[policy.C.KADABRA] = 1
    obj.hand[policy.C.ALAKAZAM] = 1
    obj.hand[policy.C.PSYCHIC_ENERGY] = 1
    obj.me.hand = [h.Card(policy.C.PSYCHIC_ENERGY)]
    assert obj._backup_eta() == 2
    fresh.appearThisTurn = False
    assert obj._backup_eta() == 1
    obj.hand[policy.C.PSYCHIC_ENERGY] = 0
    obj.me.hand = []
    assert obj._backup_eta() == 2


def test_v16_energy_enabled_ko_must_beat_end():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    target = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    attack = NS(type=h.OptionType.ATTACK, attackId=policy.POWERFUL_HAND)
    end = NS(type=h.OptionType.END)
    obj = _main(h.bare_policy(policy, hand_count=8, active=active, opp_active=target),
                [attack, end])
    assert obj._score(attack) > obj._score(end)
    assert obj.choose() == [0]


def test_v16_deck_spend_requires_enough_turns_to_finish():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    target = h.Pokemon(9000, hp=400, maxHp=400, playerIndex=1)
    obj = _main(h.bare_policy(policy, hand_count=4, active=active, opp_active=target))
    obj.me.deckCount = 6
    assert obj._turns_to_deckout(3) == 4
    assert not obj._deck_spend_ok(cost=3, allow_lethal=False)


def test_v16_ml_keeps_grimmsnarl_tactical_board_rule_only():
    source = (h.Path(__file__).resolve().parent / "ml_runtime.py").read_text(encoding="utf-8")
    assert "TACTICAL_RULE_ONLY_OPPONENT_IDS" in source
    assert '"tactical_board_rule"' in source
