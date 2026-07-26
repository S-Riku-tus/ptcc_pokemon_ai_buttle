from __future__ import annotations

from types import SimpleNamespace as NS

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option


def _attacking_alakazam(policy, serial=10):
    return h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=serial,
    )


def _attack_option(policy):
    return _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)


def _own_choice(card):
    return NS(playerIndex=0, area=h.AreaType.BENCH, index=0)


def test_v22_dudun_draws_for_missing_backup_not_for_raw_hand_growth():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=20)
    abra = h.Pokemon(policy.C.ABRA, serial=21)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=13,
        active=_attacking_alakazam(policy),
        bench=[dudun, abra],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        opponent_bench=[h.Pokemon(9001, hp=100, playerIndex=1)],
        options=[ability, _attack_option(policy)],
    )
    obj.me.deckCount = 30

    assert obj._backup_eta() == 99
    assert obj._continuity_draw_needed()
    assert not obj._dudun_draw_satisfied()
    assert obj._score_ability(ability) == 20500
    assert obj.choose() == [0]


def test_v22_dudun_stops_only_when_attack_backup_and_hand_are_all_safe():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=20)
    abra = h.Pokemon(
        policy.C.ABRA,
        energies=[h.EnergyType.PSYCHIC],
        serial=21,
    )
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.RARE_CANDY, policy.C.ALAKAZAM],
        hand_count=10,
        active=_attacking_alakazam(policy),
        bench=[dudun, abra],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        options=[ability, attack],
    )

    assert obj._backup_eta() == 1
    assert obj._attack_chain_stable()
    assert obj._dudun_draw_satisfied()
    assert obj._score_ability(ability) < 0
    assert obj.choose() == [1]


def test_v22_normal_evolution_chain_has_two_turn_backup_eta():
    policy = h.load_policy()
    abra = h.Pokemon(
        policy.C.ABRA,
        energies=[h.EnergyType.PSYCHIC],
        serial=30,
    )
    obj = h.bare_policy(policy, bench=[abra])
    obj.me.hand = [h.Card(policy.C.KADABRA), h.Card(policy.C.ALAKAZAM)]
    obj.hand = h.Counter(card.id for card in obj.me.hand)

    assert obj._backup_eta() == 2
    assert obj._continuity_draw_needed()


def test_v22_rebuilds_first_dunsparce_before_redundant_third_line_body():
    policy = h.load_policy()
    play_dunsparce = _option(h.OptionType.PLAY, index=0)
    play_abra = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.DUNSPARCE, policy.C.ABRA],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[h.Pokemon(policy.C.ABRA, serial=31)],
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        options=[play_dunsparce, play_abra, _attack_option(policy)],
    )

    assert obj.field[policy.C.DUNSPARCE] + obj.field[policy.C.DUDUNSPARCE] == 0
    assert obj._score_play_poke(h.Card(policy.C.DUNSPARCE)) > obj._score_play_poke(
        h.Card(policy.C.ABRA)
    )
    assert obj.choose() == [0]


def test_v22_prefuels_public_backup_after_current_attack_is_online():
    policy = h.load_policy()
    attach = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    abra = h.Pokemon(policy.C.ABRA, serial=40)
    obj = _main_board(
        policy,
        hand_ids=[
            policy.C.PSYCHIC_ENERGY,
            policy.C.RARE_CANDY,
            policy.C.ALAKAZAM,
        ],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[abra],
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        options=[attach, _attack_option(policy)],
    )

    energy = obj.me.hand[0]
    assert obj._backup_prefuel_score(abra, energy) > 0
    assert obj._score_attach(attach) > obj._score_attack(_attack_option(policy))
    assert obj.choose() == [0]


def test_v22_active_attack_fuel_still_dominates_backup_prefuel():
    policy = h.load_policy()
    attach_active = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.ACTIVE,
        target_index=0,
    )
    attach_backup = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[
            policy.C.PSYCHIC_ENERGY,
            policy.C.RARE_CANDY,
            policy.C.ALAKAZAM,
        ],
        hand_count=8,
        active=h.Pokemon(policy.C.ALAKAZAM, serial=50),
        bench=[h.Pokemon(policy.C.ABRA, serial=51)],
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        options=[attach_backup, attach_active],
    )

    assert obj._score_attach(attach_active) > obj._score_attach(attach_backup)
    assert obj.choose() == [1]


def test_v22_post_ko_promotes_attack_route_before_recyclable_shield():
    policy = h.load_policy()
    obj = h.bare_policy(policy, opp_active=h.Pokemon(9000, hp=200, playerIndex=1))
    obj.context = h.SelectContext.TO_ACTIVE
    obj.me.hand = [
        h.Card(policy.C.RARE_CANDY),
        h.Card(policy.C.ALAKAZAM),
        h.Card(policy.C.PSYCHIC_ENERGY),
    ]
    obj.hand = h.Counter(card.id for card in obj.me.hand)
    abra = h.Pokemon(policy.C.ABRA, serial=60)
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, serial=61)

    assert obj._promotion_attacks_next_turn(abra)
    assert obj._score_ko_promotion(abra) > obj._score_ko_promotion(dunsparce)


def test_v22_post_ko_uses_shield_when_evolution_line_cannot_attack():
    policy = h.load_policy()
    obj = h.bare_policy(policy, opp_active=h.Pokemon(9000, hp=200, playerIndex=1))
    obj.context = h.SelectContext.TO_ACTIVE
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, serial=70)
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=71)
    kadabra = h.Pokemon(policy.C.KADABRA, serial=72)

    assert not obj._promotion_attacks_next_turn(kadabra)
    assert obj._score_ko_promotion(dunsparce) > obj._score_ko_promotion(dudun)
    assert obj._score_ko_promotion(dudun) > obj._score_ko_promotion(kadabra)


def test_v22_abra_end_turn_switch_preserves_the_evolution_line():
    policy = h.load_policy()
    threat = h.Pokemon(
        678,
        energies=[h.EnergyType.COLORLESS, h.EnergyType.COLORLESS],
        playerIndex=1,
    )
    obj = h.bare_policy(
        policy,
        active=h.Pokemon(policy.C.ABRA),
        opp_active=threat,
    )
    obj.context = h.SelectContext.SWITCH
    obj.select = NS(
        context=h.SelectContext.SWITCH,
        effect=h.Card(policy.C.ABRA),
        contextCard=None,
        option=[],
    )
    obj.opponent.prize = [h.Card(9100 + index) for index in range(5)]
    dunsparce = h.Pokemon(policy.C.DUNSPARCE)
    dudun = h.Pokemon(policy.C.DUDUNSPARCE)
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    kadabra = h.Pokemon(policy.C.KADABRA)

    scores = [
        obj._score_active_choice(_own_choice(card), card)
        for card in (dunsparce, dudun, fez, kadabra)
    ]
    assert scores[0] > scores[1] > scores[3] > scores[2]


def test_v22_munkidori_is_a_priority_engine_in_grimmsnarl_board():
    policy = h.load_policy()
    munkidori = h.Pokemon(
        112,
        energies=[h.EnergyType.DARKNESS],
        playerIndex=1,
    )
    obj = h.bare_policy(
        policy,
        opp_active=h.Pokemon(policy.GRIMMSNARL_EX_ID, playerIndex=1),
        opp_bench=[munkidori],
    )

    assert obj._boss_role_bonus(munkidori) >= 4900


def test_v22_does_not_progressively_build_ordinary_fez_attacker():
    policy = h.load_policy()
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, serial=80)
    attach = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.PSYCHIC_ENERGY],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[fez],
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        options=[attach, _attack_option(policy)],
    )

    assert obj._fez_mode(fez) == "DRAW_ONLY"
    assert obj._score_attach(attach) < 0


def test_v22_real_nonlethal_attack_still_beats_end():
    policy = h.load_policy()
    attack = _attack_option(policy)
    end = _option(h.OptionType.END)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=5,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        options=[end, attack],
    )

    assert obj._score(end) < obj._score(attack)
    assert obj.choose() == [1]
