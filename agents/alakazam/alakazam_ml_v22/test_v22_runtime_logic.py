from __future__ import annotations

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


def test_v22_dudun_continues_after_ko_when_visible_abra_is_not_a_route():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=20)
    abra = h.Pokemon(policy.C.ABRA, serial=21)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=5,
        active=_attacking_alakazam(policy),
        bench=[dudun, abra],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        opponent_bench=[h.Pokemon(9001, hp=100, playerIndex=1)],
        options=[ability, _attack_option(policy)],
    )
    obj.me.deckCount = 30

    assert obj._backup_eta() == 99
    assert obj._continuity_draw_needed()
    assert not obj._draw_redundant_for_chosen_target("dudun")
    assert obj._score_ability(ability) > obj._score_attack(_attack_option(policy))
    assert obj.choose() == [0]


def test_v22_dudun_stops_when_one_turn_backup_is_concrete():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=20)
    abra = h.Pokemon(policy.C.ABRA, serial=21)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.RARE_CANDY, policy.C.ALAKAZAM, policy.C.PSYCHIC_ENERGY],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[dudun, abra],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        options=[ability, _attack_option(policy)],
    )

    assert obj._backup_eta() == 1
    assert not obj._continuity_draw_needed()
    assert obj._draw_redundant_for_chosen_target("dudun")
    assert obj.choose() == [1]


def test_v22_rebenches_dunsparce_before_extra_abra_when_cycle_is_offline():
    policy = h.load_policy()
    play_dunsparce = _option(h.OptionType.PLAY, index=0)
    play_abra = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.DUNSPARCE, policy.C.ABRA],
        hand_count=8,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        options=[play_dunsparce, play_abra, _attack_option(policy)],
    )

    assert obj._backup_eta() == 99
    assert obj._score_play_poke(h.Card(policy.C.DUNSPARCE)) > obj._score_play_poke(
        h.Card(policy.C.ABRA)
    )
    assert obj.choose() == [0]


def test_v22_progressively_builds_fez_for_visible_two_prize_snipe():
    policy = h.load_policy()
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, serial=30)
    backup = _attacking_alakazam(policy, serial=31)
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
        bench=[fez, backup],
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        opponent_bench=[h.Pokemon(666, hp=100, playerIndex=1)],
        options=[attach, _attack_option(policy)],
    )

    assert obj._backup_eta() == 0
    assert obj._fez_progressive_goal()
    assert obj._fez_progressive_build_allowed(fez)
    assert obj._fez_mode(fez) == "ALTERNATE_ATTACKER"
    assert obj._score_attach(attach) > 0
    assert obj.choose() == [0]


def test_v22_fez_build_never_breaks_exact_current_ko():
    policy = h.load_policy()
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, serial=40)
    backup = _attacking_alakazam(policy, serial=41)
    attach = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.PSYCHIC_ENERGY],
        hand_count=5,
        active=_attacking_alakazam(policy),
        bench=[fez, backup],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        opponent_bench=[h.Pokemon(666, hp=100, playerIndex=1)],
        options=[attach, attack],
    )

    assert obj._fez_progressive_build_allowed(fez)
    assert obj._score(attach) == 10
    assert obj._score(attack) > obj._score(attach)
    assert obj.choose() == [1]


def test_v22_fez_build_waits_until_alakazam_backup_is_safe():
    policy = h.load_policy()
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, serial=50)
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
        bench=[fez, h.Pokemon(policy.C.ABRA, serial=51)],
        opponent_active=h.Pokemon(9000, hp=300, playerIndex=1),
        opponent_bench=[h.Pokemon(666, hp=100, playerIndex=1)],
        options=[attach, _attack_option(policy)],
    )

    assert obj._backup_eta() == 99
    assert not obj._fez_progressive_build_allowed(fez)
    assert obj._score_attach(attach) < 0


def test_v22_ready_fez_promotion_recognizes_bench_ko():
    policy = h.load_policy()
    fez = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        energies=[
            h.EnergyType.PSYCHIC,
            h.EnergyType.PSYCHIC,
            h.EnergyType.PSYCHIC,
        ],
        serial=60,
    )
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, serial=61)
    obj = h.bare_policy(
        policy,
        opp_active=h.Pokemon(9000, hp=300, playerIndex=1),
        opp_bench=[h.Pokemon(666, hp=100, playerIndex=1)],
    )
    obj.context = h.SelectContext.TO_ACTIVE

    assert obj._promotion_attacks_next_turn(fez)
    assert obj._score_ko_promotion(fez) > obj._score_ko_promotion(dunsparce)


def test_v22_sufficient_hand_blocks_optional_dawn_after_attack_and_backup_are_ready():
    policy = h.load_policy()
    dawn = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.DAWN],
        hand_count=13,
        active=_attacking_alakazam(policy),
        bench=[_attacking_alakazam(policy, serial=71)],
        opponent_active=h.Pokemon(9000, hp=200, playerIndex=1),
        opponent_bench=[h.Pokemon(9001, hp=100, playerIndex=1)],
        options=[dawn, _attack_option(policy)],
    )

    assert not obj._optional_hand_growth_needed()
    assert obj._score_play_trainer(h.Card(policy.C.DAWN)) < 0
    assert obj.choose() == [1]


def test_v22_hand_target_is_soft_when_backup_is_missing():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=80)
    abra = h.Pokemon(policy.C.ABRA, serial=81)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=14,
        active=_attacking_alakazam(policy),
        bench=[dudun, abra],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        opponent_bench=[h.Pokemon(9001, hp=100, playerIndex=1)],
        options=[ability, _attack_option(policy)],
    )
    obj.me.deckCount = 30

    assert obj._continuity_draw_needed()
    assert obj._optional_hand_growth_needed()
    assert obj._score_ability(ability) > obj._score_attack(_attack_option(policy))
    assert obj.choose() == [0]
