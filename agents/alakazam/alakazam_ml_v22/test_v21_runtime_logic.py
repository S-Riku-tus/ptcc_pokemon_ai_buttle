from __future__ import annotations

from types import SimpleNamespace as NS

import test_v11_runtime_logic as h
from test_v13_runtime_logic import _real_effect_method
from test_v18_runtime_logic import _main_board, _option


def _attacking_alakazam(policy, *, energy=True):
    return h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC] if energy else [],
        serial=10,
    )


def _attack_option(policy):
    return _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)


def _own_choice(card):
    return NS(playerIndex=0, area=h.AreaType.BENCH, index=0)


def test_v21_dudunsparce_stops_at_exact_ko_when_a_future_line_exists():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=20)
    abra = h.Pokemon(policy.C.ABRA, serial=21)
    ability = _option(
        h.OptionType.ABILITY,
        area=h.AreaType.BENCH,
        index=0,
    )
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.RARE_CANDY, policy.C.ALAKAZAM, policy.C.PSYCHIC_ENERGY],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[dudun, abra],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        options=[ability, attack],
    )

    assert obj._active_route_plan()["actions"] == frozenset()
    assert obj._draw_redundant_for_chosen_target("dudun")
    assert obj._score_ability(ability) < 0
    assert obj.choose() == [1]


def test_v21_dudunsparce_keeps_narrow_single_attacker_continuity_exception():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=20)
    ability = _option(
        h.OptionType.ABILITY,
        area=h.AreaType.BENCH,
        index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=5,
        active=_attacking_alakazam(policy),
        bench=[dudun],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        opponent_bench=[h.Pokemon(9001, hp=100, playerIndex=1)],
        options=[ability, _attack_option(policy)],
    )
    obj.me.deckCount = 30

    assert obj._active_route_plan()["actions"] == frozenset()
    assert not obj._draw_redundant_for_chosen_target("dudun")


def test_v21_abra_switch_uses_pivots_before_ex_and_evolution_line():
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
    obj.state.turn = 5
    obj.opponent.prize = [h.Card(9000 + index) for index in range(5)]

    dunsparce = h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70)
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140)
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    kadabra = h.Pokemon(policy.C.KADABRA, hp=80, maxHp=80)

    scores = [
        obj._score_active_choice(_own_choice(card), card)
        for card in (dunsparce, dudun, fez, kadabra)
    ]
    assert scores[0] > scores[1] > scores[2] > scores[3]


def test_v21_late_abra_switch_does_not_expose_fez_to_a_visible_ko():
    policy = h.load_policy()
    policy.ATTACK_TABLE[983].damage = 270
    threat = h.Pokemon(
        678,
        energies=[h.EnergyType.COLORLESS],
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
    obj.opponent.prize = [h.Card(9100 + index) for index in range(3)]
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    kadabra = h.Pokemon(policy.C.KADABRA, hp=80, maxHp=80)

    assert obj._score_active_choice(
        _own_choice(fez), fez
    ) < obj._score_active_choice(_own_choice(kadabra), kadabra)


def test_v21_ko_promotion_attacks_if_possible_otherwise_uses_shield_order():
    policy = h.load_policy()
    target = h.Pokemon(9000, hp=200, playerIndex=1)
    obj = h.bare_policy(policy, opp_active=target)
    obj.context = h.SelectContext.TO_ACTIVE
    obj.select = NS(
        context=h.SelectContext.TO_ACTIVE,
        effect=None,
        contextCard=None,
        option=[],
    )
    obj.opponent.prize = [h.Card(9100 + index) for index in range(4)]

    ready = _attacking_alakazam(policy)
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70)
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140)
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    kadabra = h.Pokemon(policy.C.KADABRA, hp=80, maxHp=80)

    assert obj._score_active_choice(
        _own_choice(ready), ready
    ) > obj._score_active_choice(_own_choice(dunsparce), dunsparce)
    shield_scores = [
        obj._score_active_choice(_own_choice(card), card)
        for card in (dunsparce, dudun, fez, kadabra)
    ]
    assert shield_scores[0] > shield_scores[1] > shield_scores[2] > shield_scores[3]


def test_v21_end_never_beats_a_real_nonlethal_attack():
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

    assert obj._progress_attack_offered()
    assert obj._score(end) == -100000
    assert obj.choose() == [1]


def test_v21_zero_effect_powerful_hand_is_not_forced():
    policy = h.load_policy()
    target = h.Pokemon(
        9000,
        hp=100,
        playerIndex=1,
        energyCards=[h.Card(policy.C.MIST_ENERGY)],
    )
    attack = _attack_option(policy)
    end = _option(h.OptionType.END)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=10,
        active=_attacking_alakazam(policy),
        opponent_active=target,
        options=[attack, end],
    )
    _real_effect_method(policy, obj)

    assert not obj._progress_attack_offered()
    assert obj._score_attack(attack) < 0
    assert obj.choose() == [1]


def test_v21_dynamic_core_line_values_basic_enabler():
    policy = h.load_policy()
    policy.EVOLUTION_ROOT_BY_ID[9000] = "riolu"
    policy.EVOLUTION_ROOT_BY_ID[678] = "riolu"
    policy.EVOLUTION_LINE_CEILING["riolu"] = 5200
    riolu = h.Pokemon(9000, hp=80, maxHp=80, playerIndex=1)
    lucario = h.Pokemon(678, hp=340, maxHp=340, playerIndex=1)
    obj = h.bare_policy(
        policy,
        opp_active=lucario,
        opp_bench=[riolu],
    )

    assert obj._evolution_line_role_bonus(riolu) >= 1900
    assert obj._target_value(riolu) > policy.prize_count(riolu) * 2200


def test_v21_late_fez_is_blocked_when_hand_is_already_safe():
    policy = h.load_policy()
    play_fez = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.FEZANDIPITI_EX],
        hand_count=10,
        active=_attacking_alakazam(policy),
        bench=[h.Pokemon(policy.C.DUNSPARCE)],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1),
        options=[play_fez, _attack_option(policy)],
    )
    obj.opponent.prize = [h.Card(9100 + index) for index in range(3)]
    policy._V9_STATE["own_ko_turn"] = None

    assert obj._fez_two_prize_exposure()
    assert not obj._fez_bench_worthwhile()
    assert obj._score_play_poke(h.Card(policy.C.FEZANDIPITI_EX)) < 0


def test_v21_late_fez_is_allowed_when_flip_the_script_creates_ko():
    policy = h.load_policy()
    play_fez = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.FEZANDIPITI_EX],
        hand_count=6,
        active=_attacking_alakazam(policy),
        bench=[h.Pokemon(policy.C.ABRA)],
        opponent_active=h.Pokemon(9000, hp=160, playerIndex=1),
        options=[play_fez, _attack_option(policy)],
    )
    obj.opponent.prize = [h.Card(9100 + index) for index in range(3)]
    policy._V9_STATE["own_ko_turn"] = obj.state.turn - 1

    assert obj._fez_draw_creates_ko()
    assert obj._fez_entry_urgent()
    assert not obj._fez_two_prize_exposure()
    assert obj._fez_bench_worthwhile()
