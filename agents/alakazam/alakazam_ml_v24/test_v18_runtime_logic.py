from __future__ import annotations

from collections import Counter
from types import SimpleNamespace as NS

import test_v11_runtime_logic as h
from test_v13_runtime_logic import _real_effect_method


def _option(option_type, *, index=None, area=None, target_area=None,
            target_index=None, attack_id=None):
    return NS(
        type=option_type,
        index=index,
        area=area,
        inPlayArea=target_area,
        inPlayIndex=target_index,
        attackId=attack_id,
    )


def _main_board(
    policy,
    *,
    hand_ids,
    hand_count,
    active,
    bench=(),
    opponent_active,
    opponent_bench=(),
    options,
    prizes=6,
):
    obj = h.bare_policy(
        policy,
        hand_count=hand_count,
        active=active,
        bench=bench,
        opp_active=opponent_active,
        opp_bench=opponent_bench,
    )
    obj.me.hand = [
        h.Card(card_id, serial=100 + index)
        for index, card_id in enumerate(hand_ids)
    ]
    obj.me.handCount = hand_count
    obj.me.prize = [h.Card(9000 + index) for index in range(prizes)]
    obj.hand = Counter(hand_ids)
    obj.state = NS(
        supporterPlayed=False,
        energyAttached=False,
        stadiumPlayed=False,
        stadium=[],
        turn=9,
        yourIndex=0,
        players=[obj.me, obj.opponent],
    )
    obj.select = NS(
        context=h.SelectContext.MAIN,
        contextCard=None,
        effect=None,
        option=list(options),
        minCount=1,
        maxCount=1,
    )
    obj.context = h.SelectContext.MAIN
    obj.stadium_id = 0
    obj.obs = NS(current=obj.state, select=obj.select, logs=[])
    _real_effect_method(policy, obj)
    return obj


def _mark_mega_as_three_prize(policy):
    policy.card_table[678].megaEx = True


def test_v18_hammer_route_dominates_one_prize_boss():
    policy = h.load_policy()
    policy.diag_reset()
    _mark_mega_as_three_prize(policy)
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    lucario = h.Pokemon(
        678,
        hp=340,
        maxHp=340,
        playerIndex=1,
        energyCards=[h.Card(policy.C.MIST_ENERGY)],
    )
    riolu = h.Pokemon(9000, hp=70, maxHp=70, playerIndex=1)
    hammer = _option(h.OptionType.PLAY, index=0)
    boss = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.ENHANCED_HAMMER, policy.C.BOSS_ORDERS],
        hand_count=21,
        active=alakazam,
        opponent_active=lucario,
        opponent_bench=[riolu],
        options=[hammer, boss],
    )

    plan = obj._active_route_plan()

    assert plan["damage"] == 400
    assert plan["actions"] == frozenset({"hammer"})
    assert obj._boss_target_score(riolu) < 0
    assert obj._score_play_trainer(h.Card(policy.C.ENHANCED_HAMMER)) > 45000


def test_v18_rich_and_evolution_route_dominates_low_prize_boss():
    policy = h.load_policy()
    policy.diag_reset()
    _mark_mega_as_three_prize(policy)
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    kadabra = h.Pokemon(
        policy.C.KADABRA,
        energies=[h.EnergyType.PSYCHIC],
    )
    lucario = h.Pokemon(678, hp=340, maxHp=340, playerIndex=1)
    hariyama = h.Pokemon(
        9000,
        hp=150,
        maxHp=150,
        playerIndex=1,
        energies=[h.EnergyType.COLORLESS, h.EnergyType.COLORLESS],
    )
    rich = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    evolve = _option(
        h.OptionType.EVOLVE,
        index=1,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    boss = _option(h.OptionType.PLAY, index=2)
    obj = _main_board(
        policy,
        hand_ids=[
            policy.C.ENRICHING_ENERGY,
            policy.C.ALAKAZAM,
            policy.C.BOSS_ORDERS,
        ],
        hand_count=12,
        active=alakazam,
        bench=[kadabra],
        opponent_active=lucario,
        opponent_bench=[hariyama],
        options=[rich, evolve, boss],
    )

    plan = obj._active_route_plan()

    assert plan["hand"] == 17
    assert plan["damage"] == 340
    assert plan["actions"] == frozenset({"enriching", "evolve_alakazam"})
    assert obj._boss_target_score(hariyama) < 0
    assert obj._score_attach(rich) > 45000
    assert obj._score_evolve(evolve) > 45000


def test_v18_dawn_winning_route_dominates_makuhita_boss():
    policy = h.load_policy()
    policy.diag_reset()
    _mark_mega_as_three_prize(policy)
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    lucario = h.Pokemon(678, hp=440, maxHp=440, playerIndex=1)
    makuhita = h.Pokemon(
        9000,
        hp=80,
        maxHp=80,
        playerIndex=1,
        energies=[h.EnergyType.COLORLESS, h.EnergyType.COLORLESS],
    )
    dawn = _option(h.OptionType.PLAY, index=0)
    boss = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.DAWN, policy.C.BOSS_ORDERS],
        hand_count=21,
        active=alakazam,
        opponent_active=lucario,
        opponent_bench=[makuhita],
        options=[dawn, boss],
        prizes=2,
    )

    plan = obj._active_route_plan()

    assert plan["winning"]
    assert plan["hand"] == 23
    assert plan["damage"] == 460
    assert plan["actions"] == frozenset({"dawn"})
    assert obj._boss_target_score(makuhita) < 0
    assert obj._score_play_trainer(h.Card(policy.C.DAWN)) == 88000


def test_v18_still_uses_boss_when_bench_ko_wins_and_active_is_unreachable():
    policy = h.load_policy()
    policy.diag_reset()
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    tank = h.Pokemon(678, hp=440, maxHp=440, playerIndex=1)
    two_prize_target = h.Pokemon(666, hp=100, maxHp=100, playerIndex=1)
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=8,
        active=alakazam,
        opponent_active=tank,
        opponent_bench=[two_prize_target],
        options=[boss],
        prizes=2,
    )

    assert not obj._active_route_plan()["ko"]
    assert obj._boss_target_score(two_prize_target) > 0


def test_v18_board_wipe_gate_attacks_without_setup_detour():
    policy = h.load_policy()
    policy.diag_reset()
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    final_body = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    poffin = _option(h.OptionType.PLAY, index=0)
    attack = _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BUDDY_POFFIN],
        hand_count=12,
        active=alakazam,
        opponent_active=final_body,
        options=[poffin, attack],
    )

    assert obj._terminal_win_attack_offered()
    assert obj._score(poffin) < 0
    assert obj._score(attack) == 95000
    assert obj.choose() == [1]


def test_v18_active_dudunsparce_pivots_directly_to_board_wipe():
    policy = h.load_policy()
    policy.diag_reset()
    dudunsparce = h.Pokemon(policy.C.DUDUNSPARCE)
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    final_body = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    ability = _option(
        h.OptionType.ABILITY,
        area=h.AreaType.ACTIVE,
        index=0,
    )
    poffin = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BUDDY_POFFIN],
        hand_count=8,
        active=dudunsparce,
        bench=[alakazam],
        opponent_active=final_body,
        options=[ability, poffin],
    )

    assert obj._terminal_pivot_win(ability)
    assert obj._score_ability(ability) == 88000
    assert obj.choose() == [0]
    assert policy.diag_snapshot()["dudun_used_for_terminal_pivot"] == 1


def test_v18_records_dudunsparce_ability_lock_separately():
    policy = h.load_policy()
    policy.diag_reset()
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    dudunsparce = h.Pokemon(policy.C.DUDUNSPARCE)
    opponent = h.Pokemon(9000, hp=300, maxHp=300, playerIndex=1)
    end = _option(h.OptionType.END)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=8,
        active=alakazam,
        bench=[dudunsparce],
        opponent_active=opponent,
        options=[end],
    )

    obj._record_v18(0, end)

    assert policy.diag_snapshot()["dudun_block_ability_lock"] == 1


def test_v18_route_does_not_evolve_one_abra_twice():
    policy = h.load_policy()
    policy.diag_reset()
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    abra = h.Pokemon(policy.C.ABRA)
    opponent = h.Pokemon(9000, hp=320, maxHp=320, playerIndex=1)
    candy = _option(h.OptionType.PLAY, index=0)
    evolve = _option(
        h.OptionType.EVOLVE,
        index=2,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.RARE_CANDY, policy.C.ALAKAZAM, policy.C.KADABRA],
        hand_count=14,
        active=alakazam,
        bench=[abra],
        opponent_active=opponent,
        options=[candy, evolve],
    )

    assert not obj._active_route_plan()["ko"]


def test_v18_route_can_evolve_two_distinct_abras():
    policy = h.load_policy()
    policy.diag_reset()
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    first_abra = h.Pokemon(policy.C.ABRA)
    second_abra = h.Pokemon(policy.C.ABRA)
    opponent = h.Pokemon(9000, hp=320, maxHp=320, playerIndex=1)
    candy = _option(h.OptionType.PLAY, index=0)
    evolve = _option(
        h.OptionType.EVOLVE,
        index=2,
        target_area=h.AreaType.BENCH,
        target_index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.RARE_CANDY, policy.C.ALAKAZAM, policy.C.KADABRA],
        hand_count=14,
        active=alakazam,
        bench=[first_abra, second_abra],
        opponent_active=opponent,
        options=[candy, evolve],
    )

    plan = obj._active_route_plan()

    assert plan["ko"]
    assert plan["actions"] == frozenset({"rare_candy", "evolve_kadabra"})


def test_v18_hammer_does_not_bypass_non_energy_protection():
    policy = h.load_policy()
    policy.diag_reset()
    alakazam = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    rocket_basic = h.Pokemon(
        400,
        hp=100,
        maxHp=100,
        playerIndex=1,
        energyCards=[h.Card(policy.C.MIST_ENERGY)],
    )
    articuno = h.Pokemon(
        policy.ROCKET_ARTICUNO_ID,
        playerIndex=1,
    )
    hammer = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.ENHANCED_HAMMER],
        hand_count=20,
        active=alakazam,
        opponent_active=rocket_basic,
        opponent_bench=[articuno],
        options=[hammer],
    )

    assert obj._effect_prevented(rocket_basic)
    assert not obj._active_route_plan()["ko"]
