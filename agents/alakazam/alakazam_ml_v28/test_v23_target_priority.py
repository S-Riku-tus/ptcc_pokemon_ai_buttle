from __future__ import annotations

from types import SimpleNamespace as NS

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option


def _attacking_alakazam(policy):
    return h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=10,
    )


def _attack_option(policy):
    return _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)


def _boss_switch_policy(policy, targets, *, hand_count=12):
    obj = h.bare_policy(
        policy,
        hand_count=hand_count,
        active=_attacking_alakazam(policy),
        opp_active=h.Pokemon(9000, hp=100, playerIndex=1, serial=20),
        opp_bench=targets,
    )
    obj.state = NS(
        supporterPlayed=True,
        energyAttached=False,
        stadiumPlayed=False,
        stadium=[],
        turn=9,
        yourIndex=0,
        players=[obj.me, obj.opponent],
    )
    options = [
        NS(
            type=h.OptionType.CARD,
            area=h.AreaType.BENCH,
            index=index,
            playerIndex=1,
            inPlayArea=None,
            inPlayIndex=None,
        )
        for index in range(len(targets))
    ]
    obj.select = NS(
        context=h.SelectContext.SWITCH,
        contextCard=None,
        effect=h.Card(policy.C.BOSS_ORDERS),
        option=options,
        minCount=1,
        maxCount=1,
    )
    obj.context = h.SelectContext.SWITCH
    obj.obs = NS(current=obj.state, select=obj.select, logs=[])
    obj._ko_route_cache = {}
    obj._chosen_ko_plan_cache = None
    obj._chosen_ko_plan_cached = False
    return obj


def _own_end_turn_switch_policy(policy, targets):
    obj = h.bare_policy(
        policy,
        hand_count=8,
        active=h.Pokemon(policy.C.ABRA, serial=11),
        bench=targets,
        opp_active=h.Pokemon(9000, hp=100, playerIndex=1, serial=12),
    )
    obj.state = NS(
        supporterPlayed=False,
        energyAttached=False,
        stadiumPlayed=False,
        stadium=[],
        turn=3,
        yourIndex=0,
        players=[obj.me, obj.opponent],
    )
    obj.select = NS(
        context=h.SelectContext.SWITCH,
        contextCard=None,
        effect=h.Card(policy.C.ABRA),
        option=[
            NS(
                type=h.OptionType.CARD,
                area=h.AreaType.BENCH,
                index=index,
                playerIndex=0,
                inPlayArea=None,
                inPlayIndex=None,
            )
            for index in range(len(targets))
        ],
        minCount=1,
        maxCount=1,
    )
    obj.context = h.SelectContext.SWITCH
    obj.obs = NS(current=obj.state, select=obj.select, logs=[])
    return obj


def test_v23_boss_switch_dispatches_opponent_target_before_own_promotion_logic():
    policy = h.load_policy()
    dunsparce = h.Pokemon(
        policy.C.DUNSPARCE,
        hp=70,
        maxHp=70,
        playerIndex=1,
        serial=31,
    )
    bench_ex = h.Pokemon(
        666,
        hp=210,
        maxHp=210,
        playerIndex=1,
        serial=32,
    )
    obj = _boss_switch_policy(policy, [dunsparce, bench_ex], hand_count=12)

    # Boss must apply opponent prize/KO value and select the ex.
    assert obj._score(obj.select.option[1]) > obj._score(obj.select.option[0])
    assert obj.choose() == [1]


def test_v23_non_boss_switch_keeps_v22_own_shield_priority():
    policy = h.load_policy()
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, serial=41)
    fezandipiti = h.Pokemon(policy.C.FEZANDIPITI_EX, serial=42)
    obj = _own_end_turn_switch_policy(policy, [dunsparce, fezandipiti])

    assert obj._score(obj.select.option[0]) > obj._score(obj.select.option[1])
    assert obj.choose() == [0]


def test_v23_immediate_ordinary_active_to_ex_prize_upgrade_forces_boss():
    policy = h.load_policy()
    ordinary_active = h.Pokemon(
        9000,
        hp=100,
        maxHp=100,
        playerIndex=1,
        serial=50,
    )
    bench_ex = h.Pokemon(
        666,
        hp=200,
        maxHp=200,
        playerIndex=1,
        serial=51,
    )
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=ordinary_active,
        opponent_bench=[bench_ex],
        options=[attack, boss],
    )

    assert obj._boss_prize_upgrade_targets() == [bench_ex]
    assert obj._score(boss) >= policy.PRIZE_UPGRADE_BOSS_SCORE
    assert obj.choose() == [1]


def test_v23_ex_prize_upgrade_beats_competing_setup_supporter():
    policy = h.load_policy()
    ordinary_active = h.Pokemon(
        9000,
        hp=100,
        maxHp=100,
        playerIndex=1,
        serial=55,
    )
    bench_ex = h.Pokemon(
        666,
        hp=200,
        maxHp=200,
        playerIndex=1,
        serial=56,
    )
    boss = _option(h.OptionType.PLAY, index=0)
    dawn = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS, policy.C.DAWN],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=ordinary_active,
        opponent_bench=[bench_ex],
        options=[dawn, _attack_option(policy), boss],
    )

    # Dawn and Boss are mutually exclusive Supporters. A guaranteed two-prize
    # KO must not be discarded in favour of generic setup.
    assert obj._score(boss) > obj._score(dawn)
    assert obj.choose() == [2]


def test_v23_prize_upgrade_uses_post_boss_hand_damage():
    policy = h.load_policy()
    bench_ex = h.Pokemon(
        666,
        hp=210,
        maxHp=210,
        playerIndex=1,
        serial=61,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1, serial=60),
        opponent_bench=[bench_ex],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _attack_option(policy),
        ],
    )

    # 220 before Boss, only 200 after spending it: the ex is not a legal KO.
    assert obj._boss_prize_upgrade_targets() == []


def test_v23_does_not_gust_away_from_an_active_game_winning_ko():
    policy = h.load_policy()
    bench_ex = h.Pokemon(666, hp=200, playerIndex=1, serial=71)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1, serial=70),
        opponent_bench=[bench_ex],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _attack_option(policy),
        ],
        prizes=1,
    )

    assert obj._boss_prize_upgrade_targets() == []
    assert obj.choose() == [1]


def test_v24_explicit_ex_ko_request_supersedes_live_single_prize_exception():
    policy = h.load_policy()
    froslass = h.Pokemon(
        policy.FROSLASS_ID,
        hp=90,
        maxHp=90,
        playerIndex=1,
        serial=80,
    )
    bench_ex = h.Pokemon(666, hp=200, playerIndex=1, serial=81)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=froslass,
        opponent_bench=[bench_ex],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _attack_option(policy),
        ],
    )

    # The teacher often removes Froslass, but v23's blanket engine exception
    # contradicted the explicit requirement to take a concrete ex KO over a
    # one-prize Active. v24 narrows the teacher influence to target tie-breaks.
    assert obj._boss_role_bonus(froslass) >= 2200
    assert obj._boss_prize_upgrade_targets() == [bench_ex]
    assert obj.choose() == [0]


def test_v23_target_priority_has_explicit_ex_tier_over_ordinary_basic():
    policy = h.load_policy()
    ordinary = h.Pokemon(9000, hp=100, playerIndex=1, serial=90)
    bench_ex = h.Pokemon(666, hp=200, playerIndex=1, serial=91)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=ordinary,
        opponent_bench=[bench_ex],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _attack_option(policy),
        ],
    )

    assert obj._target_priority_score(bench_ex, h.AreaType.BENCH) > (
        obj._target_priority_score(ordinary, h.AreaType.ACTIVE)
    )
