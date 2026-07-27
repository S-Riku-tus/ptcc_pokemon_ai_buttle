from __future__ import annotations

import pytest

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _mark_mega_as_three_prize, _option


def _attacking_alakazam(policy):
    return h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=10,
    )


def _attack_option(policy):
    return _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)


def test_v20_terminal_two_prize_boss_is_absolute_even_when_active_also_wins():
    policy = h.load_policy()
    active = _attacking_alakazam(policy)
    opponent_active = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1, serial=20)
    bench_ex = h.Pokemon(666, hp=210, maxHp=210, playerIndex=1, serial=21)
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=12,
        active=active,
        opponent_active=opponent_active,
        opponent_bench=[bench_ex],
        options=[attack, boss],
        prizes=2,
    )

    assert obj._terminal_boss_targets() == [bench_ex]
    assert obj._score(boss) >= policy.TERMINAL_BOSS_SCORE
    assert obj._score(attack) == policy.BLOCKED_BY_TERMINAL_BOSS_SCORE
    assert obj.choose() == [1]


def test_v20_terminal_three_prize_boss_uses_mega_prize_count():
    policy = h.load_policy()
    _mark_mega_as_three_prize(policy)
    mega = h.Pokemon(678, hp=340, maxHp=340, playerIndex=1, serial=31)
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=18,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=500, playerIndex=1, serial=30),
        opponent_bench=[mega],
        options=[boss, attack],
        prizes=3,
    )

    assert policy.prize_count(mega) == 3
    assert obj._terminal_boss_targets() == [mega]
    assert obj.choose() == [0]


@pytest.mark.parametrize("remaining", [1, 4])
def test_v20_terminal_boss_gate_only_applies_at_two_or_three_prizes(remaining):
    policy = h.load_policy()
    target = h.Pokemon(666, hp=100, playerIndex=1, serial=41)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=10,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=500, playerIndex=1, serial=40),
        opponent_bench=[target],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _attack_option(policy),
        ],
        prizes=remaining,
    )

    assert obj._terminal_boss_targets() == []


def test_v20_terminal_boss_requires_post_spend_damage_and_effect_legality():
    policy = h.load_policy()
    target = h.Pokemon(666, hp=210, playerIndex=1, serial=51)
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=500, playerIndex=1, serial=50),
        opponent_bench=[target],
        options=[boss, _attack_option(policy)],
        prizes=2,
    )
    assert obj._terminal_boss_targets() == []

    target.hp = 200
    target.energyCards = [h.Card(policy.C.MIST_ENERGY, serial=500)]
    assert obj._terminal_boss_targets() == []


def test_v20_terminal_boss_requires_legal_supporter_and_offered_attack():
    policy = h.load_policy()
    target = h.Pokemon(666, hp=100, playerIndex=1, serial=61)
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=10,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=500, playerIndex=1, serial=60),
        opponent_bench=[target],
        options=[boss],
        prizes=2,
    )
    assert obj._terminal_boss_targets() == []

    obj.select.option.append(_attack_option(policy))
    obj.state.supporterPlayed = True
    assert obj._terminal_boss_targets() == []


@pytest.mark.parametrize(
    ("hp", "required"),
    [(70, 4), (130, 7), (140, 7), (210, 11), (300, 15), (340, 17)],
)
def test_v20_powerful_hand_required_hand_is_target_sized(hp, required):
    policy = h.load_policy()
    assert policy.AlakazamPolicy._attack_hand_required(h.Pokemon(9000, hp=hp)) == required


def test_v20_dawn_has_consistent_plus_two_net_hand_delta():
    policy = h.load_policy()
    dawn = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.DAWN],
        hand_count=8,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=500, playerIndex=1),
        options=[dawn],
    )

    assert obj._hand_delta(h.OptionType.PLAY, dawn) == 2
    assert obj._achievable_hand() == 10


def test_v20_target_priority_chooses_reachable_two_prizer_over_active_single():
    policy = h.load_policy()
    opponent_active = h.Pokemon(9000, hp=100, playerIndex=1, serial=70)
    bench_ex = h.Pokemon(666, hp=200, playerIndex=1, serial=71)
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=opponent_active,
        opponent_bench=[bench_ex],
        options=[boss, attack],
    )

    plan = obj._chosen_ko_plan()
    assert plan["target"] is bench_ex
    assert plan["next_actions"] == frozenset({"boss"})
    assert obj.choose() == [0]


def test_v20_target_priority_skips_unreachable_target_and_takes_next_ko():
    policy = h.load_policy()
    opponent_active = h.Pokemon(9000, hp=100, playerIndex=1, serial=80)
    unreachable_ex = h.Pokemon(666, hp=250, playerIndex=1, serial=81)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=opponent_active,
        opponent_bench=[unreachable_ex],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _attack_option(policy),
        ],
    )

    plan = obj._chosen_ko_plan()
    assert plan["target"] is opponent_active
    assert plan["next_actions"] == frozenset()
    assert obj.choose() == [1]


def test_v20_benched_run_away_draw_stops_after_chosen_target_is_lethal():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=91)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    attack = _attack_option(policy)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=10,
        active=_attacking_alakazam(policy),
        bench=[dudun, _attacking_alakazam(policy)],
        opponent_active=h.Pokemon(9000, hp=100, playerIndex=1, serial=90),
        options=[ability, attack],
    )

    assert obj._draw_redundant_for_chosen_target("dudun")
    assert obj._score_ability(ability) < 0
    assert obj.choose() == [1]


def test_v20_benched_run_away_draw_executes_when_it_is_in_minimum_ko_route():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=101)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[dudun],
        opponent_active=h.Pokemon(9000, hp=200, playerIndex=1, serial=100),
        options=[ability, _attack_option(policy)],
    )

    plan = obj._chosen_ko_plan()
    assert plan["actions"] == frozenset({"dudun"})
    assert obj._score_ability(ability) > 47000
    assert obj.choose() == [0]


def test_v20_enriching_draw_stops_only_after_surplus_target_hand_is_secured():
    policy = h.load_policy()
    target = h.Pokemon(9000, hp=100, playerIndex=1, serial=105)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.ENRICHING_ENERGY],
        hand_count=10,
        active=_attacking_alakazam(policy),
        opponent_active=target,
        options=[_attack_option(policy)],
    )
    assert not obj._enrich_draw_needed()

    target.hp = 200
    obj._ko_route_cache = {}
    obj._chosen_ko_plan_cache = None
    obj._chosen_ko_plan_cached = False
    assert obj._enrich_draw_needed()


def test_v20_boss_route_reserves_supporter_and_uses_non_supporter_draw():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=111)
    target = h.Pokemon(666, hp=240, playerIndex=1, serial=112)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS, policy.C.DAWN],
        hand_count=10,
        active=_attacking_alakazam(policy),
        bench=[dudun],
        opponent_active=h.Pokemon(9000, hp=500, playerIndex=1, serial=110),
        opponent_bench=[target],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _option(h.OptionType.PLAY, index=1),
            ability,
            _attack_option(policy),
        ],
    )

    plan = obj._chosen_ko_plan()
    assert plan["target"] is target
    assert plan["actions"] == frozenset({"boss", "dudun"})
    assert "dawn" not in plan["actions"]
    assert plan["next_actions"] == frozenset({"dudun"})
    assert obj.choose() == [2]


def test_v20_route_tiebreak_prefers_less_overkill_at_equal_action_and_deck_cost():
    policy = h.load_policy()
    dudun = h.Pokemon(policy.C.DUDUNSPARCE, serial=121)
    kadabra = h.Pokemon(policy.C.KADABRA, serial=122)
    ability = _option(h.OptionType.ABILITY, area=h.AreaType.BENCH, index=0)
    evolve = _option(
        h.OptionType.EVOLVE,
        index=0,
        target_area=h.AreaType.BENCH,
        target_index=1,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.ALAKAZAM],
        hand_count=9,
        active=_attacking_alakazam(policy),
        bench=[dudun, kadabra],
        opponent_active=h.Pokemon(9000, hp=200, playerIndex=1, serial=120),
        options=[ability, evolve, _attack_option(policy)],
    )

    plan = obj._active_route_plan()
    assert plan["actions"] == frozenset({"evolve_alakazam"})
    assert plan["damage"] == 220


def test_v20_hammer_route_binds_selection_to_the_planned_bench_target():
    policy = h.load_policy()
    active_mist = h.Card(policy.C.MIST_ENERGY, serial=1301)
    bench_mist = h.Card(policy.C.MIST_ENERGY, serial=1302)
    opponent_active = h.Pokemon(
        9000,
        hp=300,
        playerIndex=1,
        serial=130,
        energyCards=[active_mist],
    )
    target = h.Pokemon(
        666,
        hp=200,
        playerIndex=1,
        serial=131,
        energyCards=[bench_mist],
    )
    hammer = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.ENHANCED_HAMMER, policy.C.BOSS_ORDERS],
        hand_count=13,
        active=_attacking_alakazam(policy),
        opponent_active=opponent_active,
        opponent_bench=[target],
        options=[hammer, _option(h.OptionType.PLAY, index=1), _attack_option(policy)],
    )

    plan = obj._chosen_ko_plan()
    assert plan["target"] is target
    assert plan["next_actions"] == frozenset({"hammer"})
    obj._record_v9([0])
    assert (
        obj._hammer_target_score(bench_mist, target, h.AreaType.BENCH)
        > obj._hammer_target_score(active_mist, opponent_active, h.AreaType.ACTIVE)
    )
