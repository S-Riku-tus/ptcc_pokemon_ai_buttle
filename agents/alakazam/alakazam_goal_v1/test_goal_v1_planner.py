from __future__ import annotations

from types import SimpleNamespace

from goal_planner import (
    GoalCandidate,
    choose_development_action,
    choose_goal,
    choose_next_action,
    development_action_score,
)

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option


def _goal(
    key,
    *,
    winning=False,
    prizes=1,
    priority=0,
    action_count=0,
    deck_cost=0,
):
    return GoalCandidate(
        target_key=key,
        payload={"key": key},
        winning=winning,
        prizes=prizes,
        priority=priority,
        action_count=action_count,
        deck_cost=deck_cost,
        needs_boss=False,
        is_active=True,
        damage=100,
        target_hp=100,
    )


def _attacking_alakazam(policy, *, serial=10):
    return h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=serial,
    )


def _attack(policy):
    return _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)


def _boss_metadata(policy):
    policy.card_table[policy.C.BOSS_ORDERS] = h.CardData(
        policy.C.BOSS_ORDERS,
        cardType=h.CardType.SUPPORTER,
    )


def test_goal_selector_uses_outcome_lexicographic_order():
    engine = _goal("engine", prizes=1, priority=50000)
    ex = _goal("ex", prizes=2, priority=1000, action_count=1)
    win = _goal("win", winning=True, prizes=1, priority=0, action_count=3)

    assert choose_goal([engine, ex]).target_key == "ex"
    assert choose_goal([engine, ex, win]).target_key == "win"


def test_goal_lock_is_stable_but_never_hides_prize_upgrade():
    locked = _goal("locked", prizes=1, priority=3000)
    marginal = _goal("marginal", prizes=1, priority=4000)
    upgrade = _goal("upgrade", prizes=2, priority=0)

    assert choose_goal(
        [locked, marginal], locked_target_key="locked"
    ).target_key == "locked"
    assert choose_goal(
        [locked, upgrade], locked_target_key="locked"
    ).target_key == "upgrade"


def test_goal_action_order_finishes_setup_before_boss():
    actions = {"boss", "hammer", "dudun", "evolve_kadabra"}
    assert choose_next_action(actions) == "evolve_kadabra"
    actions.remove("evolve_kadabra")
    assert choose_next_action(actions) == "dudun"
    actions.remove("dudun")
    assert choose_next_action(actions) == "hammer"
    actions.remove("hammer")
    assert choose_next_action(actions) == "boss"


def test_single_prize_engine_develops_before_two_prize_fez():
    assert choose_development_action(
        {"deploy_fezandipiti", "evolve_dudunsparce"}
    ) == "evolve_dudunsparce"
    assert choose_development_action(
        {
            "deploy_fezandipiti",
            "evolve_dudunsparce",
            "evolve_kadabra",
            "evolve_alakazam",
        }
    ) == "evolve_alakazam"
    assert (
        development_action_score("evolve_alakazam")
        > development_action_score("evolve_kadabra")
        > development_action_score("evolve_dudunsparce")
        > development_action_score("deploy_fezandipiti")
    )


def test_dudunsparce_evolution_boost_exists_only_when_fez_competes():
    policy = h.load_policy()
    dunsparce = h.Pokemon(policy.C.DUNSPARCE)
    obj = h.bare_policy(
        policy,
        active=dunsparce,
        opp_active=h.Pokemon(9000, playerIndex=1),
    )
    obj.hand[policy.C.DUDUNSPARCE] = 1
    option = SimpleNamespace(
        inPlayArea=h.AreaType.ACTIVE,
        inPlayIndex=0,
        index=0,
    )
    old_get_card = policy.get_card
    policy.get_card = lambda obs, area, index, player: (
        h.Card(policy.C.DUDUNSPARCE)
        if area == h.AreaType.HAND
        else dunsparce
    )
    try:
        assert obj._score_evolve(option) == 19000
        obj.hand[policy.C.FEZANDIPITI_EX] = 1
        obj._articuno_breaker_required = lambda: True
        obj._fez_bench_worthwhile = lambda: True
        assert obj._score_evolve(option) == development_action_score(
            "evolve_dudunsparce"
        )
    finally:
        policy.get_card = old_get_card


def test_goal_v1_bosses_koable_bench_ex_over_unkoable_active_copy():
    policy = h.load_policy()
    _boss_metadata(policy)
    active_ex = h.Pokemon(
        666, hp=260, maxHp=260, playerIndex=1, serial=20
    )
    damaged_bench_copy = h.Pokemon(
        666, hp=200, maxHp=260, playerIndex=1, serial=21
    )
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=active_ex,
        opponent_bench=[damaged_bench_copy],
        options=[_attack(policy), boss],
    )

    plan = obj._chosen_ko_plan()
    assert plan is not None
    assert plan["target"] is damaged_bench_copy
    assert plan["needs_boss"]
    assert obj.choose() == [1]


def test_goal_v1_keeps_clean_two_hit_pressure_on_grimmsnarl_ex():
    policy = h.load_policy()
    _boss_metadata(policy)
    grimmsnarl = h.Pokemon(
        policy.GRIMMSNARL_EX_ID,
        hp=420,
        maxHp=420,
        playerIndex=1,
        serial=30,
    )
    munkidori = h.Pokemon(
        112,
        hp=110,
        maxHp=110,
        playerIndex=1,
        serial=31,
        energies=[h.EnergyType.DARKNESS],
    )
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=18,
        active=_attacking_alakazam(policy),
        opponent_active=grimmsnarl,
        opponent_bench=[munkidori],
        options=[boss, attack],
    )

    assert obj._active_offered_attack_damage(grimmsnarl) == 360
    assert obj._grim_ex_two_hit_pressure()
    assert obj._boss_target_score(munkidori) < 0
    assert obj.choose() == [1]


def test_goal_v1_active_dudunsparce_cycles_last_two_cards_into_ko():
    policy = h.load_policy()
    dudunsparce = h.Pokemon(policy.C.DUDUNSPARCE, serial=50)
    alakazam = _attacking_alakazam(policy, serial=51)
    opponent = h.Pokemon(
        9000, hp=180, maxHp=180, playerIndex=1, serial=52
    )
    ability = _option(
        h.OptionType.ABILITY,
        area=h.AreaType.ACTIVE,
        index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=7,
        active=dudunsparce,
        bench=[alakazam],
        opponent_active=opponent,
        options=[ability, _option(h.OptionType.END)],
    )
    obj.me.deckCount = 2

    route = obj._active_dudun_low_deck_cycle_route()
    assert route is not None and route["ko"]
    assert route["damage"] == 180
    assert obj.choose() == [0]


def test_goal_v1_requires_full_dudunsparce_retreat_cost_for_attach_route():
    policy = h.load_policy()
    dudunsparce = h.Pokemon(
        policy.C.DUDUNSPARCE,
        energies=[h.EnergyType.PSYCHIC],
        serial=60,
    )
    attach = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.ACTIVE,
        target_index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.PSYCHIC_ENERGY],
        hand_count=8,
        active=dudunsparce,
        bench=[_attacking_alakazam(policy, serial=61)],
        opponent_active=h.Pokemon(
            9000, hp=140, maxHp=140, playerIndex=1, serial=62
        ),
        options=[attach, _option(h.OptionType.END)],
    )
    source = obj.me.hand[0]
    assert obj._dudun_retreat_attack_route(source) is None

    dudunsparce.energies.append(h.EnergyType.PSYCHIC)
    assert obj._dudun_retreat_attack_route(source) is not None
