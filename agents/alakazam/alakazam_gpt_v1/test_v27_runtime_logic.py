"""Regressions for the v27 pivot, target-priority, and draw-budget rules."""

from __future__ import annotations

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option
from test_v24_runtime_logic import _attack, _attacking_alakazam
from test_v26_runtime_logic import _install_v26_cards


def _install_generic_attack_line(policy):
    """Add a card-ID-agnostic Basic -> Stage 1 -> Stage 2 attack line."""
    basic_id, stage1_id, stage2_id = 9101, 9102, 9103
    policy.card_table[basic_id] = h.CardData(
        basic_id, name="Test Basic")
    policy.card_table[stage1_id] = h.CardData(
        stage1_id, name="Test Stage 1", stage1=True, attacks=[900])
    policy.card_table[stage2_id] = h.CardData(
        stage2_id, name="Test Main Attacker", stage2=True, attacks=[900])
    policy.EVOLVES_FROM_INDEX[basic_id] = {stage1_id}
    policy.EVOLVES_FROM_INDEX[stage1_id] = {stage2_id}
    return basic_id, stage1_id, stage2_id


# ── 1. Attach -> retreat -> Alakazam attack ─────────────────────────────────


def test_v27_attaches_enriching_to_active_fez_for_same_turn_retreat_ko():
    policy = _install_v26_cards(h.load_policy())
    rich = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.ACTIVE,
        target_index=0,
    )
    fezandipiti = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        hp=210,
        maxHp=210,
        serial=10,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.ENRICHING_ENERGY],
        hand_count=10,
        active=fezandipiti,
        bench=[_attacking_alakazam(policy)],
        opponent_active=h.Pokemon(
            9000, hp=200, maxHp=200, playerIndex=1, serial=20
        ),
        options=[_option(h.OptionType.END), rich],
    )

    route = obj._support_pivot_attack_route(
        fezandipiti, h.AreaType.ACTIVE, h.Card(policy.C.ENRICHING_ENERGY)
    )
    assert route is not None
    assert route["ko"]
    assert route["damage"] == 260
    assert obj._score_attach(rich) > 30_000
    assert obj.choose() == [1]


def test_v27_retreats_fueled_fez_into_ready_alakazam():
    policy = _install_v26_cards(h.load_policy())
    fezandipiti = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        hp=210,
        maxHp=210,
        energies=[h.EnergyType.COLORLESS],
        serial=10,
    )
    retreat = _option(h.OptionType.RETREAT)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=13,
        active=fezandipiti,
        bench=[_attacking_alakazam(policy)],
        opponent_active=h.Pokemon(
            9000, hp=200, maxHp=200, playerIndex=1, serial=20
        ),
        options=[_option(h.OptionType.END), retreat],
    )

    assert obj._score_retreat() > 30_000
    assert obj.choose() == [1]


# ── 2. Strategic target and evolution-line priority ──────────────────────────


def test_v27_generic_attack_line_orders_endpoint_then_pre_evolutions():
    policy = _install_v26_cards(h.load_policy())
    basic_id, stage1_id, stage2_id = _install_generic_attack_line(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=12,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(
            9000, hp=100, maxHp=100, playerIndex=1, serial=20
        ),
        opponent_bench=[
            h.Pokemon(basic_id, hp=70, maxHp=70, playerIndex=1, serial=21),
            h.Pokemon(stage1_id, hp=100, maxHp=100, playerIndex=1, serial=22),
            h.Pokemon(stage2_id, hp=120, maxHp=120, playerIndex=1, serial=23),
        ],
        options=[_option(h.OptionType.PLAY, index=0), _attack(policy)],
    )

    stage2 = obj.opponent.bench[2]
    stage1 = obj.opponent.bench[1]
    basic = obj.opponent.bench[0]
    filler = obj.opponent.active[0]
    scores = [
        obj._target_priority_score(target, obj._target_area(target))
        for target in (stage2, stage1, basic, filler)
    ]

    assert scores[0] > scores[1] > scores[2] > scores[3]


def test_v27_incidental_low_output_stage1_line_is_not_called_main_strategy():
    policy = _install_v26_cards(h.load_policy())
    basic_id, stage1_id = 9201, 9202
    policy.card_table[basic_id] = h.CardData(basic_id, name="Utility Basic")
    policy.card_table[stage1_id] = h.CardData(
        stage1_id, name="Utility Stage 1", stage1=True, attacks=[9200]
    )
    policy.ATTACK_TABLE[9200] = h.AttackData(9200, damage=30)
    policy.EVOLVES_FROM_INDEX[basic_id] = {stage1_id}
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=8,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(
            basic_id, hp=70, maxHp=70, playerIndex=1, serial=20
        ),
        options=[_attack(policy)],
    )

    assert obj._strategic_line_bonus(obj.opponent.active[0]) == 0


def test_v27_bosses_koable_main_attack_line_basic_over_disposable_active():
    policy = _install_v26_cards(h.load_policy())
    basic_id, _, _ = _install_generic_attack_line(policy)
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=6,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(
            9000, hp=80, maxHp=80, playerIndex=1, serial=20
        ),
        opponent_bench=[
            h.Pokemon(basic_id, hp=70, maxHp=70, playerIndex=1, serial=21)
        ],
        options=[_attack(policy), boss],
    )

    target = obj.opponent.bench[0]
    assert obj._boss_damage_after_spend(target) == 100
    assert obj._boss_target_score(target) > 0
    assert obj.choose() == [1]


def test_v27_prize_closing_ex_ko_still_beats_attack_line_basic():
    policy = _install_v26_cards(h.load_policy())
    basic_id, _, _ = _install_generic_attack_line(policy)
    active_ex = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        hp=100,
        maxHp=210,
        playerIndex=1,
        serial=20,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=8,
        active=_attacking_alakazam(policy),
        opponent_active=active_ex,
        opponent_bench=[
            h.Pokemon(basic_id, hp=70, maxHp=70, playerIndex=1, serial=21)
        ],
        options=[_option(h.OptionType.PLAY, index=0), _attack(policy)],
        prizes=2,
    )

    assert obj._boss_target_score(obj.opponent.bench[0]) < 0
    assert obj.choose() == [1]


# ── 3. Stop Run Away Draw at the chosen target's exact hand requirement ─────


def _draw_budget_board(policy, *, hand_count, target_hp=200, options=None):
    dudunsparce = h.Pokemon(
        policy.C.DUDUNSPARCE, hp=140, maxHp=140, serial=11
    )
    abra = h.Pokemon(policy.C.ABRA, hp=50, maxHp=50, serial=12)
    ability = _option(
        h.OptionType.ABILITY,
        index=0,
        area=h.AreaType.BENCH,
    )
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=hand_count,
        active=_attacking_alakazam(policy),
        bench=[dudunsparce, abra],
        opponent_active=h.Pokemon(
            9000, hp=target_hp, maxHp=target_hp, playerIndex=1, serial=20
        ),
        options=options or [ability, _attack(policy)],
    )
    return obj, ability


def test_v27_holds_run_away_draw_at_exact_ko_hand():
    policy = _install_v26_cards(h.load_policy())
    obj, ability = _draw_budget_board(policy, hand_count=10)

    assert obj._chosen_ko_plan()["required_hand"] == 10
    assert obj._optional_draw_budget_met("dudun")
    assert obj._score_ability(ability) < 0
    assert obj.choose() == [1]


def test_v27_uses_run_away_draw_when_one_card_short_of_ko():
    policy = _install_v26_cards(h.load_policy())
    obj, ability = _draw_budget_board(policy, hand_count=9)

    plan = obj._chosen_ko_plan()
    assert plan["actions"] == frozenset({"dudun"})
    assert not obj._optional_draw_budget_met("dudun")
    assert obj._score_ability(ability) > 40_000
    assert obj.choose() == [0]


def test_v27_holds_draw_but_uses_boss_at_exact_bench_ko_hand():
    policy = _install_v26_cards(h.load_policy())
    dudunsparce = h.Pokemon(
        policy.C.DUDUNSPARCE, hp=140, maxHp=140, serial=11
    )
    abra = h.Pokemon(policy.C.ABRA, hp=50, maxHp=50, serial=12)
    target = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        hp=200,
        maxHp=210,
        playerIndex=1,
        serial=21,
    )
    ability = _option(
        h.OptionType.ABILITY,
        index=0,
        area=h.AreaType.BENCH,
    )
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        bench=[dudunsparce, abra],
        opponent_active=h.Pokemon(
            9000, hp=300, maxHp=300, playerIndex=1, serial=20
        ),
        opponent_bench=[target],
        options=[ability, boss, _attack(policy)],
    )

    plan = obj._chosen_ko_plan()
    assert plan["target"] is target
    assert plan["actions"] == frozenset({"boss"})
    assert plan["hand"] == plan["required_hand"] == 10
    assert obj._optional_draw_budget_met("dudun")
    assert obj._score_ability(ability) < 0
    assert obj.choose() == [1]
