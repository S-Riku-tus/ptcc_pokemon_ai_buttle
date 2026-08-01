"""Regressions for the v26 board-count floor and Grimmsnarl matchup rules.

Every case below is taken from a recorded v24/v25 ladder decision:

* 88471860 t3 / 88465811 t3 - Hilda taken over Dawn with an empty bench.
* 88455088 t11 / 88497435 t5 - turn ended (or attacked) on one body while
  Shaymin sat in hand.
* 88417447 t3 / 88422743 t4  - Run Away Draw cycled a two-body board to one.
* 88469739 t9 - Lana's Aid taken over a Boss on Froslass with two live copies
  putting a damage counter on every Ability Pokemon each Checkup.
"""

from __future__ import annotations

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option
from test_v24_runtime_logic import _attack, _attacking_alakazam


def _install_v26_cards(policy):
    """Fill in the trainer/evolution data the shared v11 stub does not carry.

    The stub card DB only models Pokemon and Energy, so Poffin/Dawn/Hilda score
    as unknown cards and the module-level evolution and Checkup-ability indexes
    come out empty. Patch them per test instead of editing the inherited golden
    fixture, which other versions' tests still assert against.
    """
    policy.card_table[policy.C.BUDDY_POFFIN] = h.CardData(
        policy.C.BUDDY_POFFIN, h.CardType.ITEM, name="Buddy-Buddy Poffin")
    policy.card_table[policy.C.POKE_PAD] = h.CardData(
        policy.C.POKE_PAD, h.CardType.ITEM, name="Poke Pad")
    policy.card_table[policy.C.SACRED_ASH] = h.CardData(
        policy.C.SACRED_ASH, h.CardType.ITEM, name="Sacred Ash")
    policy.card_table[policy.C.HILDA] = h.CardData(
        policy.C.HILDA, h.CardType.SUPPORTER, name="Hilda")
    policy.card_table[policy.C.DAWN] = h.CardData(
        policy.C.DAWN, h.CardType.SUPPORTER, name="Dawn")
    policy.EVOLVES_FROM_INDEX[646] = {647}
    policy.EVOLVES_FROM_INDEX[647] = {policy.GRIMMSNARL_EX_ID}
    policy.CHECKUP_COUNTER_ABILITY_IDS.add(policy.FROSLASS_ID)
    policy.card_table[policy.FROSLASS_ID].skills = [
        h.Skill("During Pokemon Checkup, put 1 damage counter on each Pokemon "
                "that has an Ability (both yours and your opponent's).")
    ]
    for card_id in (policy.C.ALAKAZAM, policy.C.KADABRA, policy.C.DUDUNSPARCE,
                    policy.C.FEZANDIPITI_EX):
        if not policy.card_table[card_id].skills:
            policy.card_table[card_id].skills = [h.Skill("Draw cards.")]
    return policy


def _lone_fezandipiti(policy):
    return h.Pokemon(policy.C.FEZANDIPITI_EX, hp=170, maxHp=210, serial=10)


# ── C1/C2: board-count floor ─────────────────────────────────────────────────

def test_v26_benches_shaymin_rather_than_ending_on_one_body():
    policy = _install_v26_cards(h.load_policy())
    policy.diag_reset()
    shaymin = _option(h.OptionType.PLAY, index=0)
    end = _option(h.OptionType.END)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.SHAYMIN, policy.C.SACRED_ASH],
        hand_count=2,
        active=_lone_fezandipiti(policy),
        opponent_active=h.Pokemon(9000, hp=200, maxHp=200, playerIndex=1, serial=20),
        options=[end, shaymin],
    )

    assert obj._board_body_count() == 1
    assert obj._body_floor_critical()
    assert obj._offered_body_add()
    assert obj.choose() == [1]


def test_v26_prefers_poffin_over_shaymin_at_the_body_floor():
    """Buddy-Buddy Poffin puts two bodies down for one Item."""
    policy = _install_v26_cards(h.load_policy())
    poffin = _option(h.OptionType.PLAY, index=0)
    shaymin = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BUDDY_POFFIN, policy.C.SHAYMIN],
        hand_count=2,
        active=_lone_fezandipiti(policy),
        opponent_active=h.Pokemon(9000, hp=200, maxHp=200, playerIndex=1, serial=20),
        options=[shaymin, poffin],
    )

    assert obj._poffin_targets_left() > 0
    assert obj.choose() == [1]


def test_v26_body_floor_does_not_fire_once_a_second_body_exists():
    policy = _install_v26_cards(h.load_policy())
    shaymin = _option(h.OptionType.PLAY, index=0)
    end = _option(h.OptionType.END)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.SHAYMIN],
        hand_count=1,
        active=_lone_fezandipiti(policy),
        bench=[h.Pokemon(policy.C.ABRA, hp=50, maxHp=50, serial=11)],
        opponent_active=h.Pokemon(9000, hp=200, maxHp=200, playerIndex=1, serial=20),
        options=[end, shaymin],
    )

    assert obj._board_body_count() == 2
    assert not obj._body_floor_critical()
    assert obj.choose() == [0]


def test_v26_body_floor_never_outranks_a_terminal_win_attack():
    policy = _install_v26_cards(h.load_policy())
    shaymin = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.SHAYMIN],
        hand_count=12,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(
            policy.C.FEZANDIPITI_EX, hp=100, maxHp=210, playerIndex=1, serial=20
        ),
        options=[shaymin, _attack(policy)],
        prizes=2,
    )

    assert obj._board_body_count() == 1
    assert obj.choose() == [1]


# ── C1c: Run Away Draw may not shrink the board to one body ──────────────────

def test_v26_refuses_run_away_draw_that_leaves_a_single_body():
    policy = _install_v26_cards(h.load_policy())
    policy.diag_reset()
    dudunsparce = h.Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140, serial=12)
    ability = _option(h.OptionType.ABILITY, index=0, area=h.AreaType.BENCH)
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=4,
        active=_attacking_alakazam(policy),
        bench=[dudunsparce],
        opponent_active=h.Pokemon(9000, hp=200, maxHp=200, playerIndex=1, serial=20),
        options=[ability, _option(h.OptionType.END)],
    )

    assert obj._board_body_count() == 2
    assert not obj._offered_body_add()
    assert obj._score_ability(ability) < 0


def test_v26_allows_run_away_draw_when_the_body_can_be_replaced():
    policy = _install_v26_cards(h.load_policy())
    dudunsparce = h.Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140, serial=12)
    ability = _option(h.OptionType.ABILITY, index=0, area=h.AreaType.BENCH)
    abra_play = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.ABRA],
        hand_count=4,
        active=_attacking_alakazam(policy),
        bench=[dudunsparce],
        opponent_active=h.Pokemon(9000, hp=200, maxHp=200, playerIndex=1, serial=20),
        options=[ability, abra_play, _option(h.OptionType.END)],
    )

    assert obj._offered_body_add()
    assert obj._score_ability(ability) > 0


# ── C3: Dawn beats Hilda when no Evolution target is in play ─────────────────

def test_v26_takes_dawn_over_hilda_with_no_evolution_target_in_play():
    policy = _install_v26_cards(h.load_policy())
    policy.diag_reset()
    hilda = _option(h.OptionType.PLAY, index=0)
    dawn = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.HILDA, policy.C.DAWN],
        hand_count=8,
        active=_lone_fezandipiti(policy),
        opponent_active=h.Pokemon(9000, hp=200, maxHp=200, playerIndex=1, serial=20),
        options=[hilda, dawn],
    )

    assert not obj._evolution_target_on_board()
    assert obj._score_play(dawn) > obj._score_play(hilda)


def test_v26_keeps_hilda_when_an_evolution_target_is_in_play():
    policy = _install_v26_cards(h.load_policy())
    hilda = _option(h.OptionType.PLAY, index=0)
    dawn = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.HILDA, policy.C.DAWN],
        hand_count=8,
        active=h.Pokemon(policy.C.ABRA, hp=50, maxHp=50, serial=10),
        bench=[h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70, serial=11)],
        opponent_active=h.Pokemon(9000, hp=200, maxHp=200, playerIndex=1, serial=20),
        options=[hilda, dawn],
    )

    assert obj._evolution_target_on_board()
    assert obj._score_play(hilda) >= obj._score_play(dawn)


# ── C2b: bench threat is detected before the sniper is Active ────────────────

def test_v26_sees_grimmsnarl_bench_snipe_one_evolution_early():
    policy = _install_v26_cards(h.load_policy())
    morgrem = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1, serial=20)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.SHAYMIN],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[
            h.Pokemon(policy.C.KADABRA, hp=80, maxHp=80, serial=11),
            h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70, serial=12),
        ],
        opponent_active=morgrem,
        options=[_option(h.OptionType.PLAY, index=0)],
    )

    # Morgrem itself has no bench snipe; Grimmsnarl ex, one evolution away, does.
    assert policy.GRIMMSNARL_EX_ID in obj._evolution_closure(647)
    assert obj._opponent_ready_bench_damage() == 0
    assert obj._opponent_board_bench_damage() == 30
    assert obj._opp_threatens_bench()


def test_v26_bench_threat_needs_two_exposed_bodies_and_a_spare_slot():
    policy = _install_v26_cards(h.load_policy())
    morgrem = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1, serial=20)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.SHAYMIN],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[h.Pokemon(policy.C.KADABRA, hp=80, maxHp=80, serial=11)],
        opponent_active=morgrem,
        options=[_option(h.OptionType.PLAY, index=0)],
    )

    assert obj._opponent_board_bench_damage() == 30
    assert not obj._opp_threatens_bench()


# ── C4: a Checkup damage-counter engine is a valid equal-prize Boss target ───

def test_v26_gusts_froslass_over_an_equal_prize_active_knock_out():
    policy = _install_v26_cards(h.load_policy())
    policy.diag_reset()
    froslass_a = h.Pokemon(policy.FROSLASS_ID, hp=90, maxHp=90, playerIndex=1, serial=20)
    froslass_b = h.Pokemon(policy.FROSLASS_ID, hp=90, maxHp=90, playerIndex=1, serial=21)
    morgrem = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1, serial=22)
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=17,
        active=_attacking_alakazam(policy),
        bench=[
            h.Pokemon(policy.C.DUDUNSPARCE, hp=100, maxHp=140, serial=11),
            h.Pokemon(policy.C.KADABRA, hp=60, maxHp=80, serial=12),
        ],
        opponent_active=morgrem,
        opponent_bench=[froslass_a, froslass_b],
        options=[boss, _attack(policy)],
        prizes=4,
    )

    assert obj._checkup_counter_engine_upgrade(froslass_a, morgrem)
    assert obj._boss_target_score(froslass_a) > 0
    assert obj.choose() == [0]


def test_v26_checkup_engine_upgrade_needs_at_least_two_taxed_bodies():
    policy = _install_v26_cards(h.load_policy())
    froslass = h.Pokemon(policy.FROSLASS_ID, hp=90, maxHp=90, playerIndex=1, serial=20)
    morgrem = h.Pokemon(647, hp=100, maxHp=100, playerIndex=1, serial=22)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=17,
        active=h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70, serial=10),
        opponent_active=morgrem,
        opponent_bench=[froslass],
        options=[_option(h.OptionType.PLAY, index=0)],
        prizes=4,
    )

    # Dunsparce has no Ability, so Freezing Shroud is not taxing us at all.
    assert not obj._checkup_counter_engine_upgrade(froslass, morgrem)


def test_v26_checkup_engine_upgrade_never_trades_down_on_prizes():
    policy = _install_v26_cards(h.load_policy())
    froslass = h.Pokemon(policy.FROSLASS_ID, hp=90, maxHp=90, playerIndex=1, serial=20)
    active_ex = h.Pokemon(
        policy.GRIMMSNARL_EX_ID, hp=300, maxHp=320, playerIndex=1, serial=22
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=17,
        active=_attacking_alakazam(policy),
        bench=[h.Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140, serial=11)],
        opponent_active=active_ex,
        opponent_bench=[froslass],
        options=[_option(h.OptionType.PLAY, index=0)],
        prizes=4,
    )

    assert not obj._checkup_counter_engine_upgrade(froslass, active_ex)
