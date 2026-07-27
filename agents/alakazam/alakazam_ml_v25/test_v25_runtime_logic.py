"""Focused regressions for the conservative v24 -> v25 rebuild."""

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option
from test_v24_runtime_logic import _attack, _attacking_alakazam


def _install_archaludon(policy):
    policy.card_table[policy.ARCHALUDON_EX_ID] = h.CardData(
        policy.ARCHALUDON_EX_ID,
        attacks=[901],
        ex=True,
    )


def test_v25_immediate_active_two_prize_ko_dominates_cinderace():
    policy = h.load_policy()
    _install_archaludon(policy)
    active_archaludon = h.Pokemon(
        policy.ARCHALUDON_EX_ID,
        hp=300,
        maxHp=400,
        playerIndex=1,
        serial=20,
    )
    cinderace = h.Pokemon(
        policy.CINDERACE_ID,
        hp=150,
        maxHp=260,
        playerIndex=1,
        serial=21,
    )
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=15,
        active=_attacking_alakazam(policy),
        opponent_active=active_archaludon,
        opponent_bench=[cinderace],
        options=[boss, attack],
    )

    assert obj._active_best_dmg(active_archaludon) == 300
    assert obj._boss_target_score(cinderace) < 0
    assert obj.choose() == [1]


def test_v25_keeps_clean_archaludon_two_hit_over_cinderace_gust():
    policy = h.load_policy()
    _install_archaludon(policy)
    active_archaludon = h.Pokemon(
        policy.ARCHALUDON_EX_ID,
        hp=400,
        maxHp=400,
        playerIndex=1,
        serial=30,
    )
    cinderace = h.Pokemon(
        policy.CINDERACE_ID,
        hp=150,
        maxHp=260,
        playerIndex=1,
        serial=31,
    )
    duraludon = h.Pokemon(
        169,
        hp=130,
        maxHp=130,
        playerIndex=1,
        serial=32,
    )
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=12,
        active=_attacking_alakazam(policy),
        opponent_active=active_archaludon,
        opponent_bench=[cinderace, duraludon],
        options=[boss, attack],
    )

    assert obj._active_offered_attack_damage(active_archaludon) == 240
    assert obj._arch_ex_two_hit_pressure()
    assert obj._boss_target_score(cinderace) < 0
    assert obj._boss_target_score(duraludon) < 0
    assert obj.choose() == [1]


def test_v25_live_froslass_stays_above_online_munkidori():
    policy = h.load_policy()
    froslass = h.Pokemon(
        policy.FROSLASS_ID,
        hp=90,
        maxHp=90,
        playerIndex=1,
        serial=40,
    )
    munkidori = h.Pokemon(
        112,
        hp=110,
        maxHp=110,
        playerIndex=1,
        serial=41,
        energies=[h.EnergyType.DARKNESS],
    )
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=10,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(
            9000,
            hp=200,
            maxHp=200,
            playerIndex=1,
            serial=42,
        ),
        opponent_bench=[froslass, munkidori],
        options=[_attack(policy)],
    )

    assert obj._opp_has_froslass()
    assert obj._boss_role_bonus(froslass) > obj._boss_role_bonus(munkidori)
