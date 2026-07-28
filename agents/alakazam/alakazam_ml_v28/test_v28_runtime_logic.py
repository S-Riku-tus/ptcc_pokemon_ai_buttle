"""Regressions for v28 end-turn exposure and metagame target priorities."""

from __future__ import annotations

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option
from test_v24_runtime_logic import _attacking_alakazam
from test_v26_runtime_logic import _install_v26_cards


def _switch_attack_board(policy, *, active, bench, attack_id):
    attack = _option(h.OptionType.ATTACK, attack_id=attack_id)
    end = _option(h.OptionType.END)
    return _main_board(
        policy,
        hand_ids=[],
        hand_count=8,
        active=active,
        bench=bench,
        opponent_active=h.Pokemon(
            9000, hp=200, maxHp=200, playerIndex=1, serial=90
        ),
        options=[attack, end],
    )


def test_v28_dunsparce_stays_active_in_front_of_benched_abra():
    policy = h.load_policy()
    obj = _switch_attack_board(
        policy,
        active=h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70, serial=10),
        bench=[h.Pokemon(policy.C.ABRA, hp=50, maxHp=50, serial=11)],
        attack_id=policy.DUNSPARCE_TRADE,
    )

    assert obj._score(obj.select.option[0]) < obj._score(obj.select.option[1])
    assert obj.choose() == [1]


def test_v28_dunsparce_does_not_expose_a_ready_alakazam():
    policy = h.load_policy()
    obj = _switch_attack_board(
        policy,
        active=h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70, serial=20),
        bench=[_attacking_alakazam(policy)],
        attack_id=policy.DUNSPARCE_TRADE,
    )

    assert obj._score(obj.select.option[0]) == -1
    assert obj.choose() == [1]


def test_v28_abra_teleports_into_dunsparce_to_protect_the_line():
    policy = h.load_policy()
    obj = _switch_attack_board(
        policy,
        active=h.Pokemon(policy.C.ABRA, hp=50, maxHp=50, serial=30),
        bench=[h.Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70, serial=31)],
        attack_id=policy.ABRA_TELEPORT,
    )

    assert obj._score(obj.select.option[0]) > obj._score(obj.select.option[1])
    assert obj.choose() == [0]


def test_v28_gusts_koable_duraludon_over_active_cinderace():
    policy = _install_v26_cards(h.load_policy())
    policy.card_table[policy.DURALUDON_ID] = h.CardData(
        policy.DURALUDON_ID, name="Duraludon", attacks=[901]
    )
    policy.card_table[policy.ARCHALUDON_EX_ID] = h.CardData(
        policy.ARCHALUDON_EX_ID,
        name="Archaludon ex",
        stage1=True,
        ex=True,
        attacks=[902],
    )
    policy.card_table[policy.CINDERACE_ID].ex = False
    policy.EVOLVES_FROM_INDEX[policy.DURALUDON_ID] = {
        policy.ARCHALUDON_EX_ID
    }

    cinderace = h.Pokemon(
        policy.CINDERACE_ID,
        hp=100,
        maxHp=130,
        playerIndex=1,
        serial=40,
    )
    duraludon = h.Pokemon(
        policy.DURALUDON_ID,
        hp=130,
        maxHp=130,
        playerIndex=1,
        serial=41,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=8,
        active=_attacking_alakazam(policy),
        opponent_active=cinderace,
        opponent_bench=[duraludon],
        options=[
            _option(h.OptionType.PLAY, index=0),
            _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND),
        ],
    )

    assert obj._active_best_dmg(cinderace) >= cinderace.hp
    assert obj._boss_damage_after_spend(duraludon) >= duraludon.hp
    assert obj._boss_role_bonus(duraludon) >= (
        obj._boss_role_bonus(cinderace) + 1500
    )
    assert obj._boss_target_score(duraludon) > 0
    assert obj.choose() == [0]
