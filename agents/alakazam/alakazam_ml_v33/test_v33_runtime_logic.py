"""Regressions for the measured v33 rebuild and hand-as-damage rules."""

from __future__ import annotations

import importlib

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option
from test_v24_runtime_logic import _attack, _attacking_alakazam
from test_v26_runtime_logic import _install_v26_cards


def _dudun_board(policy, *, hand_ids, hand_count, target_hp):
    dudunsparce = h.Pokemon(
        policy.C.DUDUNSPARCE,
        hp=140,
        maxHp=140,
        serial=11,
    )
    abra = h.Pokemon(
        policy.C.ABRA,
        hp=50,
        maxHp=50,
        energies=[h.EnergyType.PSYCHIC],
        serial=12,
    )
    ability = _option(
        h.OptionType.ABILITY,
        index=0,
        area=h.AreaType.BENCH,
    )
    obj = _main_board(
        policy,
        hand_ids=hand_ids,
        hand_count=hand_count,
        active=_attacking_alakazam(policy),
        bench=[dudunsparce, abra],
        opponent_active=h.Pokemon(
            9000,
            hp=target_hp,
            maxHp=target_hp,
            playerIndex=1,
            serial=20,
        ),
        options=[ability, _attack(policy)],
    )
    return obj, ability


def test_v33_cycles_even_when_backup_eta_is_one():
    policy = _install_v26_cards(h.load_policy())
    obj, ability = _dudun_board(
        policy,
        hand_ids=[policy.C.RARE_CANDY, policy.C.ALAKAZAM],
        hand_count=6,
        target_hp=300,
    )

    assert obj._backup_eta() == 1
    assert not obj._dudun_draw_satisfied()
    assert obj._continuity_draw_needed()
    assert obj._score_ability(ability) > 0
    assert obj.choose() == [0]


def test_v33_last_opponent_prize_does_not_block_nonlethal_cycle():
    policy = _install_v26_cards(h.load_policy())
    obj, ability = _dudun_board(
        policy,
        hand_ids=[],
        hand_count=6,
        target_hp=300,
    )
    obj.opponent.prize = [h.Card(9100)]

    assert len(obj.opponent.prize) == 1
    assert obj._continuity_draw_needed()
    assert obj._score_ability(ability) > 0


def test_v33_exact_powerful_hand_ko_remains_the_draw_stop():
    policy = _install_v26_cards(h.load_policy())
    obj, ability = _dudun_board(
        policy,
        hand_ids=[],
        hand_count=10,
        target_hp=200,
    )

    assert obj._dudun_draw_satisfied()
    assert not obj._continuity_draw_needed()
    assert obj._score_ability(ability) < 0
    assert obj.choose() == [1]


def test_v33_g_state_prioritizes_search_cards():
    policy = _install_v26_cards(h.load_policy())
    hilda = _option(h.OptionType.PLAY, index=0)
    dawn = _option(h.OptionType.PLAY, index=1)
    poke_pad = _option(h.OptionType.PLAY, index=2)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.HILDA, policy.C.DAWN, policy.C.POKE_PAD],
        hand_count=7,
        active=h.Pokemon(policy.C.DUNSPARCE, serial=10),
        bench=[h.Pokemon(policy.C.KADABRA, serial=11)],
        opponent_active=h.Pokemon(
            9000,
            hp=200,
            maxHp=200,
            playerIndex=1,
            serial=20,
        ),
        options=[hilda, dawn, poke_pad, _option(h.OptionType.END)],
    )

    assert obj._rebuild_search_mode()
    assert obj._score_play(dawn) >= 25000
    assert obj._score_play(hilda) >= 25500
    assert obj._score_play(poke_pad) >= 24000
    assert obj.choose() == [0]


def test_v33_f_state_uses_poffin_to_create_an_evolution_body():
    policy = _install_v26_cards(h.load_policy())
    hilda = _option(h.OptionType.PLAY, index=0)
    poffin = _option(h.OptionType.PLAY, index=1)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.HILDA, policy.C.BUDDY_POFFIN, policy.C.ALAKAZAM],
        hand_count=7,
        active=h.Pokemon(policy.C.FEZANDIPITI_EX, serial=10),
        opponent_active=h.Pokemon(
            9000,
            hp=200,
            maxHp=200,
            playerIndex=1,
            serial=20,
        ),
        options=[hilda, poffin, _option(h.OptionType.END)],
    )

    assert obj._rebuild_search_mode()
    assert not obj._evolution_target_on_board()
    assert obj._score_play(poffin) > obj._score_play(hilda)
    assert obj.choose() == [1]


def test_v33_poffin_still_builds_width_from_four_core_bodies():
    policy = _install_v26_cards(h.load_policy())
    poffin = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BUDDY_POFFIN],
        hand_count=8,
        active=_attacking_alakazam(policy),
        bench=[
            h.Pokemon(policy.C.ABRA, serial=11),
            h.Pokemon(policy.C.DUNSPARCE, serial=12),
            h.Pokemon(policy.C.DUDUNSPARCE, serial=13),
        ],
        opponent_active=h.Pokemon(
            9000,
            hp=300,
            maxHp=300,
            playerIndex=1,
            serial=20,
        ),
        options=[poffin, _option(h.OptionType.END)],
    )

    assert obj._board_body_count() == 4
    assert obj._score_play(poffin) == 16000
    assert obj.choose() == [0]


def test_v33_compact_tree_matches_lightgbm_missing_category_rule():
    h.load_policy()
    runtime = importlib.import_module("ml_runtime")
    model = {
        "trees": [{
            "f": 0,
            "d": "==",
            "c": [7],
            "x": True,
            "l": {"v": 2.0},
            "r": {"v": -1.0},
        }],
    }

    assert runtime._selector_tree_score([-1.0], model) == 2.0
    assert runtime._selector_tree_score([7.0], model) == 2.0
    assert runtime._selector_tree_score([8.0], model) == -1.0


def test_v33_does_not_treat_shaymin_as_froslass_protection():
    policy = _install_v26_cards(h.load_policy())
    active = _attacking_alakazam(policy)
    exposed = h.Pokemon(policy.C.DUNSPARCE, hp=30, maxHp=70)
    second = h.Pokemon(policy.C.KADABRA, hp=80, maxHp=80)
    grimmsnarl = h.Pokemon(
        policy.GRIMMSNARL_EX_ID,
        playerIndex=1,
    )
    froslass = h.Pokemon(
        policy.FROSLASS_ID,
        playerIndex=1,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.SHAYMIN],
        hand_count=8,
        active=active,
        bench=[exposed, second],
        opponent_active=grimmsnarl,
        opponent_bench=[froslass],
        options=[_option(h.OptionType.PLAY, index=0)],
    )

    assert obj._opp_threatens_bench()
    assert obj._opp_has_froslass()
    assert obj._score_play_poke(h.Card(policy.C.SHAYMIN)) < 0
