from __future__ import annotations

from types import SimpleNamespace

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option


def _attacking_alakazam(policy, *, player_index=0, serial=10):
    return h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        playerIndex=player_index,
        serial=serial,
    )


def test_v25_froslass_outranks_powered_munkidori_on_grimmsnarl_board():
    policy = h.load_policy()
    froslass = h.Pokemon(
        policy.FROSLASS_ID,
        hp=90,
        maxHp=90,
        playerIndex=1,
        serial=20,
    )
    munkidori = h.Pokemon(
        112,
        hp=110,
        maxHp=110,
        energies=[h.EnergyType.DARKNESS],
        playerIndex=1,
        serial=21,
    )
    obj = h.bare_policy(
        policy,
        opp_active=h.Pokemon(
            policy.GRIMMSNARL_EX_ID,
            hp=320,
            maxHp=320,
            playerIndex=1,
            serial=22,
        ),
        opp_bench=[froslass, munkidori],
    )

    froslass_role = obj._boss_role_bonus(froslass)
    assert froslass_role > obj._boss_role_bonus(munkidori)
    # Reordering the targets must not create a giant MAIN-phase Boss bonus.
    assert froslass_role < 6000
    assert obj._target_priority_score(froslass) > obj._target_priority_score(
        munkidori
    )


def test_v25_mirror_fez_requires_real_flip_the_script_window():
    policy = h.load_policy()
    policy._V9_STATE["fez_recovery_turn"] = None
    obj = h.bare_policy(
        policy,
        active=_attacking_alakazam(policy),
        bench=[h.Pokemon(policy.C.DUNSPARCE, serial=31)],
        opp_active=_attacking_alakazam(
            policy,
            player_index=1,
            serial=32,
        ),
    )
    obj.state.turn = 8

    assert obj._opponent_is_alakazam_mirror()
    assert not obj._fez_bench_worthwhile()

    policy._V9_STATE["fez_recovery_turn"] = 8
    assert obj._fez_recovery_available_this_turn()
    assert obj._fez_bench_worthwhile()
    policy._V9_STATE["fez_recovery_turn"] = None


def test_v25_records_flip_the_script_window_after_ko_promotion():
    policy = h.load_policy()
    policy._V9_STATE["fez_recovery_turn"] = None
    obs = SimpleNamespace(
        current=SimpleNamespace(turn=11),
        select=SimpleNamespace(context=h.SelectContext.TO_ACTIVE),
    )

    policy._remember_fez_recovery_window(obs)

    assert policy._V9_STATE["fez_recovery_turn"] == 12
    policy._V9_STATE["fez_recovery_turn"] = None


def test_v25_alakazam_can_visibly_take_two_prizes_from_fez():
    policy = h.load_policy()
    opponent = _attacking_alakazam(
        policy,
        player_index=1,
        serial=40,
    )
    obj = h.bare_policy(policy, opp_active=opponent)
    obj.opponent.handCount = 11

    assert obj._opponent_can_ko_fez_next_turn()


def test_v25_unfuelled_kadabra_promotion_beats_dunsparce_shield():
    policy = h.load_policy()
    obj = h.bare_policy(
        policy,
        opp_active=h.Pokemon(9000, hp=200, playerIndex=1),
    )
    obj.context = h.SelectContext.TO_ACTIVE
    kadabra = h.Pokemon(policy.C.KADABRA, serial=50)
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, serial=51)

    assert not obj._promotion_attacks_next_turn(kadabra)
    assert obj._score_ko_promotion(kadabra) > obj._score_ko_promotion(
        dunsparce
    )


def test_v25_truly_naked_abra_still_stays_behind_dunsparce():
    policy = h.load_policy()
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=4,
        active=_attacking_alakazam(policy),
        opponent_active=h.Pokemon(9000, hp=200, playerIndex=1),
        options=[_option(h.OptionType.END)],
    )
    abra = h.Pokemon(policy.C.ABRA, serial=60)
    dunsparce = h.Pokemon(policy.C.DUNSPARCE, serial=61)

    assert obj._score_ko_promotion(dunsparce) > obj._score_ko_promotion(abra)
