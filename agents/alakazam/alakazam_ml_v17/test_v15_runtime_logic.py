from __future__ import annotations

from pathlib import Path

import test_v11_runtime_logic as h


def test_v15_deck_is_byte_identical_to_v14():
    root = Path(__file__).resolve().parent
    assert (root / "deck.csv").read_bytes() == (
        root.parent / "alakazam_ml_v14" / "deck.csv"
    ).read_bytes()


def test_v15_rich_uses_hand_and_fuel_gate_but_never_blocks_active_attack():
    policy = h.load_policy()
    active = h.Pokemon(policy.C.ALAKAZAM)
    dunsparce = h.Pokemon(policy.C.DUNSPARCE)
    obj = h.bare_policy(policy, hand_count=5, active=active, bench=[dunsparce],
                        opp_active=h.Pokemon(9000, playerIndex=1))
    obj._deck_spend_ok = lambda *args, **kwargs: True
    obj._active_alakazam_can_be_fueled = lambda: False
    obj._need_p_fuel = lambda: False
    assert obj._enriching_attach_score(dunsparce) == 8450

    obj._active_alakazam_can_be_fueled = lambda: True
    assert obj._enriching_attach_score(dunsparce) < 0


def test_v15_rich_stops_when_dudunsparce_engine_and_large_hand_exist():
    policy = h.load_policy()
    dudunsparce = h.Pokemon(policy.C.DUDUNSPARCE)
    obj = h.bare_policy(policy, hand_count=8, bench=[dudunsparce],
                        opp_active=h.Pokemon(9000, playerIndex=1))
    obj._deck_spend_ok = lambda *args, **kwargs: True
    obj._active_alakazam_can_be_fueled = lambda: False
    assert obj._enriching_attach_score(dudunsparce) < 0


def test_v15_projected_active_ko_does_not_block_immediate_key_role_boss():
    policy = h.load_policy()
    mine = h.Pokemon(policy.C.ALAKAZAM, energies=[h.EnergyType.PSYCHIC])
    opposing_active = h.Pokemon(648, hp=150, maxHp=150, playerIndex=1)
    kadabra = h.Pokemon(policy.C.KADABRA, hp=100, maxHp=100, playerIndex=1)
    obj = h.bare_policy(policy, hand_count=8, active=mine,
                        opp_active=opposing_active, opp_bench=[kadabra])
    obj._boss_damage_after_spend = lambda target: 100
    obj._active_best_dmg = lambda target: 0
    obj._boss_active_reachable_damage = lambda target: 999
    obj._boss_resolving = lambda: False
    obj._articuno_escape_target = lambda target: False
    obj._boss_effect_lock_escape_ko = lambda target, damage=None: False

    assert obj._boss_target_score(kadabra) > 0


def test_v15_ml_scope_is_narrow_live_and_class_calibrated():
    source = (Path(__file__).resolve().parent / "ml_runtime.py").read_text(encoding="utf-8")
    main_source = (Path(__file__).resolve().parent / "main.py").read_text(encoding="utf-8")
    assert 'ML_ALLOWED_ACTIONS = {"bench", "evolve"}' in source
    assert 'ALAKAZAM_ML_V15_ENABLE_OVERRIDE", "1"' in source
    assert "predicted_action in ML_ALLOWED_ACTIONS" in source
    assert 'ALAKAZAM_ML_THRESHOLD", "0.37"' in main_source
