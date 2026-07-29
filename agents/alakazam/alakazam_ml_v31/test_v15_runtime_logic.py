from __future__ import annotations

from pathlib import Path

import test_v11_runtime_logic as h


def test_v15_deck_is_byte_identical_to_v14():
    import hashlib

    root = Path(__file__).resolve().parent
    assert hashlib.sha256((root / "deck.csv").read_bytes()).hexdigest() == (
        "57c7d4800cfc0f36581077a40b24912d33056cafcc14cca3783094ce6c122bfe"
    )


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


def test_v31_ml_scope_is_memory_first_v29_residual_and_safety_guarded():
    source = (Path(__file__).resolve().parent / "ml_runtime.py").read_text(encoding="utf-8")
    main_source = (Path(__file__).resolve().parent / "main.py").read_text(encoding="utf-8")
    assert 'ALAKAZAM_ML_V31_ENABLE_OVERRIDE", "1"' in source
    assert "teacher_memory_keys" in source
    assert "v29_ranker_score" in source
    assert "fallback_selected" in source
    assert "legacy_ranker_score" in source
    assert "lethal_guard" in source
    assert "_candidate_safety_reason" in source
    assert "_V29_RUNTIME" in main_source
    assert 'ALAKAZAM_ML_V31_THRESHOLD", "0.0"' in main_source
