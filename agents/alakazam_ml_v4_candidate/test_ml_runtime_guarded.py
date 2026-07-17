from ml_runtime import (
    _candidate_scope_reason,
    _fallback_scope_reason,
)


def ctx(action, card=-1, *, breaks=False, lethal=False):
    return {
        "action_type": action,
        "card_id": card,
        "target_id": -1,
        "breaks_current_ko": breaks,
        "attack_lethal": lethal,
    }


def test_all_strategic_actions_are_rule_only():
    for action in (
        "ability", "end", "trainer", "energy", "boss",
        "retreat", "xerosic", "hammer", "other",
    ):
        assert _fallback_scope_reason(ctx(action)) == f"rule_only_{action}"


def test_safe_fallback_scopes_remain_ml_eligible():
    for action in ("bench", "evolve", "attack"):
        assert _fallback_scope_reason(ctx(action)) is None


def test_role_pokemon_benching_is_rule_only():
    fallback = ctx("bench", 741)
    assert _candidate_scope_reason(ctx("bench", 140), fallback) == "role_fezandipiti"
    assert _candidate_scope_reason(ctx("bench", 343), fallback) == "role_shaymin"
    assert _candidate_scope_reason(ctx("bench", 142), fallback) == "role_genesect"
    assert _candidate_scope_reason(ctx("bench", 858), fallback) == "role_psyduck"


def test_only_abra_and_dunsparce_are_ml_safe_bench_choices():
    fallback = ctx("bench", 741)
    assert _candidate_scope_reason(ctx("bench", 741), fallback) is None
    assert _candidate_scope_reason(ctx("bench", 305), fallback) is None
    assert _candidate_scope_reason(ctx("bench", 66), fallback) == "bench_not_allowlisted"


def test_fallback_attack_cannot_be_spent_on_development():
    fallback = ctx("attack")
    assert _candidate_scope_reason(ctx("bench", 741), fallback) == "preserve_fallback_attack"
    assert _candidate_scope_reason(ctx("evolve", 743), fallback) == "preserve_fallback_attack"
    assert _candidate_scope_reason(ctx("attack"), fallback) is None


def test_candidate_that_breaks_ko_is_blocked():
    fallback = ctx("evolve", 743)
    assert _candidate_scope_reason(ctx("evolve", 743, breaks=True), fallback) == "breaks_current_ko"
