"""The damage columns must agree with the card rules, not with 180 flat.

v1 and the first cut of v2 computed `attack_kills_active` as `0 < hp <= 180`
with no immunity, weakness or Bench-shield term. Against an Active Crustle the
model therefore read "this swing knocks it out" while the real damage was 0,
and the ladder logs show 53 such swings across 61 games. These tests pin the
corrected resolver so the columns cannot silently regress to the printed
number again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_features as F  # noqa: E402

CRUSTLE = 345
SYLVEON = 330
CORNERSTONE_OGERPON = 117
RABSCA = 74
SHAYMIN = 343
MUNKIDORI = 112          # Darkness-weak in the current pool
IMPIDIMP = 646
GRIMMSNARL_EX = 648


def body(card_id: int, hp: int, max_hp: int | None = None) -> dict:
    return {
        "id": card_id, "hp": hp, "maxHp": max_hp if max_hp else hp,
        "energies": [], "energyCards": [], "tools": [], "preEvolution": [],
    }


def board(
    opp_active: dict,
    opp_bench: list[dict] | None = None,
    stadium: int = -1,
    self_active: dict | None = None,
) -> dict:
    grimmsnarl = self_active or body(GRIMMSNARL_EX, 340)
    grimmsnarl["energyCards"] = [{"id": F.DARK_ENERGY_ID}] * 2
    me = {
        "active": [grimmsnarl], "bench": [], "hand": [], "discard": [],
        "prize": [{}] * 6, "deckCount": 40, "handCount": 0, "benchMax": 5,
    }
    opp = {
        "active": [opp_active], "bench": list(opp_bench or []), "hand": [],
        "discard": [], "prize": [{}] * 6, "deckCount": 40, "handCount": 5,
        "benchMax": 5,
    }
    current = {
        "players": [me, opp], "yourIndex": 0, "turn": 5,
        "stadium": [{"id": stadium}] if stadium >= 0 else [],
    }
    return current


# ----- the Active spot ------------------------------------------------------

@pytest.mark.parametrize("wall", [CRUSTLE, SYLVEON, CORNERSTONE_OGERPON])
def test_wall_active_takes_nothing(wall: int) -> None:
    out = F.state_features(board(body(wall, 120)))
    assert out["opp_active_shadow_damage"] == 0.0
    assert out["opp_active_is_damage_immune"] == 1
    # The defect: 120 <= 180 used to read as a knockout.
    assert out["shadow_bullet_kills_active"] == 0
    assert out["shadow_bullet_is_dead_swing"] == 1
    assert out["shadow_bullet_prizes_now"] == 0


def test_normal_active_still_dies() -> None:
    out = F.state_features(board(body(IMPIDIMP, 70)))
    assert out["opp_active_shadow_damage"] == 180.0
    assert out["opp_active_is_damage_immune"] == 0
    assert out["shadow_bullet_kills_active"] == 1
    assert out["shadow_bullet_is_dead_swing"] == 0


def test_darkness_weakness_doubles_the_swing() -> None:
    # 300 HP survives 180 but not 360, and the old flat model said it lived.
    out = F.state_features(board(body(MUNKIDORI, 300, 300)))
    assert out["opp_active_shadow_damage"] == 360.0
    assert out["shadow_bullet_kills_active"] == 1


def test_neutralization_zone_zeroes_non_rule_box() -> None:
    zone = F.NEUTRALIZATION_ZONE_ID
    out = F.state_features(board(body(IMPIDIMP, 70), stadium=zone))
    assert out["opp_active_shadow_damage"] == 0.0
    assert out["shadow_bullet_kills_active"] == 0
    # A Rule Box body keeps taking damage under the same stadium.
    out = F.state_features(board(body(GRIMMSNARL_EX, 170), stadium=zone))
    assert out["opp_active_shadow_damage"] == 180.0
    assert out["shadow_bullet_kills_active"] == 1


# ----- the Bench-30 ---------------------------------------------------------

def test_bench_snipe_blocked_by_wall_body() -> None:
    out = F.state_features(
        board(body(CRUSTLE, 120), [body(CRUSTLE, 20), body(IMPIDIMP, 20)])
    )
    assert out["opp_bench_snipe_target_count"] == 1
    assert out["opp_bench_snipe_kill_count"] == 1
    assert out["opp_bench_wall_count"] == 1
    # A prize is still on offer off the Bench, so the swing is not dead.
    assert out["shadow_bullet_is_dead_swing"] == 0
    assert out["shadow_bullet_prizes_now"] == 1


def test_rabsca_shields_the_whole_bench() -> None:
    out = F.state_features(
        board(body(CRUSTLE, 120), [body(RABSCA, 90), body(IMPIDIMP, 20)])
    )
    assert out["opp_bench_snipe_target_count"] == 0
    assert out["opp_bench_fully_shielded"] == 1
    assert out["shadow_bullet_is_worthless"] == 1
    assert out["shadow_bullet_prizes_now"] == 0


def test_shaymin_shields_only_non_rule_box() -> None:
    out = F.state_features(board(
        body(IMPIDIMP, 200),
        [body(SHAYMIN, 70), body(IMPIDIMP, 20), body(GRIMMSNARL_EX, 20)],
    ))
    # Shaymin and Impidimp are covered; the ex on the bench is not.
    assert out["opp_bench_snipe_target_count"] == 1
    assert out["opp_bench_snipe_kill_count"] == 1
    assert out["opp_has_bench_shield"] == 1


def test_dead_swing_needs_both_halves_dead() -> None:
    # Wall Active, bench body present but out of snipe range: no prize now.
    out = F.state_features(board(body(CRUSTLE, 120), [body(IMPIDIMP, 70)]))
    assert out["shadow_bullet_is_dead_swing"] == 1
    assert out["shadow_bullet_is_worthless"] == 0  # the 30 still chips
    assert out["opp_bench_snipe_target_count"] == 1


# ----- Boss's Orders, by purpose --------------------------------------------

def test_boss_unlocks_a_walled_board() -> None:
    out = F.state_features(board(body(CRUSTLE, 120), [body(IMPIDIMP, 70)]))
    assert out["boss_gust_kill_count"] == 1
    assert out["boss_unlocks_wall"] == 1
    assert out["boss_kill_ready"] == 1


def test_boss_unlock_does_not_require_an_immediate_ko() -> None:
    out = F.state_features(board(body(CRUSTLE, 120), [body(IMPIDIMP, 200)]))
    assert out["boss_gust_hittable_count"] == 1
    assert out["boss_gust_kill_count"] == 0
    assert out["boss_unlocks_wall"] == 1
    assert out["boss_kill_ready"] == 0


def test_boss_gust_prize_value_prefers_the_ex() -> None:
    out = F.state_features(board(
        body(IMPIDIMP, 200), [body(IMPIDIMP, 70), body(GRIMMSNARL_EX, 100)]
    ))
    assert out["boss_gust_kill_count"] == 2
    assert out["boss_gust_best_prize"] == 2


# ----- the option row reads the same numbers --------------------------------

def _attack_option() -> dict:
    return {"type": 13, "index": 0, "attackId": F.SHADOW_BULLET_ID}


def test_attack_option_sees_the_wall() -> None:
    current = board(body(CRUSTLE, 120))
    select = {"context": 0, "type": 0, "minCount": 1, "maxCount": 1,
              "option": [_attack_option(), {"type": 14}]}
    row = F.option_features(current, select, _attack_option())
    assert row["attack_is_shadow_bullet"] == 1
    assert row["attack_effective_damage"] == 0.0
    assert row["attack_kills_active"] == 0
    assert row["attack_into_damage_immune"] == 1
    assert row["attack_is_dead_swing"] == 1
    assert row["attack_prizes_now"] == 0


def test_attack_option_on_a_live_target() -> None:
    current = board(body(IMPIDIMP, 70))
    select = {"context": 0, "type": 0, "minCount": 1, "maxCount": 1,
              "option": [_attack_option(), {"type": 14}]}
    row = F.option_features(current, select, _attack_option())
    assert row["attack_effective_damage"] == 180.0
    assert row["attack_kills_active"] == 1
    assert row["attack_is_dead_swing"] == 0
    assert row["attack_prizes_now"] == 1


# ----- the snipe-target context ---------------------------------------------

def test_snipe_target_context_knows_the_swing_does_not_land() -> None:
    """Context 15 picks the Bench-30 target: a benched wall takes nothing."""
    current = board(body(IMPIDIMP, 200), [body(CRUSTLE, 20), body(IMPIDIMP, 20)])
    select = {"context": F.CTX_DAMAGE, "type": 3, "minCount": 1,
              "maxCount": 1, "option": []}
    options = [
        {"type": 3, "area": F.AREA_BENCH, "index": 0, "playerIndex": 1},
        {"type": 3, "area": F.AREA_BENCH, "index": 1, "playerIndex": 1},
    ]
    select["option"] = options
    walled = F.option_features(current, select, options[0])
    live = F.option_features(current, select, options[1])
    assert walled["ctx_swing_lands"] == 0
    assert walled["ctx_target_dies_to_swing"] == 0
    assert walled["ctx_target_is_damage_immune"] == 1
    assert live["ctx_swing_lands"] == 1
    assert live["ctx_target_dies_to_swing"] == 1


def test_damage_counters_ignore_the_wall() -> None:
    """Adrena-Brain places counters, which damage prevention does not stop."""
    current = board(body(IMPIDIMP, 200), [body(CRUSTLE, 20)])
    select = {"context": F.CTX_DAMAGE_COUNTER, "type": 3, "minCount": 1,
              "maxCount": 1, "remainDamageCounter": 3, "option": []}
    option = {"type": 3, "area": F.AREA_BENCH, "index": 0, "playerIndex": 1}
    select["option"] = [option]
    row = F.option_features(current, select, option)
    assert row["ctx_swing_lands"] == 1
    assert row["ctx_target_dies_to_swing"] == 1


def test_battle_cage_stops_counters_on_the_bench() -> None:
    current = board(
        body(IMPIDIMP, 200), [body(IMPIDIMP, 20)],
        stadium=F.BATTLE_CAGE_ID,
    )
    select = {"context": F.CTX_DAMAGE_COUNTER, "type": 3, "minCount": 1,
              "maxCount": 1, "remainDamageCounter": 3, "option": []}
    option = {"type": 3, "area": F.AREA_BENCH, "index": 0, "playerIndex": 1}
    select["option"] = [option]
    row = F.option_features(current, select, option)
    assert row["ctx_swing_lands"] == 0
    assert row["ctx_target_dies_to_swing"] == 0
    assert row["counters_land_on_bench"] == 0


def test_battle_cage_does_not_stop_adrena_brain_removal() -> None:
    source = body(MUNKIDORI, 70, 110)
    current = board(body(IMPIDIMP, 200), stadium=F.BATTLE_CAGE_ID)
    current["players"][0]["bench"] = [source]
    select = {
        "context": F.CTX_REMOVE_DAMAGE_COUNTER,
        "type": 3,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 3,
        "option": [],
    }
    option = {
        "type": 3,
        "area": F.AREA_BENCH,
        "index": 0,
        "playerIndex": 0,
    }
    select["option"] = [option]
    row = F.option_features(current, select, option)
    assert row["ctx_swing_lands"] == 1
    assert row["ctx_is_counter_source"] == 1
    assert row["ctx_removable_damage"] == 30.0
    assert row["ctx_target_dies_to_swing"] == 0
    assert row["ctx_target_hp_after_swing"] == 100.0


def test_neutralization_zone_blocks_active_damage_context() -> None:
    current = board(
        body(IMPIDIMP, 20), stadium=F.NEUTRALIZATION_ZONE_ID
    )
    select = {
        "context": F.CTX_DAMAGE,
        "type": 3,
        "minCount": 1,
        "maxCount": 1,
        "option": [],
    }
    option = {
        "type": 3,
        "area": F.AREA_ACTIVE,
        "index": 0,
        "playerIndex": 1,
    }
    select["option"] = [option]
    row = F.option_features(current, select, option)
    assert row["ctx_swing_lands"] == 0
    assert row["ctx_target_dies_to_swing"] == 0


# ----- the tables themselves ------------------------------------------------

def test_generated_tables_are_populated() -> None:
    """A regenerated-but-empty table would silently disable every check."""
    assert {CRUSTLE, SYLVEON, CORNERSTONE_OGERPON} <= F.EX_DAMAGE_BLOCKER_IDS
    assert GRIMMSNARL_EX in F.RULE_BOX_IDS
    assert len(F.RULE_BOX_IDS) > 100
    assert len(F.DARK_WEAK_IDS) > 50
    assert F.prize_value(IMPIDIMP) == 1
    assert F.prize_value(GRIMMSNARL_EX) == 2
    assert all(F.prize_value(c) == 3 for c in list(F.MEGA_EX_IDS)[:5])


def test_no_leakage_from_the_new_columns() -> None:
    out = F.state_features(board(body(CRUSTLE, 120)))
    F.assert_no_leakage(sorted(out))
