"""v6-specific golden-state tests.

These run against the REAL vendor card database (vendor/cg) so they exercise the
true Ability text, ex/mega flags, weaknesses and evolution chains — the exact
data the submitted agent sees.  They cover the v6 changes:

  * generalised wall detection (Cornerstone Mask Ogerpon ex blocks Punk Up);
  * stop hammering a wall (0-damage Shadow Bullet ranks below END unless a Bench
    KO justifies it);
  * two-turn Boss's Orders value (protect a high-prize 2-attack KO route);
  * meta target priority (anti-Grimmsnarl tech -> main -> pre-evolution -> other);
  * bench-damage shields (Shaymin / Rabsca);
  * first_attacker_eta and the faster/safer opening.
"""
from __future__ import annotations

import importlib
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT.parents[2] / "vendor"


def load_main_real():
    """Import main.py against the real vendor cg so card_table is authoritative."""
    if not (VENDOR / "cg" / "cards.json").exists():
        pytest.skip("vendor/cg card database not available")
    for module in ("fallback_policy", "policy_base", "cg", "cg.api"):
        sys.modules.pop(module, None)
    for path in (str(VENDOR), str(ROOT)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    # The rule policy moved to fallback_policy.py; main.py is the ML wrapper.
    return importlib.import_module("fallback_policy")


def api():
    import cg.api as cg_api
    return cg_api


def poke(main, card_id, *, hp=None, energies=None):
    """Build a real cg.api.Pokemon for a card id, defaulting hp to its printed HP."""
    from policy_base import card_table
    data = card_table.get(card_id)
    printed = int(getattr(data, "hp", 0) or 0) if data is not None else 0
    P = api().Pokemon
    return P(
        id=card_id,
        hp=printed if hp is None else hp,
        maxHp=printed or (hp or 0),
        energies=list(energies or []),
    )


def bare_policy(main):
    policy = main.GrimmsnarlPolicy.__new__(main.GrimmsnarlPolicy)
    policy.hand = defaultdict(int)
    policy.field = defaultdict(int)
    policy.discard = defaultdict(int)
    policy.stadium_id = 0
    policy.effect_id = None
    return policy


def two_energy(main):
    D = api().EnergyType.DARKNESS
    return [D, D]


# ── generalised wall detection ───────────────────────────────────────────────
def test_ogerpon_ability_wall_is_detected():
    main = load_main_real()
    # Cornerstone Mask Ogerpon ex prevents all damage from Pokemon that have an
    # Ability; Grimmsnarl ex has Punk Up, so it walls us just like Crustle.
    assert main.C.CORNERSTONE_OGERPON in main.EX_ACTIVE_BLOCKERS
    assert main.C.CRUSTLE in main.EX_ACTIVE_BLOCKERS
    assert main.C.SYLVEON in main.EX_ACTIVE_BLOCKERS
    policy = bare_policy(main)
    assert policy.shadow_damage(poke(main, main.C.CORNERSTONE_OGERPON)) == 0


def test_farigiraf_and_milotic_are_not_walls_for_stage2_non_tera():
    main = load_main_real()
    # Farigiraf ex only stops *Basic* ex (we are Stage 2); Milotic ex only stops
    # *Tera* Pokemon (we are not Tera).  Neither should zero our Shadow Bullet.
    assert 83 not in main.EX_ACTIVE_BLOCKERS   # Farigiraf ex
    assert 207 not in main.EX_ACTIVE_BLOCKERS  # Milotic ex
    policy = bare_policy(main)
    assert policy.shadow_damage(poke(main, 83)) == 180


def test_ogerpon_active_marks_locked_state():
    main = load_main_real()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[poke(main, main.C.CORNERSTONE_OGERPON)], bench=[])
    assert policy.active_target_immune_to_ex()
    assert not policy.live_attack_ready()


# ── stop hammering a wall ────────────────────────────────────────────────────
def test_walled_shadow_ranks_below_end_without_bench_ko():
    main = load_main_real()
    policy = bare_policy(main)
    active = poke(main, main.C.GRIMMSNARL_EX, energies=two_energy(main))
    ogerpon = poke(main, main.C.CORNERSTONE_OGERPON)
    tall_bench = poke(main, main.C.ARCHALUDON_EX)  # 300 hp, Bench-30 cannot KO it
    policy.me = types.SimpleNamespace(active=[active], bench=[], prize=[0, 1, 2, 3, 4, 5])
    policy.opponent = types.SimpleNamespace(active=[ogerpon], bench=[tall_bench])
    policy.active_shadow_ready = lambda: True
    option = types.SimpleNamespace(attackId=main.A.SHADOW_BULLET, type=None)
    # 0-damage with no reachable Bench KO -> below END (base END scores 0).
    assert policy.score_attack(option) < 0


def test_walled_shadow_allowed_when_bench_ko_takes_a_prize():
    main = load_main_real()
    policy = bare_policy(main)
    active = poke(main, main.C.GRIMMSNARL_EX, energies=two_energy(main))
    crustle = poke(main, main.C.CRUSTLE)
    finish = poke(main, main.C.ABRA, hp=30)  # Bench-30 KOs this
    policy.me = types.SimpleNamespace(active=[active], bench=[], prize=[0, 1])
    policy.opponent = types.SimpleNamespace(active=[crustle], bench=[finish])
    policy.active_shadow_ready = lambda: True
    policy.powered_munkidori = lambda: False
    option = types.SimpleNamespace(attackId=main.A.SHADOW_BULLET, type=None)
    assert policy.score_attack(option) >= 700_000


# ── two-turn Boss's Orders value ─────────────────────────────────────────────
def test_boss_does_not_gust_chip_target_off_high_prize_two_attack_ko():
    main = load_main_real()
    policy = bare_policy(main)
    # Mega Lucario ex (340hp, 3 prizes): 180 now, 180 next turn = KO.  Do NOT Boss
    # a 1-prize Makuhita off that route (the exact v5 mistake).
    lucario = poke(main, main.C.MEGA_LUCARIO_EX)
    makuhita = poke(main, 673, hp=80)
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[lucario], bench=[makuhita])
    assert policy.best_boss_value() == -1


def test_boss_gusts_koable_engine_over_partial_active_route():
    main = load_main_real()
    policy = bare_policy(main)
    # Dragapult ex active is only a 2-attack KO in progress; a benched Fezandipiti
    # ex is a KO-able 2-prize engine now -> Boss grabs it.
    dragapult = poke(main, main.C.DRAGAPULT_EX)
    fez = poke(main, main.C.FEZANDIPITI_EX, hp=170)
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[dragapult], bench=[fez])
    assert policy.best_boss_value() > 0


def test_boss_unlocks_any_hittable_target_when_locked():
    main = load_main_real()
    policy = bare_policy(main)
    crustle = poke(main, main.C.CRUSTLE)
    survivor = poke(main, main.C.DRAGAPULT_EX)  # survives 180 but unlock still beats 0
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[crustle], bench=[survivor])
    assert policy.best_boss_value() > 0


# ── meta target priority ─────────────────────────────────────────────────────
def test_target_priority_ranks_tech_over_main_over_preevo_over_other():
    main = load_main_real()
    policy = bare_policy(main)
    policy.opponent = types.SimpleNamespace(active=[], bench=[])
    shaymin = policy.target_priority_bonus(poke(main, main.C.SHAYMIN))       # tech
    wall = policy.target_priority_bonus(poke(main, main.C.CRUSTLE))          # wall
    mega = policy.target_priority_bonus(poke(main, main.C.MEGA_LUCARIO_EX))  # main mega
    ex = policy.target_priority_bonus(poke(main, main.C.ARCHALUDON_EX))      # main ex
    engine = policy.target_priority_bonus(poke(main, main.C.DUDUNSPARCE))    # engine
    preevo = policy.target_priority_bonus(poke(main, main.C.ABRA))           # pre-evolution
    other = policy.target_priority_bonus(poke(main, 235))                    # Budew, vanilla
    assert shaymin > wall > mega > ex >= engine > preevo > other
    assert other == 0


def test_riolu_valued_as_mega_lucario_preevolution():
    main = load_main_real()
    policy = bare_policy(main)
    policy.opponent = types.SimpleNamespace(active=[], bench=[])
    # Riolu leads to Mega Lucario ex (3 prizes) so it outranks a dead-end basic.
    riolu = policy.target_priority_bonus(poke(main, 677))
    relicanth = policy.target_priority_bonus(poke(main, 57))
    assert riolu > relicanth


def test_preevolution_boosted_when_its_evolution_is_on_board():
    main = load_main_real()
    policy = bare_policy(main)
    # Abra on the bench while Alakazam is already Active: its line is live.
    policy.opponent = types.SimpleNamespace(active=[poke(main, main.C.ALAKAZAM)], bench=[poke(main, main.C.ABRA)])
    with_evo = policy.target_priority_bonus(poke(main, main.C.ABRA))
    policy.opponent = types.SimpleNamespace(active=[], bench=[poke(main, main.C.ABRA)])
    without_evo = policy.target_priority_bonus(poke(main, main.C.ABRA))
    assert with_evo > without_evo


# ── bench-damage shields ─────────────────────────────────────────────────────
def test_shaymin_blocks_non_rule_box_bench_damage():
    main = load_main_real()
    policy = bare_policy(main)
    policy.opponent = types.SimpleNamespace(active=[], bench=[poke(main, main.C.SHAYMIN)])
    assert not policy.bench_damage_lands(poke(main, main.C.ABRA))         # non-rule-box: blocked
    assert policy.bench_damage_lands(poke(main, main.C.ARCHALUDON_EX))    # rule-box: still lands


def test_rabsca_blocks_all_bench_damage():
    main = load_main_real()
    policy = bare_policy(main)
    policy.opponent = types.SimpleNamespace(active=[], bench=[poke(main, main.C.RABSCA)])
    assert not policy.bench_damage_lands(poke(main, main.C.ABRA))
    assert not policy.bench_damage_lands(poke(main, main.C.ARCHALUDON_EX))


# ── faster / safer opening ───────────────────────────────────────────────────
def test_first_attacker_eta_levels():
    main = load_main_real()
    D = api().EnergyType.DARKNESS

    def eta_for(setup):
        policy = bare_policy(main)
        me = types.SimpleNamespace(active=setup.get("active", []), bench=setup.get("bench", []), benchMax=5)
        policy.me = me
        policy.hand = defaultdict(int, setup.get("hand", {}))
        policy.field = defaultdict(int, setup.get("field", {}))
        return policy.first_attacker_eta()

    ready = poke(main, main.C.GRIMMSNARL_EX, energies=[D, D])
    assert eta_for({"active": [ready]}) == 0
    # Morgrem in play + Grimmsnarl in hand -> evolve + Punk Up -> attack this turn.
    assert eta_for({"bench": [poke(main, main.C.MORGREM)], "field": {main.C.MORGREM: 1},
                    "hand": {main.C.GRIMMSNARL_EX: 1}}) == 0
    # Impidimp in play + Grimmsnarl + Candy in hand -> Candy skips to Stage 2 and
    # Punk Up powers it, so it attacks THIS turn (eta 0, better than the rough table).
    assert eta_for({"bench": [poke(main, main.C.IMPIDIMP)], "field": {main.C.IMPIDIMP: 1},
                    "hand": {main.C.GRIMMSNARL_EX: 1, main.C.RARE_CANDY: 1}}) == 0
    # Impidimp in play + Morgrem + Grimmsnarl in hand but NO Candy -> evolve to
    # Morgrem this turn, complete next turn.
    assert eta_for({"bench": [poke(main, main.C.IMPIDIMP)], "field": {main.C.IMPIDIMP: 1},
                    "hand": {main.C.GRIMMSNARL_EX: 1, main.C.MORGREM: 1}}) == 1
    # Only a lone Impidimp, nothing else -> partial line.
    assert eta_for({"bench": [poke(main, main.C.IMPIDIMP)], "field": {main.C.IMPIDIMP: 1}}) == 2
    # Nothing at all.
    assert eta_for({}) == 99


def test_setup_active_order_impidimp_munkidori_snorunt():
    main = load_main_real()
    policy = bare_policy(main)
    policy.hand[main.C.IMPIDIMP] = 1
    policy.hand[main.C.DARKNESS] = 1  # v7: Munkidori needs an escape route to outrank Snorunt
    imp = policy.score_setup_active(poke(main, main.C.IMPIDIMP))
    munk = policy.score_setup_active(poke(main, main.C.MUNKIDORI))
    sno = policy.score_setup_active(poke(main, main.C.SNORUNT))
    assert imp > munk > sno


def test_lone_board_forces_a_basic():
    main = load_main_real()
    policy = bare_policy(main)
    policy.me = types.SimpleNamespace(active=[poke(main, main.C.MUNKIDORI)], bench=[], benchMax=5)
    policy.opponent = types.SimpleNamespace(active=[], bench=[])
    policy.field[main.C.MUNKIDORI] = 1
    # board_count == 1: developing an Impidimp must be near-top priority so a
    # single KO cannot wipe us (the Ceruledge board-wipe loss).
    assert policy.score_play_poke(poke(main, main.C.IMPIDIMP)) >= 940_000
