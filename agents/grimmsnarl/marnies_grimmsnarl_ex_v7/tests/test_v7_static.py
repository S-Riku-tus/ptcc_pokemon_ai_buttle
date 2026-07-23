"""v7-specific golden-state tests (run against the REAL vendor card database).

They cover the v7 changes and, just as importantly, pin the v6 successes that
must NOT regress:

  * purpose-split Boss's Orders (WIN_NOW / WALL_UNLOCK / HIGHER_PRIZE_KO /
    ENGINE_KO / TEMPO_GUST) replacing the flat 10_000 gate — while the
    Makuhita-over-Mega-Lucario protection stays fixed;
  * legal-step first/backup attacker ETAs (appearThisTurn, evolution piece in
    hand, and a real path to the Active spot);
  * the fast-race gear and its optional-setup hold;
  * the conditional initial Active (a sole escape-less Munkidori drops below a
    Snorunt that has its Froslass route);
  * best-effort temporary (Dodge/Hide) immunity.
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
    if not (VENDOR / "cg" / "cards.json").exists():
        pytest.skip("vendor/cg card database not available")
    for module in ("main", "policy_base", "cg", "cg.api"):
        sys.modules.pop(module, None)
    for path in (str(VENDOR), str(ROOT)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    return importlib.import_module("main")


def api():
    import cg.api as cg_api
    return cg_api


def poke(main, card_id, *, hp=None, energies=None, appear=False, serial=None):
    from policy_base import card_table
    data = card_table.get(card_id)
    printed = int(getattr(data, "hp", 0) or 0) if data is not None else 0
    P = api().Pokemon
    return P(
        id=card_id,
        hp=printed if hp is None else hp,
        maxHp=printed or (hp or 0),
        energies=list(energies or []),
        appearThisTurn=appear,
        serial=serial,
    )


def bare_policy(main):
    policy = main.GrimmsnarlPolicy.__new__(main.GrimmsnarlPolicy)
    policy.hand = defaultdict(int)
    policy.field = defaultdict(int)
    policy.discard = defaultdict(int)
    policy.stadium_id = 0
    policy.effect_id = None
    policy._boss_mode = None
    return policy


def two_energy(main):
    D = api().EnergyType.DARKNESS
    return [D, D]


def set_opp(policy, active=None, bench=None):
    policy.opponent = types.SimpleNamespace(active=active or [], bench=bench or [])


# ── purpose-split Boss's Orders ──────────────────────────────────────────────
def test_boss_win_now_when_gust_takes_the_last_prize():
    main = load_main_real()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.me = types.SimpleNamespace(prize=[0])  # one prize card left
    tank = poke(main, main.C.ARCHALUDON_EX)        # Active survives 180
    kill = poke(main, main.C.FEZANDIPITI_EX, hp=120)  # 2-prize KO on the bench
    set_opp(policy, active=[tank], bench=[kill])
    value = policy.best_boss_value()
    assert policy._boss_mode == "WIN_NOW"
    assert value >= 5_000_000


def test_boss_engine_ko_fires_over_a_partial_active_route():
    main = load_main_real()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.me = types.SimpleNamespace(prize=[0, 1, 2, 3, 4, 5])
    dragapult = poke(main, main.C.DRAGAPULT_EX)          # only a 2-attack KO
    engine = poke(main, main.C.DUDUNSPARCE, hp=160)      # KO-able draw engine now
    set_opp(policy, active=[dragapult], bench=[engine])
    value = policy.best_boss_value()
    assert value > 0
    assert policy._boss_mode in ("ENGINE_KO", "HIGHER_PRIZE_KO")


def test_boss_tempo_gust_strands_unkoable_engine_when_active_route_is_weak():
    main = load_main_real()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.me = types.SimpleNamespace(prize=[0, 1, 2, 3, 4, 5])
    # No opposing Active route to protect; a benched Archaludon ex (300hp, retreat
    # 2) cannot be KO'd but can be stranded up to buy a turn.
    archaludon = poke(main, main.C.ARCHALUDON_EX)
    set_opp(policy, active=[], bench=[archaludon])
    value = policy.best_boss_value()
    assert value > 0
    assert policy._boss_mode == "TEMPO_GUST"


def test_boss_still_refuses_chip_gust_off_confirmed_high_prize_route():
    main = load_main_real()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.me = types.SimpleNamespace(prize=[0, 1, 2, 3, 4, 5])
    lucario = poke(main, main.C.MEGA_LUCARIO_EX)   # 3 prizes, 2-attack KO route
    makuhita = poke(main, 673, hp=80)              # 1-prize chip, not an engine
    set_opp(policy, active=[lucario], bench=[makuhita])
    assert policy.best_boss_value() == -1          # the v6 fix stays fixed


def test_boss_gate_dropped_engine_gust_now_playable_as_supporter():
    main = load_main_real()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.state = types.SimpleNamespace(supporterPlayed=False)
    policy.me = types.SimpleNamespace(prize=[0, 1, 2, 3, 4, 5], deckCount=30)
    policy.live_attack_ready = lambda: True
    dragapult = poke(main, main.C.DRAGAPULT_EX)
    engine = poke(main, main.C.DUDUNSPARCE, hp=160)
    set_opp(policy, active=[dragapult], bench=[engine])
    boss = poke(main, main.C.BOSSES_ORDERS)
    # v6 required best_boss_value >= 10_000 and this engine gust never cleared it;
    # v7 plays it as a normal-band supporter.
    assert policy.score_play_trainer(boss) > 800_000


# ── legal-step attacker ETAs ─────────────────────────────────────────────────
def test_first_eta_not_zero_when_morgrem_evolved_this_turn():
    main = load_main_real()
    policy = bare_policy(main)
    policy.state = types.SimpleNamespace(retreated=False, turn=4)
    fresh_morgrem = poke(main, main.C.MORGREM, appear=True)
    policy.me = types.SimpleNamespace(active=[poke(main, main.C.IMPIDIMP)],
                                      bench=[fresh_morgrem], benchMax=5)
    policy.field = defaultdict(int, {main.C.MORGREM: 1, main.C.IMPIDIMP: 1})
    policy.hand = defaultdict(int, {main.C.GRIMMSNARL_EX: 1})
    # v6 returned 0 (Morgrem in field + Grimmsnarl in hand); v7 knows the Morgrem
    # evolved this turn and cannot evolve again -> next turn.
    assert policy.first_attacker_eta() == 1


def test_first_eta_not_zero_when_ready_grimm_is_trapped_on_bench():
    main = load_main_real()
    policy = bare_policy(main)
    policy.state = types.SimpleNamespace(retreated=False, turn=6)
    trapped_active = poke(main, main.C.MUNKIDORI, energies=[])  # cannot pay retreat
    ready = poke(main, main.C.GRIMMSNARL_EX, energies=two_energy(main))
    policy.me = types.SimpleNamespace(active=[trapped_active], bench=[ready], benchMax=5)
    policy.field = defaultdict(int, {main.C.MUNKIDORI: 1, main.C.GRIMMSNARL_EX: 1})
    policy.hand = defaultdict(int)
    assert policy.first_attacker_eta() == 1  # ready but no legal path to the Active this turn


def test_first_eta_zero_when_ready_grimm_can_be_promoted():
    main = load_main_real()
    policy = bare_policy(main)
    D = api().EnergyType.DARKNESS
    policy.state = types.SimpleNamespace(retreated=False, turn=6)
    active = poke(main, main.C.MUNKIDORI, energies=[D])  # 1 energy pays retreat cost 1
    ready = poke(main, main.C.GRIMMSNARL_EX, energies=two_energy(main))
    policy.me = types.SimpleNamespace(active=[active], bench=[ready], benchMax=5)
    policy.field = defaultdict(int, {main.C.MUNKIDORI: 1, main.C.GRIMMSNARL_EX: 1})
    policy.hand = defaultdict(int)
    assert policy.first_attacker_eta() == 0


def test_backup_eta_rejects_lone_morgrem_without_grimmsnarl_in_hand():
    main = load_main_real()
    policy = bare_policy(main)
    policy.state = types.SimpleNamespace(retreated=False, turn=6)
    live = poke(main, main.C.GRIMMSNARL_EX, energies=two_energy(main))
    morgrem = poke(main, main.C.MORGREM, energies=two_energy(main))
    policy.me = types.SimpleNamespace(active=[live], bench=[morgrem], benchMax=5)
    policy.field = defaultdict(int, {main.C.GRIMMSNARL_EX: 1, main.C.MORGREM: 1})
    policy.hand = defaultdict(int)          # no Grimmsnarl ex in hand
    policy.fast_race = lambda: False
    assert policy.backup_attacker_eta() >= 2   # v6 scored this as ETA 0
    assert not policy.backup_is_close()


def test_backup_eta_close_with_morgrem_and_grimmsnarl_in_hand():
    main = load_main_real()
    policy = bare_policy(main)
    policy.state = types.SimpleNamespace(retreated=False, turn=6)
    live = poke(main, main.C.GRIMMSNARL_EX, energies=two_energy(main))
    morgrem = poke(main, main.C.MORGREM, energies=two_energy(main))
    policy.me = types.SimpleNamespace(active=[live], bench=[morgrem], benchMax=5)
    policy.field = defaultdict(int, {main.C.GRIMMSNARL_EX: 1, main.C.MORGREM: 1})
    policy.hand = defaultdict(int, {main.C.GRIMMSNARL_EX: 1})
    policy.fast_race = lambda: False
    assert policy.backup_attacker_eta() == 1


# ── fast race ────────────────────────────────────────────────────────────────
def test_fast_race_detected_from_mega_lucario_and_its_preevo():
    main = load_main_real()
    policy = bare_policy(main)
    set_opp(policy, active=[poke(main, main.C.MEGA_LUCARIO_EX)], bench=[])
    assert policy.fast_race()
    policy2 = bare_policy(main)
    set_opp(policy2, active=[poke(main, 677)], bench=[])   # Riolu -> Mega Lucario
    assert policy2.fast_race()
    policy3 = bare_policy(main)
    set_opp(policy3, active=[poke(main, main.C.MUNKIDORI)], bench=[])
    assert not policy3.fast_race()


def test_fast_race_holds_optional_setup_until_attacker_is_live():
    main = load_main_real()
    policy = bare_policy(main)
    policy.state = types.SimpleNamespace(retreated=False, turn=2)
    set_opp(policy, active=[poke(main, main.C.MEGA_LUCARIO_EX)], bench=[])
    policy.me = types.SimpleNamespace(active=[poke(main, main.C.IMPIDIMP)], bench=[], benchMax=5)
    policy.field = defaultdict(int, {main.C.IMPIDIMP: 1})
    policy.hand = defaultdict(int)
    policy.ready_grimms = lambda: []
    assert policy.hold_optional_setup()


# ── conditional initial Active ───────────────────────────────────────────────
def test_sole_munkidori_without_escape_ranks_below_snorunt():
    main = load_main_real()
    policy = bare_policy(main)
    policy.hand[main.C.MUNKIDORI] = 1     # our only Munkidori
    policy.hand[main.C.DARKNESS] = 0      # no energy to pay its retreat
    munk = policy.score_setup_active(poke(main, main.C.MUNKIDORI))
    sno = policy.score_setup_active(poke(main, main.C.SNORUNT))
    assert munk < sno
    # give it an escape and it climbs back above Snorunt
    policy.hand[main.C.DARKNESS] = 1
    assert policy.score_setup_active(poke(main, main.C.MUNKIDORI)) > sno


# ── temporary immunity (best effort) ─────────────────────────────────────────
def test_temp_immunity_attacks_include_coin_dodges():
    main = load_main_real()
    # "Hide" (684) and "Dig" (75) flip a coin to prevent all damage next turn.
    assert 684 in main.TEMP_IMMUNITY_COIN
    assert 75 in main.TEMP_IMMUNITY_COIN


def test_temp_immunity_zeroes_shadow_bullet_for_the_turn():
    main = load_main_real()
    policy = bare_policy(main)
    policy.state = types.SimpleNamespace(turn=7)
    dodged = poke(main, main.C.DRAGAPULT_EX, serial=99999)
    main.TEMP_IMMUNITY[99999] = 7          # immune through turn 7
    try:
        assert policy.temp_immune(dodged)
        assert policy.shadow_damage(dodged) == 0
        assert not policy.bench_damage_lands(dodged)
        policy.state = types.SimpleNamespace(turn=8)   # immunity lapsed
        assert not policy.temp_immune(dodged)
        assert policy.shadow_damage(dodged) == 180
    finally:
        main.TEMP_IMMUNITY.pop(99999, None)
