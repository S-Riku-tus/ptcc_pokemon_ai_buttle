from __future__ import annotations

import csv
import importlib
import sys
import types
from collections import defaultdict
from enum import IntEnum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AreaType(IntEnum):
    DECK = 1; HAND = 2; DISCARD = 3; ACTIVE = 4; BENCH = 5; PRIZE = 6; STADIUM = 7; LOOKING = 8


class CardType(IntEnum):
    POKEMON = 1; BASIC_ENERGY = 2; SPECIAL_ENERGY = 3; TRAINER = 4


class EnergyType(IntEnum):
    COLORLESS = 0; GRASS = 1; FIRE = 2; WATER = 3; LIGHTNING = 4; PSYCHIC = 5; FIGHTING = 6; DARKNESS = 7


class OptionType(IntEnum):
    YES = 0; NO = 1; NUMBER = 2; CARD = 3; PLAY = 4; ENERGY = 5; ATTACH = 6; EVOLVE = 7; ABILITY = 8; RETREAT = 9; ATTACK = 10; END = 11


class SelectContext(IntEnum):
    MAIN = 0; SETUP_ACTIVE_POKEMON = 1; SETUP_BENCH_POKEMON = 2; SWITCH = 3; TO_ACTIVE = 4; TO_BENCH = 5; TO_FIELD = 6; TO_HAND = 7; ATTACH_TO = 8; ATTACH_FROM = 9; REMOVE_DAMAGE_COUNTER = 10; DAMAGE_COUNTER = 11; DAMAGE_COUNTER_ANY = 12; DAMAGE = 13; IS_FIRST = 14; MULLIGAN = 15; PRIZE = 16; EVOLVES_TO = 17; EVOLVES_FROM = 18; TO_HAND_ENERGY = 19; DISCARD = 20; DISCARD_CARD_OR_ATTACHED_CARD = 21; DISCARD_ENERGY = 22; DISCARD_ENERGY_CARD = 23; TO_DECK = 24; TO_DECK_BOTTOM = 25; TO_PRIZE = 26


class Card:
    def __init__(self, card_id, *, card_type=CardType.TRAINER, attacks=None, skills=None, ex=False, mega=False, weakness=None, resistance=None):
        self.cardId = card_id
        self.id = card_id
        self.cardType = card_type
        self.attacks = attacks or []
        self.skills = skills or []
        self.ex = ex
        self.megaEx = mega
        self.stage1 = False
        self.stage2 = False
        self.energyType = EnergyType.DARKNESS if card_id == 7 else EnergyType.COLORLESS
        self.weakness = weakness
        self.resistance = resistance
        self.serial = card_id


class Pokemon(Card):
    def __init__(self, card_id, *, hp=100, max_hp=None, energies=None, tools=None, attacks=None):
        super().__init__(card_id, card_type=CardType.POKEMON, attacks=attacks or [])
        self.hp = hp
        self.maxHp = max_hp if max_hp is not None else hp
        self.energies = list(energies or [])
        self.energyCards = []
        self.tools = list(tools or [])
        self.preEvolution = []
        self.appearThisTurn = False


class Observation: pass


class Attack:
    def __init__(self, attack_id, energies=None, text=""):
        self.attackId = attack_id
        self.energies = list(energies or [])
        self.text = text


def build_cards():
    pokemon_ids = {104, 112, 343, 646, 647, 648, 860, 140, 741, 742, 743, 400, 401, 431, 414, 379, 380, 381, 341, 342, 65, 66, 305, 344, 345, 120, 121, 756}
    deck_ids = [int(x) for x in (ROOT / "deck.csv").read_text().splitlines() if x.strip()]
    result = []
    for card_id in sorted(set(deck_ids) | pokemon_ids | {1264}):
        if card_id == 7:
            result.append(Card(card_id, card_type=CardType.BASIC_ENERGY))
        elif card_id in pokemon_ids:
            attacks = [937] if card_id == 648 else ([936] if card_id == 647 else [])
            result.append(Card(card_id, card_type=CardType.POKEMON, attacks=attacks, ex=card_id in {648, 140, 431, 381, 121, 756}))
        else:
            result.append(Card(card_id, card_type=CardType.TRAINER))
    return result


ATTACKS = [Attack(937, [EnergyType.DARKNESS, EnergyType.DARKNESS]), Attack(936, [EnergyType.DARKNESS])]


def install_fake_cg():
    cg = types.ModuleType("cg")
    api = types.ModuleType("cg.api")
    for name, value in {
        "AreaType": AreaType, "Card": Card, "CardType": CardType, "EnergyType": EnergyType,
        "Observation": Observation, "OptionType": OptionType, "Pokemon": Pokemon,
        "SelectContext": SelectContext, "all_card_data": build_cards,
        "all_attack": lambda: ATTACKS, "to_observation_class": lambda x: x,
    }.items():
        setattr(api, name, value)
    cg.api = api
    sys.modules["cg"] = cg
    sys.modules["cg.api"] = api


def load_main():
    install_fake_cg()
    sys.path.insert(0, str(ROOT))
    for module in ("fallback_policy", "policy_base"):
        sys.modules.pop(module, None)
    # The rule policy moved to fallback_policy.py; main.py is now the
    # ML wrapper. These golden-state tests still target the rule policy.
    return importlib.import_module("fallback_policy")


def bare_policy(main):
    policy = main.GrimmsnarlPolicy.__new__(main.GrimmsnarlPolicy)
    policy.hand = defaultdict(int)
    policy.field = defaultdict(int)
    policy.discard = defaultdict(int)
    policy.cynthia_pressure = lambda: False
    policy.spidops_pressure = lambda: False
    return policy


def test_deck_is_60_and_unchanged():
    cards = [int(x) for x in (ROOT / "deck.csv").read_text().splitlines() if x.strip()]
    assert len(cards) == 60
    assert cards.count(7) == 10
    assert cards.count(648) == 3
    assert cards.count(112) == 4
    assert cards.count(1259) == 4


def test_runtime_is_self_contained():
    assert (ROOT / "main.py").exists()
    assert (ROOT / "policy_base.py").exists()
    assert (ROOT / "fallback_policy.py").exists()
    assert (ROOT / "ml_runtime.py").exists()
    assert (ROOT / "ml_features.py").exists()
    assert (ROOT / "ranker_model.json").exists()
    assert "from policy_base import" in (
        ROOT / "fallback_policy.py"
    ).read_text(encoding="utf-8")
    assert "from ml_runtime import" in (
        ROOT / "main.py"
    ).read_text(encoding="utf-8")


def test_initial_active_prefers_impidimp_then_munkidori_then_snorunt():
    # v6 order was a flat Impidimp > Munkidori > Snorunt.  v7 keeps that when the
    # Munkidori can escape (a spare copy or Darkness to pay retreat), so give it a
    # Darkness here; the sole-with-no-escape case is covered in the v7 tests.
    main = load_main()
    policy = bare_policy(main)
    imp = Pokemon(main.C.IMPIDIMP)
    munk = Pokemon(main.C.MUNKIDORI)
    snorunt = Pokemon(main.C.SNORUNT)
    policy.hand[main.C.IMPIDIMP] = 1
    policy.hand[main.C.DARKNESS] = 1
    assert policy.score_setup_active(imp) > policy.score_setup_active(munk)
    assert policy.score_setup_active(munk) > policy.score_setup_active(snorunt)
    policy.hand[main.C.IMPIDIMP] = 2
    assert policy.score_setup_active(imp) > policy.score_setup_active(snorunt)


def test_punk_up_builds_two_energy_backup_before_third_energy():
    main = load_main()
    policy = bare_policy(main)
    backup_one = Pokemon(main.C.IMPIDIMP, energies=[EnergyType.DARKNESS])
    backup_two = Pokemon(main.C.MORGREM, energies=[EnergyType.DARKNESS, EnergyType.DARKNESS])
    score_to_two = policy.punk_target_score(backup_one, AreaType.BENCH)
    score_third = policy.punk_target_score(backup_two, AreaType.BENCH)
    assert score_to_two > score_third


def test_attack_reservation_pushes_search_below_shadow_bullet():
    main = load_main()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.active_ko_ready = lambda: False
    policy.best_boss_value = lambda: -1
    policy.backup_is_close = lambda: True
    policy.opponent = types.SimpleNamespace(handCount=4)
    assert policy.reserve_adjust(main.C.POKE_PAD, 900_000) < 780_000


def test_night_stretcher_prefers_energy_for_unpowered_munkidori():
    main = load_main()
    policy = bare_policy(main)
    policy.effect_id = main.C.NIGHT_STRETCHER
    policy.powered_munkidori = lambda: False
    policy.backup_is_close = lambda: True
    policy.open_bench = lambda: 2
    energy = Card(main.C.DARKNESS, card_type=CardType.BASIC_ENERGY)
    grim = Pokemon(main.C.GRIMMSNARL_EX)
    policy.field[main.C.IMPIDIMP] = 1
    assert policy.score_search_target(energy) > policy.score_search_target(grim)


def test_boss_does_not_replace_equal_active_ko():
    main = load_main()
    policy = bare_policy(main)
    active = Pokemon(431, hp=100)
    bench = Pokemon(140, hp=100)
    policy.active_shadow_ready = lambda: True
    policy.shadow_damage = lambda target: 180
    policy.route_piece_bonus = lambda target: 0
    policy.opponent = types.SimpleNamespace(active=[active], bench=[bench])
    assert policy.best_boss_value() == -1


def test_direct_candy_route_does_not_demand_morgrem_search():
    main = load_main()
    policy = bare_policy(main)
    policy.eligible_impidimps = lambda: [Pokemon(main.C.IMPIDIMP)]
    policy.hand[main.C.RARE_CANDY] = 1
    policy.hand[main.C.GRIMMSNARL_EX] = 1
    policy.field[main.C.IMPIDIMP] = 1
    policy.field[main.C.MUNKIDORI] = 1
    assert policy.direct_candy_route()
    assert not policy.needs_morgrem_bridge()
    assert not policy.needs_non_rule_search()


def test_punk_up_prefers_visible_candy_route_over_unready_morgrem():
    main = load_main()
    policy = bare_policy(main)
    policy.hand[main.C.RARE_CANDY] = 1
    policy.hand[main.C.GRIMMSNARL_EX] = 1
    imp = Pokemon(main.C.IMPIDIMP, energies=[EnergyType.DARKNESS])
    morg = Pokemon(main.C.MORGREM, energies=[EnergyType.DARKNESS])
    assert policy.punk_target_score(imp, AreaType.BENCH) > policy.punk_target_score(morg, AreaType.BENCH)


def test_munkidori_can_be_powered_when_grim_route_is_visible():
    main = load_main()
    policy = bare_policy(main)
    munk = Pokemon(main.C.MUNKIDORI)
    me = types.SimpleNamespace(active=[], bench=[munk])
    opp = types.SimpleNamespace(active=[], bench=[])
    policy.obs = types.SimpleNamespace(current=types.SimpleNamespace(players=[me, opp]))
    policy.me = me
    policy.my_index = 0
    policy.active_shadow_ready = lambda: False
    policy.backup_is_close = lambda: False
    policy.grim_route_visible = lambda: True
    option = types.SimpleNamespace(inPlayArea=AreaType.BENCH, inPlayIndex=0)
    assert policy.score_attach(option) == 805_000


def test_crustle_blocks_shadow_bullet_active_damage_and_enables_boss_route():
    main = load_main()
    policy = bare_policy(main)
    crustle = Pokemon(main.C.CRUSTLE, hp=150)
    kang = Pokemon(main.C.MEGA_KANGASKHAN_EX, hp=180)
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[crustle], bench=[kang])
    assert policy.shadow_damage(crustle) == 0
    assert policy.best_boss_value() > 0


def test_live_shadow_reserves_attack_over_froslass_and_third_route_evolution():
    main = load_main()
    policy = bare_policy(main)
    snorunt = Pokemon(main.C.SNORUNT)
    froslass = Card(main.C.FROSLASS, card_type=CardType.POKEMON)
    imp = Pokemon(main.C.IMPIDIMP)
    grim = Card(main.C.GRIMMSNARL_EX, card_type=CardType.POKEMON)
    me = types.SimpleNamespace(hand=[froslass, grim], active=[], bench=[snorunt, imp])
    opp_mon = Pokemon(main.C.MUNKIDORI)
    opp = types.SimpleNamespace(active=[opp_mon], bench=[])
    policy.obs = types.SimpleNamespace(current=types.SimpleNamespace(players=[me, opp]))
    policy.me = me
    policy.opponent = opp
    policy.my_index = 0
    policy.active_shadow_ready = lambda: True
    policy.backup_is_close = lambda: True
    policy.field[main.C.GRIMMSNARL_EX] = 2
    policy.field[main.C.MORGREM] = 1
    fros_opt = types.SimpleNamespace(index=0, inPlayArea=AreaType.BENCH, inPlayIndex=0)
    grim_opt = types.SimpleNamespace(index=1, inPlayArea=AreaType.BENCH, inPlayIndex=1)
    assert policy.score_evolve(fros_opt) < 780_000
    assert policy.score_evolve(grim_opt) < 780_000


def test_counter_source_saves_critical_low_hp_munkidori():
    main = load_main()
    policy = bare_policy(main)
    munk = Pokemon(main.C.MUNKIDORI, hp=20, max_hp=100)
    grim = Pokemon(main.C.GRIMMSNARL_EX, hp=240, max_hp=320)
    assert policy.score_counter_source(munk) > policy.score_counter_source(grim)


def test_large_sample_route_pieces_receive_bonus():
    main = load_main()
    policy = bare_policy(main)
    assert policy.route_piece_bonus(Pokemon(main.C.DWEBBLE)) > 0
    assert policy.route_piece_bonus(Pokemon(main.C.DUNSPARCE_A)) > 0
    assert policy.route_piece_bonus(Pokemon(main.C.CRUSTLE)) > 0


def test_spikemuth_does_not_search_morgrem_when_candy_route_is_complete():
    main = load_main()
    policy = bare_policy(main)
    policy.effect_id = main.C.SPIKEMUTH_GYM
    policy.hand[main.C.RARE_CANDY] = 1
    policy.hand[main.C.GRIMMSNARL_EX] = 1
    policy.field[main.C.IMPIDIMP] = 1
    policy.me = types.SimpleNamespace(active=[Pokemon(main.C.GRIMMSNARL_EX)], bench=[], benchMax=5)
    morg = Pokemon(main.C.MORGREM)
    imp = Pokemon(main.C.IMPIDIMP)
    assert not policy.needs_morgrem_bridge()
    assert policy.score_search_target(imp) > policy.score_search_target(morg)


def test_non_rule_search_detects_missing_entire_grimmsnarl_line():
    main = load_main()
    policy = bare_policy(main)
    policy.field[main.C.MUNKIDORI] = 1
    assert policy.needs_non_rule_search()


def test_unfair_stamp_protects_large_unique_backup_hand():
    main = load_main()
    policy = bare_policy(main)
    policy.field[main.C.IMPIDIMP] = 1
    policy.hand[main.C.RARE_CANDY] = 1
    policy.hand[main.C.GRIMMSNARL_EX] = 1
    policy.backup_is_close = lambda: False
    policy.opponent_ids = lambda: {main.C.ALAKAZAM}
    policy.me = types.SimpleNamespace(handCount=9)
    policy.opponent = types.SimpleNamespace(handCount=5)
    assert policy.unique_backup_route_in_hand()
    assert policy.stamp_score() < 780_000


def test_unfair_stamp_is_pre_attack_when_refilling_small_hand_vs_large_hand():
    main = load_main()
    policy = bare_policy(main)
    policy.backup_is_close = lambda: True
    policy.opponent_ids = lambda: {main.C.ALAKAZAM}
    policy.me = types.SimpleNamespace(handCount=3)
    policy.opponent = types.SimpleNamespace(handCount=8)
    assert policy.stamp_score() > 780_000


def test_useful_second_munkidori_can_be_played_before_shadow_bullet():
    main = load_main()
    policy = bare_policy(main)
    active = Pokemon(main.C.GRIMMSNARL_EX, hp=290, max_hp=320, energies=[EnergyType.DARKNESS, EnergyType.DARKNESS])
    munk = Pokemon(main.C.MUNKIDORI, energies=[EnergyType.DARKNESS])
    opp_bench = Pokemon(main.C.ABRA, hp=60)
    policy.me = types.SimpleNamespace(active=[active], bench=[munk], benchMax=5, handCount=5)
    policy.opponent = types.SimpleNamespace(active=[Pokemon(main.C.ALAKAZAM, hp=300)], bench=[opp_bench], handCount=5)
    policy.field[main.C.MUNKIDORI] = 1
    policy.field[main.C.GRIMMSNARL_EX] = 1
    policy.open_bench = lambda: 3
    policy.reserved_bench_slots = lambda: 1
    policy.active_shadow_ready = lambda: True
    policy.powered_munkidori = lambda: True
    policy.backup_is_close = lambda: True
    policy.my_board = lambda: [active, munk]
    new_munk = Pokemon(main.C.MUNKIDORI)
    assert policy.useful_second_munkidori()
    assert policy.score_play_poke(new_munk) > 780_000


def test_second_munkidori_is_blocked_when_it_consumes_reserved_line_slot():
    main = load_main()
    policy = bare_policy(main)
    munk = Pokemon(main.C.MUNKIDORI, energies=[EnergyType.DARKNESS])
    policy.me = types.SimpleNamespace(active=[Pokemon(main.C.GRIMMSNARL_EX)], bench=[munk, Pokemon(main.C.SNORUNT), Pokemon(main.C.FROSLASS)], benchMax=5)
    policy.opponent = types.SimpleNamespace(active=[], bench=[Pokemon(main.C.ABRA, hp=60)])
    policy.field[main.C.MUNKIDORI] = 1
    policy.open_bench = lambda: 2
    policy.reserved_bench_slots = lambda: 2
    policy.my_board = lambda: [munk]
    assert not policy.useful_second_munkidori()


def test_rare_candy_completing_backup_stays_above_live_attack():
    main = load_main()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.can_rare_candy_now = lambda: True
    policy.backup_is_close = lambda: False
    policy.best_boss_value = lambda: -1
    assert policy.reserve_adjust(main.C.RARE_CANDY, 780_000) > 780_000


def test_two_prize_short_route_beats_isolated_one_prize_chip():
    main = load_main()
    policy = bare_policy(main)
    policy.powered_munkidori = lambda: True
    policy.opponent = types.SimpleNamespace(active=[], bench=[])
    policy.stadium_id = None
    two_prize = Pokemon(main.C.FEZANDIPITI_EX, hp=60)
    one_prize = Pokemon(main.C.ABRA, hp=30)
    # The two-prize target is reachable by Bullet+Brain and should outrank the
    # immediate but lower-value one-prize chip target.
    assert policy.spread_target_score(two_prize, True) > policy.spread_target_score(one_prize, True)
