"""v5-specific golden-state tests.

These cover the two fundamentals fixed over v4:
  * the damage-immune wall ("locked") handling (Crustle / Sylveon / Neutralization
    Zone) and the Boss's Orders unlock;
  * evolving on the Bench instead of feeding the Active into a grave.
"""
from __future__ import annotations

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


class Observation:
    pass


class Attack:
    def __init__(self, attack_id, energies=None, text="", damage=0):
        self.attackId = attack_id
        self.energies = list(energies or [])
        self.text = text
        self.damage = damage


def build_cards():
    # Includes Sylveon (330) so the generalised immune-wall path is exercised.
    pokemon_ids = {104, 112, 330, 343, 646, 647, 648, 860, 140, 741, 742, 743, 400, 401, 431, 414, 379, 380, 381, 341, 342, 65, 66, 305, 344, 345, 120, 121, 756}
    deck_ids = [int(x) for x in (ROOT / "deck.csv").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    result = []
    for card_id in sorted(set(deck_ids) | pokemon_ids | {1264, 1247}):
        if card_id == 7:
            result.append(Card(card_id, card_type=CardType.BASIC_ENERGY))
        elif card_id in pokemon_ids:
            attacks = [937] if card_id == 648 else ([936] if card_id == 647 else [])
            ex = card_id in {648, 140, 431, 381, 121, 756}
            result.append(Card(card_id, card_type=CardType.POKEMON, attacks=attacks, ex=ex))
        else:
            result.append(Card(card_id, card_type=CardType.TRAINER))
    return result


ATTACKS = [
    Attack(937, [EnergyType.DARKNESS, EnergyType.DARKNESS], damage=180),
    Attack(936, [EnergyType.DARKNESS, EnergyType.DARKNESS], damage=60),
]


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
    policy.stadium_id = 0
    policy.cynthia_pressure = lambda: False
    policy.spidops_pressure = lambda: False
    return policy


# ── Bug 1: damage-immune wall ("locked") ────────────────────────────────────
def test_generalized_ex_immunity_covers_sylveon_and_crustle():
    main = load_main()
    assert main.C.CRUSTLE in main.EX_ACTIVE_BLOCKERS
    assert main.C.SYLVEON in main.EX_ACTIVE_BLOCKERS
    policy = bare_policy(main)
    assert policy.shadow_damage(Pokemon(main.C.SYLVEON, hp=200)) == 0
    assert policy.shadow_damage(Pokemon(main.C.CRUSTLE, hp=150)) == 0


def test_neutralization_zone_prevents_damage_to_non_rule_box():
    main = load_main()
    policy = bare_policy(main)
    policy.stadium_id = main.C.NEUTRALIZATION_ZONE
    non_rule_box = Pokemon(main.C.MUNKIDORI, hp=110)  # not an ex
    rule_box = Pokemon(main.C.MEGA_KANGASKHAN_EX, hp=180)  # ex, protected only vs non-ex zone
    assert policy.shadow_damage(non_rule_box) == 0
    assert policy.shadow_damage(rule_box) == 180


def test_live_attack_ready_is_false_against_wall():
    main = load_main()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[Pokemon(main.C.CRUSTLE, hp=150)], bench=[])
    assert policy.active_target_immune_to_ex()
    assert not policy.live_attack_ready()


def test_walled_shadow_bullet_scores_below_development_and_boss():
    main = load_main()
    policy = bare_policy(main)
    crustle = Pokemon(main.C.CRUSTLE, hp=150)
    fez = Pokemon(main.C.FEZANDIPITI_EX, hp=170)
    active = Pokemon(main.C.GRIMMSNARL_EX, hp=320, energies=[EnergyType.DARKNESS, EnergyType.DARKNESS])
    policy.me = types.SimpleNamespace(active=[active], bench=[], prize=[0, 1, 2, 3], deckCount=40)
    policy.opponent = types.SimpleNamespace(active=[crustle], bench=[fez], handCount=5)
    policy.stadium_id = 0
    policy.active_shadow_ready = lambda: True
    option = types.SimpleNamespace(attackId=main.A.SHADOW_BULLET, type=OptionType.ATTACK)
    walled_attack = policy.score_attack(option)
    # gusting the benched Fezandipiti up is a real 180 unlock and must outrank the 0.
    assert policy.best_boss_value() >= 10_000
    assert walled_attack < 700_000


def test_boss_unlock_available_without_immediate_ko_when_locked():
    main = load_main()
    policy = bare_policy(main)
    crustle = Pokemon(main.C.CRUSTLE, hp=150)
    healthy_bench = Pokemon(main.C.MEGA_KANGASKHAN_EX, hp=180)  # cannot be KO'd by 180? hp==180 -> KO
    tanky = Pokemon(main.C.DRAGAPULT_EX, hp=320)               # survives 180
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[crustle], bench=[tanky])
    policy.stadium_id = 0
    # Even though the bench Dragapult survives 180, gusting it up unlocks real
    # damage vs the 0 we do to Crustle, so Boss is still valued.
    assert policy.best_boss_value() > 0


def test_walled_state_does_not_suppress_development():
    main = load_main()
    policy = bare_policy(main)
    policy.active_shadow_ready = lambda: True
    policy.opponent = types.SimpleNamespace(active=[Pokemon(main.C.CRUSTLE, hp=150)], bench=[])
    policy.best_boss_value = lambda: -1
    policy.backup_is_close = lambda: True
    # In v4 a "ready" Shadow Bullet capped optional search at 735k; while walled it
    # must not, because that attack is worthless.
    assert policy.reserve_adjust(main.C.POKE_PAD, 900_000) == 900_000


# ── Bug 2: evolve on the bench, not into a grave ─────────────────────────────
def _evolve_policy(main, *, active, bench_bodies, hand_card_id, opp_active, energies_on_active=None):
    """Build a policy whose select offers evolving hand_card onto Active and Bench."""
    policy = bare_policy(main)
    active_mon = Pokemon(active, energies=energies_on_active or [])
    bench_mons = [Pokemon(b) for b in bench_bodies]
    hand_card = Card(hand_card_id, card_type=CardType.POKEMON)
    me = types.SimpleNamespace(active=[active_mon], bench=bench_mons, hand=[hand_card], benchMax=5)
    opp = types.SimpleNamespace(active=[opp_active] if opp_active else [], bench=[])
    policy.obs = types.SimpleNamespace(current=types.SimpleNamespace(players=[me, opp]))
    policy.me = me
    policy.opponent = opp
    policy.my_index = 0
    policy.op_index = 1
    active_opt = types.SimpleNamespace(type=OptionType.EVOLVE, index=0, inPlayArea=AreaType.ACTIVE, inPlayIndex=0)
    bench_opt = types.SimpleNamespace(type=OptionType.EVOLVE, index=0, inPlayArea=AreaType.BENCH, inPlayIndex=0)
    policy.select = types.SimpleNamespace(option=[active_opt, bench_opt])
    return policy, active_opt, bench_opt


def test_morgrem_prefers_bench_when_active_cannot_attack():
    main = load_main()
    # Active Impidimp with no energy -> the evolved Morgrem cannot attack this turn.
    threat = Pokemon(main.C.DRAGAPULT_EX, hp=320, energies=[EnergyType.DARKNESS, EnergyType.DARKNESS])
    policy, active_opt, bench_opt = _evolve_policy(
        main, active=main.C.IMPIDIMP, bench_bodies=[main.C.IMPIDIMP],
        hand_card_id=main.C.MORGREM, opp_active=threat, energies_on_active=[])
    assert policy.score_evolve(bench_opt) > policy.score_evolve(active_opt)


def test_morgrem_evolves_active_when_it_can_attack_this_turn():
    main = load_main()
    # Active already has two energy: evolving to Morgrem in the Active can attack now.
    policy, active_opt, bench_opt = _evolve_policy(
        main, active=main.C.IMPIDIMP, bench_bodies=[main.C.IMPIDIMP],
        hand_card_id=main.C.MORGREM, opp_active=Pokemon(main.C.MUNKIDORI),
        energies_on_active=[EnergyType.DARKNESS, EnergyType.DARKNESS])
    assert policy.score_evolve(active_opt) > policy.score_evolve(bench_opt)


def test_grimmsnarl_prefers_bench_into_immune_wall():
    main = load_main()
    crustle = Pokemon(main.C.CRUSTLE, hp=150)
    policy, active_opt, bench_opt = _evolve_policy(
        main, active=main.C.MORGREM, bench_bodies=[main.C.MORGREM],
        hand_card_id=main.C.GRIMMSNARL_EX, opp_active=crustle)
    policy.active_shadow_ready = lambda: False
    assert policy.score_evolve(bench_opt) > policy.score_evolve(active_opt)


def test_grimmsnarl_prefers_active_normally():
    main = load_main()
    policy, active_opt, bench_opt = _evolve_policy(
        main, active=main.C.MORGREM, bench_bodies=[main.C.MORGREM],
        hand_card_id=main.C.GRIMMSNARL_EX, opp_active=Pokemon(main.C.MUNKIDORI))
    policy.active_shadow_ready = lambda: False
    assert policy.score_evolve(active_opt) > policy.score_evolve(bench_opt)
