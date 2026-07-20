from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from collections import Counter


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5


class CardType(IntEnum):
    POKEMON = 1
    BASIC_ENERGY = 2
    SPECIAL_ENERGY = 3
    ITEM = 4
    SUPPORTER = 5
    STADIUM = 6
    TOOL = 7


class EnergyType(IntEnum):
    COLORLESS = 0
    PSYCHIC = 5


class OptionType(IntEnum):
    CARD = 1
    PLAY = 7
    ENERGY = 8
    ATTACH = 9
    EVOLVE = 10
    ABILITY = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    YES = 15
    NO = 16


class SelectContext(IntEnum):
    MAIN = 1
    ATTACH_TO = 2
    ACTIVE = 3
    SETUP = 4


@dataclass
class Skill:
    text: str = ""


@dataclass
class CardData:
    cardId: int
    cardType: CardType = CardType.POKEMON
    attacks: list[int] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    weakness: int | None = None
    resistance: int | None = None
    stage1: bool = False
    stage2: bool = False
    ex: bool = False
    megaEx: bool = False
    energyType: int = EnergyType.COLORLESS
    aceSpec: bool = False
    name: str = ""


@dataclass
class AttackData:
    attackId: int
    energies: list[int] = field(default_factory=list)
    text: str = ""
    damage: int = 0


@dataclass
class Card:
    id: int
    serial: int = 0
    playerIndex: int = 0


@dataclass
class Pokemon:
    id: int
    hp: int = 100
    maxHp: int = 100
    energies: list[int] = field(default_factory=list)
    energyCards: list[Card] = field(default_factory=list)
    tools: list[Card] = field(default_factory=list)
    preEvolution: list[Card] = field(default_factory=list)
    playerIndex: int = 0
    serial: int = 0


class Observation: pass
class Option: pass


def _cards():
    return [
        CardData(5, CardType.BASIC_ENERGY, energyType=EnergyType.PSYCHIC),
        CardData(19, CardType.SPECIAL_ENERGY, energyType=EnergyType.PSYCHIC),
        CardData(305, attacks=[423, 424]),
        CardData(66, attacks=[76], stage1=True),
        CardData(741, attacks=[1070]),
        CardData(742, attacks=[1071], stage1=True),
        CardData(743, attacks=[1072], stage2=True),
        CardData(104, attacks=[509], stage1=True, name="Froslass"),
        CardData(343, attacks=[], skills=[Skill(
            "Prevent all damage done to your Benched Pokemon by attacks."
        )], name="Shaymin"),
        CardData(414, attacks=[], skills=[Skill(
            "Prevent all effects of attacks done to your Team Rocket's Pokemon."
        )], name="Team Rocket's Articuno"),
        CardData(675, attacks=[], name="Lunatone"),
        CardData(678, attacks=[983], stage1=True, ex=True, name="Mega Lucario ex"),
        CardData(666, attacks=[900], ex=True),
        CardData(140, attacks=[500], ex=True),
        CardData(112, attacks=[501]),
        CardData(121, attacks=[502], stage2=True, ex=True),
        CardData(345, attacks=[503], stage2=True),
        CardData(756, attacks=[504], stage2=True),
        CardData(878, attacks=[505]),
        CardData(879, attacks=[506], stage1=True, ex=True),
        CardData(647, attacks=[507], stage1=True),
        CardData(648, attacks=[508], stage2=True, ex=True, name="Marnie's Grimmsnarl ex"),
        CardData(9000, attacks=[]),
    ]


def _attacks():
    return [
        AttackData(1072, [EnergyType.PSYCHIC]),
        AttackData(1071, [EnergyType.PSYCHIC]),
        AttackData(1070, [EnergyType.PSYCHIC]),
        AttackData(76, [EnergyType.COLORLESS, EnergyType.COLORLESS, EnergyType.COLORLESS]),
        AttackData(423, []), AttackData(424, [EnergyType.COLORLESS]),
        AttackData(983, [EnergyType.COLORLESS]),
        *[AttackData(i, [EnergyType.COLORLESS]) for i in range(500, 508)],
        AttackData(508, [EnergyType.COLORLESS], text="This attack also does 30 damage to each of your opponent's Benched Pokemon."),
        AttackData(509, [EnergyType.COLORLESS]),
        AttackData(900, [EnergyType.COLORLESS]),
    ]


def install_cg_stub():
    cg = types.ModuleType("cg")
    api = types.ModuleType("cg.api")
    for name, value in {
        "AreaType": AreaType,
        "Card": Card,
        "CardType": CardType,
        "EnergyType": EnergyType,
        "Observation": Observation,
        "OptionType": OptionType,
        "Pokemon": Pokemon,
        "SelectContext": SelectContext,
        "all_card_data": _cards,
        "all_attack": _attacks,
        "to_observation_class": lambda x: x,
    }.items():
        setattr(api, name, value)
    cg.api = api
    sys.modules["cg"] = cg
    sys.modules["cg.api"] = api


def load_policy():
    install_cg_stub()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    sys.modules.pop("fallback_v3", None)
    return importlib.import_module("fallback_v3")


def bare_policy(policy, *, hand_count=8, active=None, bench=(), opp_active=None, opp_bench=()):
    obj = policy.AlakazamPolicy.__new__(policy.AlakazamPolicy)
    obj.my_index = 0
    obj.op_index = 1
    obj.me = SimpleNamespace(
        active=[active] if active is not None else [],
        bench=list(bench) + [None] * (5 - len(bench)),
        hand=[], handCount=hand_count, discard=[], deckCount=30,
        prize=[Card(9000 + i) for i in range(6)], benchMax=5,
    )
    obj.opponent = SimpleNamespace(
        active=[opp_active] if opp_active is not None else [],
        bench=list(opp_bench) + [None] * (5 - len(opp_bench)),
        hand=[], handCount=6, discard=[], deckCount=30,
        prize=[Card(9100 + i) for i in range(6)], benchMax=5,
    )
    obj.state = SimpleNamespace(supporterPlayed=False)
    obj.select = SimpleNamespace(contextCard=None, option=[])
    obj.obs = object()
    obj.field = Counter(p.id for p in obj.me.active + obj.me.bench if p is not None)
    obj.hand = Counter()
    obj.discard = Counter()
    obj._effect_prevented = lambda target: False
    return obj


def test_deck_is_v8_sixty_cards_with_shaymin():
    policy = load_policy()
    counts = Counter(policy.my_deck)
    assert len(policy.my_deck) == 60
    assert counts[5] == 2
    assert counts[1182] == 3
    assert counts[66] == 2
    assert counts[343] == 1
    assert counts[142] == 0
    assert counts[1156] == 0


def test_only_active_dudunsparce_can_never_use_run_away_draw():
    policy = load_policy()
    dudun = Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140)
    obj = bare_policy(policy, active=dudun)
    obj._deck_spend_ok = lambda *args, **kwargs: True
    obj._item_locked = lambda: True
    obj._bench_attacker_ready = lambda: False
    old_get_card = policy.get_card
    policy.get_card = lambda *args, **kwargs: dudun
    try:
        option = SimpleNamespace(area=AreaType.ACTIVE, index=0)
        assert obj._board_body_count() == 1
        assert obj._score_ability(option) == -1
    finally:
        policy.get_card = old_get_card


def test_active_dudunsparce_may_cycle_when_a_ready_body_remains():
    policy = load_policy()
    dudun = Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140)
    alakazam = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                       energies=[EnergyType.PSYCHIC])
    obj = bare_policy(policy, active=dudun, bench=[alakazam])
    obj._deck_spend_ok = lambda *args, **kwargs: True
    obj._item_locked = lambda: False
    obj._bench_attacker_ready = lambda: True
    old_get_card = policy.get_card
    policy.get_card = lambda *args, **kwargs: dudun
    try:
        option = SimpleNamespace(area=AreaType.ACTIVE, index=0)
        assert obj._board_body_count() == 2
        assert obj._score_ability(option) == 14000
    finally:
        policy.get_card = old_get_card


def test_boss_prefers_rocket_articuno_over_disposable_basic():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    active = Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    articuno = Pokemon(policy.ROCKET_ARTICUNO_ID, hp=110, maxHp=110, playerIndex=1)
    disposable = Pokemon(9000, hp=70, maxHp=70, playerIndex=1)
    obj = bare_policy(policy, hand_count=8, active=mine,
                      opp_active=active, opp_bench=[articuno, disposable])
    assert obj._boss_damage_after_spend(articuno) == 140
    assert obj._boss_target_score(articuno) > 0
    assert obj._boss_target_score(disposable) < 0
    assert obj._gust_value(articuno) > obj._gust_value(disposable)


def test_current_rocket_articuno_id_is_not_lunatone():
    policy = load_policy()
    obj = bare_policy(policy)
    articuno = Pokemon(414, hp=110, maxHp=110, playerIndex=1)
    lunatone = Pokemon(675, hp=90, maxHp=90, playerIndex=1)
    assert policy.ROCKET_ARTICUNO_ID == 414
    assert obj._boss_role_bonus(articuno) >= 7000
    assert obj._boss_role_bonus(lunatone) == 0


def test_boss_is_rejected_when_hand_spend_loses_the_ko():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    active = Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    articuno = Pokemon(policy.ROCKET_ARTICUNO_ID, hp=130, maxHp=130, playerIndex=1)
    # Six cards: current Powerful Hand is 120, after Boss only 100. The 130 HP
    # Articuno is therefore not a same-turn KO and must not be gusted.
    obj = bare_policy(policy, hand_count=6, active=mine,
                      opp_active=active, opp_bench=[articuno])
    assert obj._boss_damage_after_spend(articuno) == 100
    assert obj._boss_target_score(articuno) < 0


def test_boss_does_not_replace_a_better_active_ko():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    # Active two-prize ex is already KO-able; a developed but one-prize bench
    # target without a stronger role must not replace it.
    active = Pokemon(666, hp=120, maxHp=230, playerIndex=1)
    bench = Pokemon(647, hp=80, maxHp=100, playerIndex=1)
    obj = bare_policy(policy, hand_count=8, active=mine,
                      opp_active=active, opp_bench=[bench])
    assert obj._boss_target_score(bench) < 0


def test_boss_never_gusts_away_an_immediate_game_winning_active_ko():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    active = Pokemon(666, hp=100, maxHp=230, playerIndex=1)
    articuno = Pokemon(policy.ROCKET_ARTICUNO_ID, hp=80, maxHp=110, playerIndex=1)
    obj = bare_policy(policy, hand_count=8, active=mine,
                      opp_active=active, opp_bench=[articuno])
    obj.me.prize = [Card(9999), Card(10000)]  # two remain; Active ex wins, Articuno does not
    assert obj._boss_target_score(articuno) < 0


def test_boss_can_choose_urgent_two_hit_multi_prize_route():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    active = Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    grimmsnarl = Pokemon(policy.GRIMMSNARL_EX_ID, hp=320, maxHp=320,
                         energies=[EnergyType.COLORLESS], playerIndex=1)
    obj = bare_policy(policy, hand_count=9, active=mine,
                      opp_active=active, opp_bench=[grimmsnarl])
    obj.me.prize = [Card(9999 + i) for i in range(3)]
    assert obj._boss_damage_after_spend(grimmsnarl) == 160
    assert obj._boss_target_score(grimmsnarl) > 0


def test_boss_rejects_early_two_hit_two_prize_route():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    active = Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    grimmsnarl = Pokemon(policy.GRIMMSNARL_EX_ID, hp=320, maxHp=320,
                         energies=[EnergyType.COLORLESS], playerIndex=1)
    obj = bare_policy(policy, hand_count=9, active=mine,
                      opp_active=active, opp_bench=[grimmsnarl])
    assert len(obj.me.prize) == 6
    assert obj._boss_target_score(grimmsnarl) < 0


def test_dual_kadabra_choice_prefers_bench_development():
    policy = load_policy()
    active_abra = Pokemon(policy.C.ABRA, hp=40, maxHp=40)
    bench_abra = Pokemon(policy.C.ABRA, hp=40, maxHp=40)
    opponent = Pokemon(9000, hp=100, maxHp=100, energies=[EnergyType.COLORLESS],
                       playerIndex=1)
    obj = bare_policy(policy, active=active_abra, bench=[bench_abra], opp_active=opponent)
    obj._same_evolution_area_available = lambda card_id, area: True
    obj._can_attack = lambda target: True
    obj._opp_has_froslass = lambda: False
    active_option = SimpleNamespace(inPlayArea=AreaType.ACTIVE)
    bench_option = SimpleNamespace(inPlayArea=AreaType.BENCH)
    assert obj._kadabra_target_bonus(bench_option, bench_abra) > obj._kadabra_target_bonus(
        active_option, active_abra
    )


def test_dual_kadabra_choice_keeps_immediate_active_ko():
    policy = load_policy()
    active_abra = Pokemon(policy.C.ABRA, hp=40, maxHp=40,
                          energies=[EnergyType.PSYCHIC])
    bench_abra = Pokemon(policy.C.ABRA, hp=40, maxHp=40)
    opponent = Pokemon(9000, hp=30, maxHp=100, playerIndex=1)
    obj = bare_policy(policy, active=active_abra, bench=[bench_abra], opp_active=opponent)
    obj._same_evolution_area_available = lambda card_id, area: True
    obj._can_attack = lambda target: False
    obj._opp_has_froslass = lambda: False
    active_option = SimpleNamespace(inPlayArea=AreaType.ACTIVE)
    bench_option = SimpleNamespace(inPlayArea=AreaType.BENCH)
    assert obj._kadabra_target_bonus(active_option, active_abra) > obj._kadabra_target_bonus(
        bench_option, bench_abra
    )


def test_shaymin_is_only_benched_against_attack_spread():
    policy = load_policy()
    grimmsnarl = Pokemon(policy.GRIMMSNARL_EX_ID, playerIndex=1)
    shaymin = Card(policy.C.SHAYMIN)
    obj = bare_policy(policy, opp_active=grimmsnarl)
    assert obj._opp_threatens_bench()
    assert obj._score_play_poke(shaymin) == 17000

    quiet = bare_policy(policy, opp_active=Pokemon(9000, playerIndex=1))
    assert not quiet._opp_threatens_bench()
    assert quiet._score_play_poke(shaymin) < 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"{len(tests)} v8 runtime logic tests passed")
