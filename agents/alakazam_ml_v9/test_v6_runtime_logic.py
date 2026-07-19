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
    NUMBER = 0
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
    DISCARD_ENERGY = 30
    IS_FIRST = 41
    ACTIVATE = 43


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
    retreatCost: int = 0


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
        CardData(11, CardType.SPECIAL_ENERGY, skills=[Skill(
            "Prevent all effects of attacks done to the Pokemon this card is attached to."
        )], name="Mist Energy"),
        CardData(14, CardType.SPECIAL_ENERGY, skills=[Skill(
            "If damaged by an attack, put 2 damage counters on the Attacking Pokemon."
        )], name="Spiky Energy"),
        CardData(18, CardType.SPECIAL_ENERGY, skills=[Skill(
            "The Grass Pokemon this card is attached to gets +20 HP."
        )], name="Grow Grass Energy"),
        CardData(20, CardType.SPECIAL_ENERGY, skills=[Skill(
            "Prevent all effects of attacks done to this Pokemon."
        )], name="Rock Fighting Energy"),
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
        CardData(678, attacks=[983], stage1=True, ex=True, name="Mega Lucario ex",
                 retreatCost=3),
        CardData(666, attacks=[900], ex=True),
        CardData(140, attacks=[183], ex=True, retreatCost=1),
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
        AttackData(183, [EnergyType.COLORLESS, EnergyType.COLORLESS, EnergyType.COLORLESS]),
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


def test_v9_boss_rejects_non_closing_two_hit_multi_prize_route():
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
    assert obj._boss_target_score(grimmsnarl) < 0


def test_v9_boss_allows_sticky_prize_closing_main_attacker():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    active = Pokemon(9000, hp=200, maxHp=200, playerIndex=1)
    lucario = Pokemon(678, hp=320, maxHp=320,
                      energies=[EnergyType.COLORLESS, EnergyType.COLORLESS],
                      playerIndex=1)
    obj = bare_policy(policy, hand_count=9, active=mine,
                      opp_active=active, opp_bench=[lucario])
    obj.me.prize = [Card(9999), Card(10000)]
    assert obj._boss_damage_after_spend(lucario) == 160
    assert obj._boss_target_score(lucario) > 0


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


def test_v9_dual_kadabra_uses_active_for_only_immediate_attack():
    policy = load_policy()
    active_abra = Pokemon(policy.C.ABRA, hp=50, maxHp=50,
                          energies=[EnergyType.PSYCHIC])
    bench_abra = Pokemon(policy.C.ABRA, hp=50, maxHp=50)
    opponent = Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    obj = bare_policy(policy, active=active_abra, bench=[bench_abra], opp_active=opponent)
    obj._same_evolution_area_available = lambda card_id, area: True
    active_option = SimpleNamespace(inPlayArea=AreaType.ACTIVE)
    bench_option = SimpleNamespace(inPlayArea=AreaType.BENCH)
    assert obj._kadabra_target_bonus(active_option, active_abra) > obj._kadabra_target_bonus(
        bench_option, bench_abra
    )


def test_v9_hammer_scores_attached_mist_not_owning_pokemon():
    policy = load_policy()
    owner = Pokemon(678, hp=300, maxHp=320, playerIndex=1,
                    energies=[EnergyType.COLORLESS, EnergyType.COLORLESS],
                    energyCards=[Card(11), Card(14)])
    obj = bare_policy(policy, opp_active=owner)
    obj.context = SelectContext.DISCARD_ENERGY
    obj.select = SimpleNamespace(contextCard=Card(policy.C.ENHANCED_HAMMER), option=[])
    old_get_card = policy.get_card
    policy.get_card = lambda *args, **kwargs: owner
    try:
        mist = SimpleNamespace(type=OptionType.ENERGY, area=AreaType.ACTIVE,
                               inPlayArea=AreaType.ACTIVE, index=0,
                               playerIndex=1, energyIndex=0)
        spiky = SimpleNamespace(type=OptionType.ENERGY, area=AreaType.ACTIVE,
                                inPlayArea=AreaType.ACTIVE, index=0,
                                playerIndex=1, energyIndex=1)
        assert obj._score(mist) > obj._score(spiky)
    finally:
        policy.get_card = old_get_card


def test_v9_reserves_actual_last_hammer_for_crustle_mist():
    policy = load_policy()
    crustle = Pokemon(345, playerIndex=1)
    obj = bare_policy(policy, opp_active=Pokemon(9000, playerIndex=1), opp_bench=[crustle])
    obj.hand[policy.C.ENHANCED_HAMMER] = 1
    obj.discard[policy.C.ENHANCED_HAMMER] = 3
    assert obj._mist_probability() >= 0.65
    assert obj._should_reserve_last_hammer()


def test_v9_releases_hammer_reservation_for_immediate_attack_denial():
    policy = load_policy()
    active = Pokemon(678, hp=300, maxHp=320, playerIndex=1,
                     energies=[EnergyType.COLORLESS], energyCards=[Card(14)])
    crustle = Pokemon(345, playerIndex=1)
    obj = bare_policy(policy, opp_active=active, opp_bench=[crustle])
    obj.hand[policy.C.ENHANCED_HAMMER] = 1
    obj.discard[policy.C.ENHANCED_HAMMER] = 3
    assert obj._non_mist_hammer_exception()
    assert not obj._should_reserve_last_hammer()


def test_v9_fez_do_not_bench_at_two_opponent_prizes():
    policy = load_policy()
    obj = bare_policy(policy, active=Pokemon(policy.C.ABRA),
                      opp_active=Pokemon(9000, playerIndex=1))
    obj.opponent.prize = [Card(1), Card(2)]
    assert obj._fez_mode(for_bench=True) == "DO_NOT_BENCH"
    assert not obj._fez_bench_worthwhile()


def test_v9_fez_pivot_pays_one_energy_to_reach_ready_alakazam():
    policy = load_policy()
    fez = Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    alakazam = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                       energies=[EnergyType.PSYCHIC])
    obj = bare_policy(policy, active=fez, bench=[alakazam],
                      opp_active=Pokemon(9000, playerIndex=1))
    assert obj._fez_mode(fez) == "PIVOT"
    assert obj._fez_attach_score(fez, True) == 17500


def test_v9_fez_alternate_attacker_is_limited_to_short_spidops_route():
    policy = load_policy()
    fez = Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210,
                  energies=[EnergyType.PSYCHIC])
    articuno = Pokemon(policy.ROCKET_ARTICUNO_ID, hp=110, maxHp=110, playerIndex=1)
    obj = bare_policy(policy, active=fez, opp_active=articuno)
    obj.me.hand = [Card(policy.C.PSYCHIC_ENERGY), Card(policy.C.TELEPATH_ENERGY)]
    obj.me.handCount = 2
    obj.hand[policy.C.PSYCHIC_ENERGY] = 1
    obj.hand[policy.C.TELEPATH_ENERGY] = 1
    assert obj._fez_energy_eta(fez) == 2
    assert obj._fez_mode(fez) == "ALTERNATE_ATTACKER"
    assert obj._fez_attach_score(fez, True) > 0


def test_v9_fez_cruel_arrow_attacks_a_100_hp_prize():
    policy = load_policy()
    fez = Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210,
                  energies=[EnergyType.PSYCHIC] * 3)
    target = Pokemon(9000, hp=80, maxHp=100, playerIndex=1)
    obj = bare_policy(policy, active=fez, opp_active=target)
    option = SimpleNamespace(attackId=policy.FEZANDIPITI_ATTACK)
    assert obj._score_attack(option) >= 26000


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
