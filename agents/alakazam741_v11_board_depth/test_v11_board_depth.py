from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from collections import Counter, defaultdict
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent


class AreaType(IntEnum):
    DECK = 0
    HAND = 1
    DISCARD = 2
    ACTIVE = 3
    BENCH = 4
    PRIZE = 5
    STADIUM = 6
    LOOKING = 7


class CardType(IntEnum):
    POKEMON = 1
    BASIC_ENERGY = 2
    SPECIAL_ENERGY = 3
    ITEM = 4
    SUPPORTER = 5
    STADIUM = 6


class EnergyType(IntEnum):
    COLORLESS = 0
    PSYCHIC = 5


class OptionType(IntEnum):
    YES = 1
    NO = 2
    NUMBER = 3
    CARD = 4
    PLAY = 5
    ENERGY = 6
    ATTACH = 7
    EVOLVE = 8
    ABILITY = 9
    RETREAT = 10
    ATTACK = 11
    END = 12


class SelectContext(IntEnum):
    MAIN = 1
    IS_FIRST = 2
    MULLIGAN = 3
    ACTIVATE = 4
    SWITCH = 5
    TO_ACTIVE = 6
    SETUP_ACTIVE_POKEMON = 7
    SETUP_BENCH_POKEMON = 8
    TO_BENCH = 9
    TO_FIELD = 10
    TO_HAND = 11
    ATTACH_TO = 12
    ATTACH_FROM = 13
    TO_HAND_ENERGY = 14
    DISCARD = 15
    DISCARD_CARD_OR_ATTACHED_CARD = 16
    DISCARD_ENERGY = 17
    DISCARD_ENERGY_CARD = 18
    DAMAGE_COUNTER = 19
    DAMAGE_COUNTER_ANY = 20
    TO_DECK = 21
    TO_DECK_BOTTOM = 22
    TO_PRIZE = 23


class Pokemon:
    def __init__(self, cid, hp=100, energies=None, energy_cards=None, appear=False):
        self.id = cid
        self.hp = hp
        self.maxHp = hp
        self.energies = list(energies or [])
        self.energyCards = list(energy_cards or [])
        self.tools = []
        self.preEvolution = []
        self.appearThisTurn = appear
        self.serial = cid * 10 + hp


class Card:
    def __init__(self, cid):
        self.id = cid
        self.serial = cid * 10


class Observation:
    pass


class CardData:
    def __init__(self, cid, card_type, attacks=(), *, stage1=0, stage2=0,
                 energy_type=EnergyType.COLORLESS, name=None, ex=False):
        self.cardId = cid
        self.cardType = card_type
        self.attacks = list(attacks)
        self.stage1 = stage1
        self.stage2 = stage2
        self.energyType = energy_type
        self.name = name or str(cid)
        self.ex = ex
        self.megaEx = False
        self.skills = []
        self.weakness = None
        self.resistance = None


class AttackData:
    def __init__(self, aid, energies, text=""):
        self.attackId = aid
        self.energies = list(energies)
        self.text = text


ATTACKS = [
    AttackData(1070, [EnergyType.PSYCHIC]),
    AttackData(1071, [EnergyType.PSYCHIC]),
    AttackData(1072, [EnergyType.PSYCHIC]),
    AttackData(423, []),
    AttackData(424, [EnergyType.COLORLESS]),
    AttackData(76, [EnergyType.COLORLESS, EnergyType.COLORLESS, EnergyType.COLORLESS]),
    AttackData(183, [EnergyType.PSYCHIC, EnergyType.COLORLESS, EnergyType.COLORLESS]),
    AttackData(9001, [EnergyType.PSYCHIC, EnergyType.PSYCHIC, EnergyType.COLORLESS],
               "This attack does 30 damage to 1 of your opponent's Benched Pokémon."),
    AttackData(9002, [EnergyType.PSYCHIC, EnergyType.COLORLESS]),
]

POKEMON_DATA = {
    741: CardData(741, CardType.POKEMON, [1070], name="Abra"),
    742: CardData(742, CardType.POKEMON, [1071], stage1=741, name="Kadabra"),
    743: CardData(743, CardType.POKEMON, [1072], stage2=742, name="Alakazam"),
    305: CardData(305, CardType.POKEMON, [423, 424], name="Dunsparce"),
    66: CardData(66, CardType.POKEMON, [76], stage1=305, name="Dudunsparce"),
    140: CardData(140, CardType.POKEMON, [183], name="Fezandipiti ex", ex=True),
    343: CardData(343, CardType.POKEMON, [], name="Shaymin"),
    999: CardData(999, CardType.POKEMON, [9001], name="Bench attacker"),
    998: CardData(998, CardType.POKEMON, [9002], name="Two-energy attacker"),
}

OTHER_DATA = {
    5: CardData(5, CardType.BASIC_ENERGY, energy_type=EnergyType.PSYCHIC),
    13: CardData(13, CardType.SPECIAL_ENERGY, energy_type=EnergyType.COLORLESS),
    19: CardData(19, CardType.SPECIAL_ENERGY, energy_type=EnergyType.PSYCHIC),
    1079: CardData(1079, CardType.ITEM),
    1081: CardData(1081, CardType.ITEM),
    1082: CardData(1082, CardType.ITEM),
    1086: CardData(1086, CardType.ITEM),
    1097: CardData(1097, CardType.ITEM),
    1129: CardData(1129, CardType.ITEM),
    1152: CardData(1152, CardType.ITEM),
    1184: CardData(1184, CardType.SUPPORTER),
    1197: CardData(1197, CardType.SUPPORTER),
    1225: CardData(1225, CardType.SUPPORTER),
    1231: CardData(1231, CardType.SUPPORTER),
    1264: CardData(1264, CardType.STADIUM),
    2000: CardData(2000, CardType.SPECIAL_ENERGY, energy_type=EnergyType.PSYCHIC),
}
ALL_CARDS = list(POKEMON_DATA.values()) + list(OTHER_DATA.values())


def install_cg_stub():
    cg = types.ModuleType("cg")
    api = types.ModuleType("cg.api")
    for name, value in {
        "AreaType": AreaType,
        "CardType": CardType,
        "EnergyType": EnergyType,
        "Observation": Observation,
        "OptionType": OptionType,
        "Pokemon": Pokemon,
        "SelectContext": SelectContext,
        "all_card_data": lambda: ALL_CARDS,
        "all_attack": lambda: ATTACKS,
        "to_observation_class": lambda x: x,
    }.items():
        setattr(api, name, value)
    cg.api = api
    sys.modules["cg"] = cg
    sys.modules["cg.api"] = api


install_cg_stub()
sys.path.insert(0, str(ROOT))
for mod in ("main", "policy_base"):
    sys.modules.pop(mod, None)
main = importlib.import_module("main")


def opt(kind, *, area=AreaType.HAND, index=0, in_area=AreaType.BENCH,
        in_index=0, attack_id=None, player_index=0):
    return NS(type=kind, area=area, index=index, inPlayArea=in_area,
              inPlayIndex=in_index, attackId=attack_id, playerIndex=player_index,
              number=None)


def make_player(active=None, bench=None, hand=None, discard=None, prize_count=6,
                deck_count=40):
    active_list = [] if active is None else [active]
    bench_list = list(bench or [])
    while len(bench_list) < 5:
        bench_list.append(None)
    cards = list(hand or [])
    return NS(active=active_list, bench=bench_list, hand=cards,
              handCount=len(cards), discard=list(discard or []), prize=[object()] * prize_count,
              deckCount=deck_count, benchMax=5)


def make_policy(*, mine, opp, options=None, first_player=0, your_index=0, turn=2,
                supporter_played=False, energy_attached=False, stadium_played=False,
                stadium=None):
    state = NS(yourIndex=your_index, players=[mine, opp] if your_index == 0 else [opp, mine],
               stadium=list(stadium or []), supporterPlayed=supporter_played,
               energyAttached=energy_attached, stadiumPlayed=stadium_played,
               firstPlayer=first_player, turn=turn)
    select = NS(context=SelectContext.MAIN, option=list(options or []), minCount=1,
                maxCount=1, contextCard=None, deck=[])
    obs = NS(current=state, select=select)
    return main.AlakazamPolicy(obs)


class V11Tests(unittest.TestCase):
    def test_deck_is_expected_sixty(self):
        ids = [int(x) for x in (ROOT / "deck.csv").read_text().splitlines() if x.strip()]
        c = Counter(ids)
        self.assertEqual(len(ids), 60)
        self.assertEqual(c[140], 1)
        self.assertEqual(c[343], 1)
        self.assertEqual(c[1152], 4)
        self.assertEqual(c[1231], 4)
        self.assertEqual(c[1264], 1)
        self.assertEqual(c[1097], 2)
        self.assertEqual(c[13], 1)
        self.assertEqual(c[5], 2)
        self.assertEqual(c[1227], 0)

    def test_second_player_asks_for_three_routes(self):
        ala = Pokemon(743, energies=[EnergyType.PSYCHIC])
        kad = Pokemon(742, energies=[EnergyType.PSYCHIC])
        abra = Pokemon(741)
        mine = make_player(active=ala, bench=[kad, abra], hand=[Card(743)])
        opp = make_player(active=Pokemon(999, energies=[EnergyType.PSYCHIC] * 3))
        p = make_policy(mine=mine, opp=opp, first_player=1, your_index=0, turn=3)
        self.assertEqual(p._desired_route_depth(), 3)
        # Only one held Alakazam makes the Kadabra a represented backup; the Abra is incomplete.
        self.assertEqual(p._independent_route_count(), 2)
        self.assertTrue(p._needs_route_depth())

    def test_extra_abra_is_high_value_when_route_depth_is_short(self):
        ala = Pokemon(743, energies=[EnergyType.PSYCHIC])
        mine = make_player(active=ala, bench=[Pokemon(742)], hand=[Card(741), Card(743)])
        opp = make_player(active=Pokemon(999))
        p = make_policy(mine=mine, opp=opp, first_player=1, your_index=0, turn=3)
        self.assertGreater(p._score_play_poke(Card(741)), 20000)

    def test_shaymin_is_conditional_bench_protection(self):
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]),
                           bench=[Pokemon(741)])
        threatening = make_player(active=Pokemon(999, energies=[EnergyType.PSYCHIC] * 3))
        p = make_policy(mine=mine, opp=threatening)
        self.assertTrue(p._shaymin_worthwhile())
        self.assertGreater(p._score_play_poke(Card(343)), 19000)
        quiet = make_player(active=Pokemon(741))
        p2 = make_policy(mine=mine, opp=quiet)
        self.assertFalse(p2._shaymin_worthwhile())
        self.assertLess(p2._score_play_poke(Card(343)), 1000)

    def test_fez_is_real_fallback_attacker(self):
        fez = Pokemon(140, hp=210, energies=[EnergyType.PSYCHIC,
                                             EnergyType.COLORLESS,
                                             EnergyType.COLORLESS])
        opp_active = Pokemon(741, hp=90)
        attack = opt(OptionType.ATTACK, attack_id=183)
        mine = make_player(active=fez, prize_count=1)
        opp = make_player(active=opp_active)
        p = make_policy(mine=mine, opp=opp, options=[attack])
        self.assertEqual(p._attack_damage_for_option(attack), 100)
        self.assertEqual(p._state, main.TurnState.ENDGAME)
        self.assertGreaterEqual(p._score_attack(attack), 90000)

    def test_fez_ability_is_allowed_before_attack_when_safe(self):
        ala = Pokemon(743, energies=[EnergyType.PSYCHIC])
        fez = Pokemon(140)
        ability = opt(OptionType.ABILITY, area=AreaType.BENCH, index=0)
        attack = opt(OptionType.ATTACK, attack_id=1072)
        mine = make_player(active=ala, bench=[fez, Pokemon(742)],
                           hand=[Card(743), Card(5)], deck_count=30)
        opp = make_player(active=Pokemon(741, hp=200))
        p = make_policy(mine=mine, opp=opp, options=[ability, attack])
        self.assertGreater(p._score_ability(ability), 15000)
        self.assertTrue(p._improves_plan(ability))

    def test_partial_fez_funding_is_supported_but_not_before_primary(self):
        ala = Pokemon(743, energies=[EnergyType.PSYCHIC])
        fez = Pokemon(140, hp=210, energies=[EnergyType.PSYCHIC, EnergyType.COLORLESS])
        energy = Card(13)
        attach = opt(OptionType.ATTACH, index=0, in_area=AreaType.BENCH, in_index=0)
        mine = make_player(active=ala, bench=[fez, Pokemon(742, energies=[EnergyType.PSYCHIC])],
                           hand=[energy, Card(743)])
        opp = make_player(active=Pokemon(741))
        p = make_policy(mine=mine, opp=opp, options=[attach])
        self.assertGreater(p._score_attach(attach), 8000)

    def test_hammer_tempo_requires_two_energy_deficit(self):
        ala = Pokemon(743, energies=[EnergyType.PSYCHIC])
        attack = opt(OptionType.ATTACK, attack_id=1072)
        special = Card(2000)
        enemy = Pokemon(999, energies=[EnergyType.PSYCHIC], energy_cards=[special])
        mine = make_player(active=ala, hand=[Card(1081), Card(741), Card(742)])
        opp = make_player(active=enemy)
        p = make_policy(mine=mine, opp=opp, options=[attack])
        self.assertIn(2000, p._hammer_tempo_targets())
        self.assertTrue(p._enhanced_hammer_worthwhile())

    def test_cage_is_redundant_when_shaymin_is_present(self):
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]),
                           bench=[Pokemon(343), Pokemon(741)])
        opp = make_player(active=Pokemon(999, energies=[EnergyType.PSYCHIC] * 3))
        p = make_policy(mine=mine, opp=opp)
        self.assertFalse(p._battle_cage_worthwhile())

    def test_hand_only_three_stage_line_is_not_complete(self):
        mine = make_player(active=Pokemon(305),
                           hand=[Card(741), Card(742), Card(743), Card(5)])
        opp = make_player(active=Pokemon(741))
        p = make_policy(mine=mine, opp=opp)
        self.assertFalse(p._holds_complete_route())

    def test_in_play_kadabra_plus_alakazam_and_energy_is_complete(self):
        mine = make_player(active=Pokemon(305), bench=[Pokemon(742)],
                           hand=[Card(743), Card(5)])
        opp = make_player(active=Pokemon(741))
        p = make_policy(mine=mine, opp=opp)
        self.assertTrue(p._holds_complete_route())

    def test_battle_cage_cannot_erase_twenty_damage_attack(self):
        cage = Card(1264)
        play = opt(OptionType.PLAY, index=0)
        attack = opt(OptionType.ATTACK, attack_id=1072)
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]), hand=[cage])
        opp = make_player(active=Pokemon(741, hp=100))
        p = make_policy(mine=mine, opp=opp, options=[play, attack])
        self.assertFalse(p._preserves_attack(play))

    def test_non_mirror_xerosic_is_blocked(self):
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]),
                           hand=[Card(1197), Card(741), Card(742)])
        opp = make_player(active=Pokemon(998), hand=[Card(1)] * 8)
        p = make_policy(mine=mine, opp=opp)
        p.opponent.handCount = 8
        self.assertEqual(p._score_play_trainer(Card(1197)), -1)

    def test_attackable_end_is_blocked(self):
        attack = opt(OptionType.ATTACK, attack_id=1072)
        end = opt(OptionType.END)
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]),
                           hand=[Card(741), Card(742), Card(743)])
        opp = make_player(active=Pokemon(741, hp=100))
        p = make_policy(mine=mine, opp=opp, options=[attack, end])
        self.assertLess(p._score_main(end, 0), 0)
        self.assertGreater(p._score_main(attack, p._score_attack(attack)), 0)

    def test_no_extra_abra_after_body_depth_is_filled(self):
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]),
                           bench=[Pokemon(742), Pokemon(741)], hand=[Card(741), Card(743)])
        opp = make_player(active=Pokemon(741))
        p = make_policy(mine=mine, opp=opp, first_player=0, your_index=0, turn=5)
        self.assertFalse(p._needs_more_abra_body())
        self.assertLess(p._score_play_poke(Card(741)), 2000)

    def test_fez_not_deployed_without_a_missing_role(self):
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]),
                           bench=[Pokemon(742, energies=[EnergyType.PSYCHIC])], prize_count=6)
        opp = make_player(active=Pokemon(741))
        p = make_policy(mine=mine, opp=opp, first_player=0, your_index=0, turn=5)
        self.assertFalse(p._fezandipiti_worthwhile())
        self.assertLess(p._score_play_poke(Card(140)), 1000)

    def test_hammer_does_not_fire_for_one_energy_deficit(self):
        ala = Pokemon(743, energies=[EnergyType.PSYCHIC])
        attack = opt(OptionType.ATTACK, attack_id=1072)
        special = Card(2000)
        basic = Card(5)
        enemy = Pokemon(998, energies=[EnergyType.PSYCHIC, EnergyType.PSYCHIC],
                        energy_cards=[special, basic])
        mine = make_player(active=ala, hand=[Card(1081), Card(741)])
        opp = make_player(active=enemy)
        p = make_policy(mine=mine, opp=opp, options=[attack])
        self.assertNotIn(2000, p._hammer_tempo_targets())

    def test_second_stretcher_requires_direct_improvement(self):
        mine = make_player(active=Pokemon(743, energies=[EnergyType.PSYCHIC]),
                           discard=[Card(741)], hand=[Card(1097)])
        opp = make_player(active=Pokemon(741))
        p = make_policy(mine=mine, opp=opp, turn=4)
        old = dict(main._TURN_STATE)
        try:
            main._TURN_STATE.update({"turn": 4, "night_stretcher": 1})
            self.assertEqual(p._night_stretcher_allowed_targets(), [])
        finally:
            main._TURN_STATE.clear()
            main._TURN_STATE.update(old)

    def test_fez_funding_waits_for_primary_route(self):
        fez = Pokemon(140, energies=[EnergyType.PSYCHIC, EnergyType.COLORLESS])
        energy = Card(13)
        attach = opt(OptionType.ATTACH, index=0, in_area=AreaType.BENCH, in_index=0)
        mine = make_player(active=Pokemon(741), bench=[fez], hand=[energy])
        opp = make_player(active=Pokemon(741))
        p = make_policy(mine=mine, opp=opp, options=[attach])
        self.assertEqual(p._score_attach(attach), -1)

    def test_no_duplicate_function_definitions(self):
        import ast
        tree = ast.parse((ROOT / "main.py").read_text())
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        dup = [name for name, count in Counter(names).items() if count > 1]
        self.assertEqual(dup, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
