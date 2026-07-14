from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_cg_stub() -> None:
    cg = types.ModuleType("cg")
    api = types.ModuleType("cg.api")

    class Values:
        pass

    api.AreaType = Values()
    for i, name in enumerate(["DECK", "HAND", "DISCARD", "ACTIVE", "BENCH", "PRIZE", "STADIUM", "LOOKING"]):
        setattr(api.AreaType, name, i + 1)

    api.CardType = Values()
    api.CardType.POKEMON = 1
    api.CardType.BASIC_ENERGY = 2
    api.CardType.SPECIAL_ENERGY = 3

    api.EnergyType = Values()
    api.EnergyType.COLORLESS = 0

    api.OptionType = Values()
    for i, name in enumerate(["NUMBER", "YES", "NO", "CARD", "PLAY", "ENERGY", "ATTACH", "EVOLVE", "ABILITY", "RETREAT", "ATTACK", "END"]):
        setattr(api.OptionType, name, i)

    api.SelectContext = Values()
    for i, name in enumerate([
        "IS_FIRST", "MULLIGAN", "MAIN", "SWITCH", "TO_ACTIVE", "SETUP_ACTIVE_POKEMON",
        "SETUP_BENCH_POKEMON", "TO_BENCH", "TO_FIELD", "TO_HAND", "EVOLVES_TO",
        "EVOLVES_FROM", "ATTACH_TO", "ATTACH_FROM", "TO_HAND_ENERGY", "DISCARD",
        "DISCARD_CARD_OR_ATTACHED_CARD", "DISCARD_ENERGY", "DISCARD_ENERGY_CARD",
        "DAMAGE_COUNTER", "DAMAGE_COUNTER_ANY", "DAMAGE", "TO_DECK", "TO_DECK_BOTTOM", "TO_PRIZE",
    ]):
        setattr(api.SelectContext, name, i)

    class Card:
        def __init__(self, card_id=0):
            self.id = card_id

    class Pokemon(Card):
        pass

    class Observation:
        pass

    api.Card = Card
    api.Pokemon = Pokemon
    api.Observation = Observation

    class CardData:
        def __init__(self, card_id, card_type, attacks=(), ex=False, mega_ex=False):
            self.cardId = card_id
            self.cardType = card_type
            self.attacks = list(attacks)
            self.ex = ex
            self.megaEx = mega_ex
            self.skills = []
            self.energyType = 1 if card_id == 1 else 0

    class AttackData:
        def __init__(self, attack_id, energies, damage):
            self.attackId = attack_id
            self.energies = list(energies)
            self.damage = damage
            self.text = ""

    pokemon_ids = [400, 401, 414, 431, 434]
    trainer_ids = [1094, 1121, 1134, 1152, 1159, 1175, 1216, 1217, 1218, 1220, 1227, 1257]
    cards = [CardData(1, api.CardType.BASIC_ENERGY), CardData(15, api.CardType.SPECIAL_ENERGY)]
    cards += [CardData(cid, api.CardType.POKEMON, [aid], ex=(cid == 431)) for cid, aid in zip(pokemon_ids, [559, 560, 999, 608, 154])]
    cards += [CardData(cid, 9) for cid in trainer_ids]
    attacks = [
        AttackData(559, [1], 30),
        AttackData(560, [1, 0, 0], 0),
        AttackData(608, [0, 0], 160),
        AttackData(154, [0, 0], 200),
        AttackData(999, [1], 20),
    ]
    api.all_card_data = lambda: cards
    api.all_attack = lambda: attacks
    api.to_observation_class = lambda value: value

    cg.api = api
    sys.modules["cg"] = cg
    sys.modules["cg.api"] = api


class PackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_cg_stub()
        sys.path.insert(0, str(ROOT))
        sys.modules.pop("policy_base", None)
        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")

    def test_exact_sixty_card_deck(self) -> None:
        expected = Counter({
            1: 9, 15: 4, 400: 4, 401: 4, 414: 2, 431: 2, 434: 3,
            1094: 3, 1121: 1, 1134: 4, 1152: 4, 1159: 1, 1175: 1,
            1216: 4, 1217: 1, 1218: 3, 1220: 4, 1227: 3, 1257: 3,
        })
        self.assertEqual(len(self.main.MY_DECK), 60)
        self.assertEqual(Counter(self.main.MY_DECK), expected)
        self.assertNotIn(5, self.main.MY_DECK)

    def test_deck_handshake(self) -> None:
        self.assertEqual(self.main.agent({"select": None}), self.main.MY_DECK)

    def test_spidops_damage_model(self) -> None:
        self.assertEqual(self.main.SpidopsPolicy.spidops_damage_from_count(1), 30)
        self.assertEqual(self.main.SpidopsPolicy.spidops_damage_from_count(5), 150)
        self.assertEqual(self.main.SpidopsPolicy.spidops_damage_from_count(6), 180)

    def test_mewtwo_bonus_damage_model(self) -> None:
        self.assertEqual(self.main.SpidopsPolicy.mewtwo_damage_with_discards(0), 160)
        self.assertEqual(self.main.SpidopsPolicy.mewtwo_damage_with_discards(1), 220)
        self.assertEqual(self.main.SpidopsPolicy.mewtwo_damage_with_discards(2), 280)
        self.assertEqual(self.main.SpidopsPolicy.mewtwo_damage_with_discards(99), 280)

    def test_brave_bangle_spidops_bonus(self) -> None:
        policy = self.main.SpidopsPolicy.__new__(self.main.SpidopsPolicy)
        tool = types.SimpleNamespace(id=self.main.C.BRAVE_BANGLE)
        active = types.SimpleNamespace(id=self.main.C.SPIDOPS, tools=[tool])
        policy.me = types.SimpleNamespace(active=[active], bench=[], prize=[None] * 6)
        policy.field = Counter({self.main.C.SPIDOPS: 1})
        policy.my_board = lambda: [active] * 6
        target = types.SimpleNamespace(id=self.main.C.MEWTWO_EX, hp=380)
        self.assertEqual(policy.attack_damage(self.main.A.SPIDOPS, target), 210)


    def _mewtwo_bonus_policy(self, target_hp: int, energy_ids: list[int], prizes: int = 6):
        Pokemon = sys.modules["cg.api"].Pokemon
        OptionType = sys.modules["cg.api"].OptionType
        AreaType = sys.modules["cg.api"].AreaType

        def pokemon(card_id: int, energies: list[int] | None = None):
            p = Pokemon(card_id)
            p.energies = list(energies or [])
            p.energyCards = [types.SimpleNamespace(id=eid) for eid in (energies or [])]
            p.tools = []
            p.hp = 200
            p.maxHp = 200
            return p

        mewtwo = pokemon(self.main.C.MEWTWO_EX, [0, 0])
        bench = pokemon(self.main.C.SPIDOPS, energy_ids)
        target = pokemon(self.main.C.MEWTWO_EX, [])
        target.hp = target_hp
        me = types.SimpleNamespace(
            active=[mewtwo], bench=[bench], hand=[], discard=[], prize=[None] * prizes,
            handCount=0, deckCount=20,
        )
        opponent = types.SimpleNamespace(
            active=[target], bench=[], hand=[], discard=[], prize=[None] * 6,
            handCount=0, deckCount=20,
        )
        options = [
            types.SimpleNamespace(
                type=OptionType.ENERGY, area=AreaType.BENCH, index=0, energyIndex=i,
                playerIndex=0,
            )
            for i in range(len(energy_ids))
        ]
        select = types.SimpleNamespace(
            context=26, effect=types.SimpleNamespace(id=self.main.C.MEWTWO_EX),
            option=options, minCount=0, maxCount=min(2, len(options)),
        )
        state = types.SimpleNamespace(
            yourIndex=0, players=[me, opponent], stadium=[], supporterPlayed=False,
        )
        obs = types.SimpleNamespace(current=state, select=select)
        return self.main.SpidopsPolicy(obs)

    def test_mewtwo_bonus_chooses_minimum_recyclable_energy(self) -> None:
        self.assertEqual(self._mewtwo_bonus_policy(160, [1, 1]).choose(), [])
        self.assertEqual(self._mewtwo_bonus_policy(220, [1, 1]).choose(), [1])
        self.assertEqual(set(self._mewtwo_bonus_policy(280, [1, 1]).choose()), {0, 1})

    def test_mewtwo_bonus_preserves_rocket_energy_without_winning_ko(self) -> None:
        self.assertEqual(self._mewtwo_bonus_policy(220, [15]).choose(), [])
        # With one prize left, spending Rocket Energy is allowed to close the game.
        self.assertEqual(self._mewtwo_bonus_policy(220, [15], prizes=1).choose(), [0])


    def test_type_aware_energy_payment_and_wild_rocket_energy(self) -> None:
        Pokemon = sys.modules["cg.api"].Pokemon
        policy = self._mewtwo_bonus_policy(160, [1])

        tarountula = Pokemon(self.main.C.TAROUNTULA)
        tarountula.energyCards = [types.SimpleNamespace(id=self.main.C.GRASS)]
        tarountula.energies = [self.main.C.GRASS]
        self.assertTrue(policy.can_attack(tarountula))

        spidops = Pokemon(self.main.C.SPIDOPS)
        spidops.energyCards = [
            types.SimpleNamespace(id=self.main.C.GRASS),
            types.SimpleNamespace(id=self.main.C.ROCKET_ENERGY),
            types.SimpleNamespace(id=self.main.C.GRASS),
        ]
        spidops.energies = [self.main.C.GRASS, self.main.C.ROCKET_ENERGY, self.main.C.GRASS]
        self.assertTrue(policy.can_attack(spidops))

        mewtwo = Pokemon(self.main.C.MEWTWO_EX)
        mewtwo.energyCards = [
            types.SimpleNamespace(id=self.main.C.ROCKET_ENERGY),
            types.SimpleNamespace(id=self.main.C.GRASS),
        ]
        mewtwo.energies = [self.main.C.ROCKET_ENERGY, self.main.C.GRASS]
        self.assertTrue(policy.can_attack(mewtwo))

    def test_metadata_matches_source(self) -> None:
        meta = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["deck_source_submission_id"], 54603674)
        self.assertEqual(meta["comparison_submission_id"], 54613990)
        self.assertEqual(meta["deck_source_rating"], 1255.2)

    def test_required_files_exist(self) -> None:
        for filename in ("main.py", "policy_base.py", "deck.csv", "metadata.json", "STRATEGY.md", "LOG_ANALYSIS.md"):
            self.assertTrue((ROOT / filename).exists(), filename)

    def _bare_policy(self):
        policy = self.main.SpidopsPolicy.__new__(self.main.SpidopsPolicy)
        policy.field = Counter()
        policy.hand = Counter()
        policy.discard = Counter()
        policy.me = types.SimpleNamespace(
            active=[], bench=[], prize=[None] * 6, handCount=5, deckCount=30,
        )
        policy.opponent = types.SimpleNamespace(
            active=[], bench=[], prize=[None] * 6, handCount=5, deckCount=30,
        )
        policy.state = types.SimpleNamespace(supporterPlayed=False)
        policy.my_index = 0
        policy.op_index = 1
        policy.stadium_id = None
        policy.select = types.SimpleNamespace(effect=None, option=[])
        policy.low_deck = lambda: False
        policy.rocket_count = lambda: 4
        policy.tarountula_lines = lambda: 1
        policy.phase = lambda: self.main.Phase.PRESSURE
        policy.current_attack_ko = lambda target=None: False
        policy.active = lambda: None
        policy.opponent_active = lambda: None
        return policy

    def test_poke_pad_candidates_are_pokemon_scored_positive(self) -> None:
        policy = self._bare_policy()
        policy.select.effect = types.SimpleNamespace(id=self.main.C.POKE_PAD)
        policy.field[self.main.C.TAROUNTULA] = 1
        spidops = types.SimpleNamespace(id=self.main.C.SPIDOPS)
        tarountula = types.SimpleNamespace(id=self.main.C.TAROUNTULA)
        mimikyu = types.SimpleNamespace(id=self.main.C.MIMIKYU)
        articuno = types.SimpleNamespace(id=self.main.C.ARTICUNO)
        scores = [policy.score_to_hand(card) for card in (spidops, tarountula, mimikyu, articuno)]
        self.assertTrue(all(score > 0 for score in scores), scores)
        self.assertGreater(scores[0], scores[1])

    def test_poke_pad_is_legal_after_supporter(self) -> None:
        policy = self._bare_policy()
        policy.state.supporterPlayed = True
        score = policy.score_play_trainer(types.SimpleNamespace(id=self.main.C.POKE_PAD))
        self.assertGreater(score, self.main.Tier.ATTACK)

    def test_factory_large_hand_still_precedes_non_ko_attack(self) -> None:
        policy = self._bare_policy()
        policy.me.handCount = 9
        policy.obs = types.SimpleNamespace()
        factory = types.SimpleNamespace(id=self.main.C.FACTORY)
        option = types.SimpleNamespace()
        original_get_card = self.main.get_card
        try:
            self.main.get_card = lambda *args, **kwargs: factory
            score = policy.score_ability(option)
        finally:
            self.main.get_card = original_get_card
        self.assertGreater(score, self.main.Tier.ATTACK)

    def test_giovanni_declined_without_same_turn_ko(self) -> None:
        policy = self._bare_policy()
        policy.opponent.bench = [types.SimpleNamespace(id=self.main.C.TAROUNTULA, hp=100)]
        policy.gust_value = lambda card: -1
        self.assertEqual(
            policy.score_play_trainer(types.SimpleNamespace(id=self.main.C.GIOVANNI)),
            -1,
        )
        policy.gust_value = lambda card: self.main.Tier.KO + 20_000
        self.assertGreater(
            policy.score_play_trainer(types.SimpleNamespace(id=self.main.C.GIOVANNI)),
            self.main.Tier.KO,
        )

    def test_transceiver_prefers_ariana_on_developed_board(self) -> None:
        policy = self._bare_policy()
        policy.me.handCount = 5
        policy.rocket_count = lambda: 5
        policy.tarountula_lines = lambda: 2
        self.assertGreater(
            policy.supporter_search_value(self.main.C.ARIANA),
            policy.supporter_search_value(self.main.C.PROTON),
        )

    def test_transceiver_prefers_proton_when_setup_is_missing(self) -> None:
        policy = self._bare_policy()
        policy.me.handCount = 5
        policy.rocket_count = lambda: 2
        policy.tarountula_lines = lambda: 0
        self.assertGreater(
            policy.supporter_search_value(self.main.C.PROTON),
            policy.supporter_search_value(self.main.C.ARIANA),
        )

    def test_setup_active_prefers_mimikyu_over_tarountula(self) -> None:
        policy = self._bare_policy()
        self.assertGreater(
            policy.score_setup_active(types.SimpleNamespace(id=self.main.C.MIMIKYU)),
            policy.score_setup_active(types.SimpleNamespace(id=self.main.C.TAROUNTULA)),
        )

    def test_mimikyu_pivots_to_ready_spidops(self) -> None:
        policy = self._bare_policy()
        mimikyu = types.SimpleNamespace(id=self.main.C.MIMIKYU)
        spidops = types.SimpleNamespace(id=self.main.C.SPIDOPS)
        target = types.SimpleNamespace(id=self.main.C.TAROUNTULA, hp=200)
        policy.me.active = [mimikyu]
        policy.me.bench = [spidops]
        policy.opponent.active = [target]
        policy.active = lambda: mimikyu
        policy.opponent_active = lambda: target
        policy.can_attack = lambda card: card is spidops
        policy.best_attack_damage = lambda card, target=None: 150 if card is spidops else 0
        self.assertGreater(policy.score_retreat(), self.main.Tier.ATTACK)

    def test_low_deck_guard_is_only_two_cards(self) -> None:
        policy = self._bare_policy()
        self.assertEqual(policy.deck_floor(), 2)

    def test_evolution_never_outranks_existing_ko_tier(self) -> None:
        policy = self._bare_policy()
        source = types.SimpleNamespace(id=self.main.C.SPIDOPS)
        target = types.SimpleNamespace(id=self.main.C.TAROUNTULA, energies=[1, 1, 1])
        option = types.SimpleNamespace(index=0, inPlayArea=sys.modules["cg.api"].AreaType.ACTIVE, inPlayIndex=0)
        original_get_card = self.main.get_card
        try:
            self.main.get_card = lambda obs, area, index, player: source if area == sys.modules["cg.api"].AreaType.HAND else target
            policy.obs = types.SimpleNamespace()
            policy.my_index = 0
            score = policy.score_evolve(option)
        finally:
            self.main.get_card = original_get_card
        self.assertLess(score, self.main.Tier.KO)

    def test_mewtwo_damage_is_zero_into_crustle(self) -> None:
        policy = self._bare_policy()
        mewtwo = types.SimpleNamespace(id=self.main.C.MEWTWO_EX)
        crustle = types.SimpleNamespace(id=self.main.C.CRUSTLE, hp=200)
        self.assertEqual(
            policy.attack_damage_for_pokemon(mewtwo, self.main.A.MEWTWO_EX, crustle),
            0,
        )

    def test_mewtwo_is_not_developed_into_visible_crustle(self) -> None:
        policy = self._bare_policy()
        crustle = types.SimpleNamespace(id=self.main.C.CRUSTLE)
        policy.opponent.active = [crustle]
        policy.active = lambda: None
        self.assertEqual(
            policy.score_play_poke(types.SimpleNamespace(id=self.main.C.MEWTWO_EX)),
            -1,
        )

    def test_articuno_priority_rises_against_alakazam(self) -> None:
        policy = self._bare_policy()
        policy.opponent.active = [types.SimpleNamespace(id=self.main.C.ALAKAZAM)]
        articuno = policy.score_play_poke(types.SimpleNamespace(id=self.main.C.ARTICUNO))
        mimikyu = policy.score_play_poke(types.SimpleNamespace(id=self.main.C.MIMIKYU))
        self.assertGreater(articuno, mimikyu)

    def test_second_tarountula_is_high_priority_against_fast_pressure(self) -> None:
        policy = self._bare_policy()
        policy.field[self.main.C.TAROUNTULA] = 1
        policy.opponent.active = [types.SimpleNamespace(id=self.main.C.MEGA_LUCARIO_EX)]
        policy.tarountula_lines = lambda: 1
        score = policy.score_play_poke(types.SimpleNamespace(id=self.main.C.TAROUNTULA))
        self.assertGreater(score, self.main.Tier.PRE_ATTACK)

    def _main_choice_policy(self, target_hp: int):
        api = sys.modules["cg.api"]
        active = api.Pokemon(self.main.C.SPIDOPS)
        active.energyCards = [
            types.SimpleNamespace(id=self.main.C.GRASS),
            types.SimpleNamespace(id=self.main.C.ROCKET_ENERGY),
            types.SimpleNamespace(id=self.main.C.GRASS),
        ]
        active.energies = [self.main.C.GRASS, self.main.C.ROCKET_ENERGY, self.main.C.GRASS]
        active.tools = []
        active.hp = 120
        bench = []
        for cid in (self.main.C.TAROUNTULA, self.main.C.ARTICUNO, self.main.C.MIMIKYU, self.main.C.MEWTWO_EX):
            p = api.Pokemon(cid)
            p.energyCards = []
            p.energies = []
            p.tools = []
            p.hp = 100
            bench.append(p)
        target = api.Pokemon(self.main.C.TAROUNTULA)
        target.energyCards = []
        target.energies = []
        target.tools = []
        target.hp = target_hp
        me = types.SimpleNamespace(
            active=[active], bench=bench, hand=[api.Card(self.main.C.POKE_PAD)], discard=[],
            prize=[None] * 6, handCount=1, deckCount=30,
        )
        opponent = types.SimpleNamespace(
            active=[target], bench=[], hand=[], discard=[], prize=[None] * 6,
            handCount=0, deckCount=30,
        )
        options = [
            types.SimpleNamespace(type=api.OptionType.PLAY, index=0),
            types.SimpleNamespace(type=api.OptionType.ATTACK, attackId=self.main.A.SPIDOPS),
        ]
        select = types.SimpleNamespace(
            context=api.SelectContext.MAIN, effect=None, option=options, minCount=1, maxCount=1,
        )
        state = types.SimpleNamespace(
            yourIndex=0, players=[me, opponent], stadium=[], supporterPlayed=False,
        )
        return self.main.SpidopsPolicy(types.SimpleNamespace(current=state, select=select))

    def test_main_phase_uses_engine_before_non_ko_attack(self) -> None:
        policy = self._main_choice_policy(target_hp=200)
        self.assertEqual(policy.choose(), [0])

    def test_main_phase_keeps_current_ko_over_engine(self) -> None:
        policy = self._main_choice_policy(target_hp=150)
        self.assertEqual(policy.choose(), [1])


if __name__ == "__main__":
    unittest.main()
