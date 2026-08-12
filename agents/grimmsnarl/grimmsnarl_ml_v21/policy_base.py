"""Shared, self-contained policy primitives for competition agents.

This copy is bundled beside ``main.py`` so the runtime does not accidentally
import another agent's policy module.  It provides type-aware energy payment,
legal selection normalization, robust observation fallbacks and conservative
prize-card deduction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType, Pokemon,
    SelectContext, all_card_data, all_attack, to_observation_class,
)

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_table = {a.attackId: a for a in all_attack()}

ATTACK_COST_ENERGIES = {a.attackId: list(a.energies or []) for a in all_attack()}
SELF_SCALING_ATTACKS = set()
for _a in all_attack():
    _text = (_a.text or "").lower()
    if "for each" in _text and "energy attached to this" in _text:
        SELF_SCALING_ATTACKS.add(_a.attackId)

ENERGY_PROVIDES = {}
for _c in all_card:
    if _c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        ENERGY_PROVIDES[_c.cardId] = getattr(_c, "energyType", EnergyType.COLORLESS)

EFFECT_PREVENT_ENERGY = set()
EFFECT_PREVENT_SELF = set()
for _c in all_card:
    for _skill in (_c.skills or []):
        _text = (_skill.text or "")
        if "effects of attacks" in _text and "prevent" in _text.lower():
            if _c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                EFFECT_PREVENT_ENERGY.add(_c.cardId)
            elif "to this Pokémon" in _text or "to this Pok" in _text:
                EFFECT_PREVENT_SELF.add(_c.cardId)


def normalize_selection(ranked, scores, select):
    n = len(select.option)
    min_count = max(0, min(select.minCount, n))
    max_count = max(min_count, min(select.maxCount, n))
    result, seen = [], set()
    for index in ranked:
        if not (0 <= index < n) or index in seen:
            continue
        score = scores[index] if index < len(scores) else 0
        if score > 0 or len(result) < min_count:
            result.append(index)
            seen.add(index)
        if len(result) >= max_count:
            break
    for index in range(n):
        if len(result) >= min_count:
            break
        if index not in seen:
            result.append(index)
            seen.add(index)
    return result


def legal_fallback(select):
    try:
        count = min(max(0, select.minCount), len(select.option))
        return list(range(count))
    except Exception:
        return []


def legal_fallback_from_dict(obs_dict):
    try:
        selection = obs_dict.get("select") or {}
        count = min(max(0, selection.get("minCount", 0)), len(selection.get("option") or []))
        return list(range(count))
    except Exception:
        return []


def _safe_get(sequence, index):
    try:
        if sequence is None or index is None or index < 0 or index >= len(sequence):
            return None
        return sequence[index]
    except Exception:
        return None


def get_card(obs, area, index, player_index):
    try:
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            return _safe_get(getattr(obs.select, "deck", None), index)
        if area == AreaType.HAND:
            return _safe_get(getattr(player, "hand", None), index)
        if area == AreaType.DISCARD:
            return _safe_get(getattr(player, "discard", None), index)
        if area == AreaType.ACTIVE:
            return _safe_get(getattr(player, "active", None), index)
        if area == AreaType.BENCH:
            return _safe_get(getattr(player, "bench", None), index)
        if area == AreaType.PRIZE:
            return _safe_get(getattr(player, "prize", None), index)
        if area == AreaType.STADIUM:
            return _safe_get(getattr(obs.current, "stadium", None), index)
        if area == AreaType.LOOKING:
            return _safe_get(getattr(obs.current, "looking", None), index)
        return None
    except Exception:
        return None


def prize_count(pokemon):
    data = card_table.get(getattr(pokemon, "id", None))
    if data is None:
        return 1
    return 3 if getattr(data, "megaEx", False) else 2 if getattr(data, "ex", False) else 1


class PrizeTracker:
    """Conservatively infer our face-down prizes only during an exact deck search."""

    def __init__(self, decklist):
        self._decklist = list(decklist)
        self._deck_total = Counter(self._decklist)
        self._prized = None
        self._last_prize_count = None
        self._last_hand_by_serial = {}

    def deck_total(self, card_id):
        return self._deck_total.get(card_id, 0)

    def update(self, obs, obs_dict=None):
        player_index = obs.current.yourIndex
        player = obs.current.players[player_index]
        prize_total = len(player.prize)
        hand_by_serial = {
            c.serial: c.id for c in (player.hand or [])
            if c is not None and getattr(c, "serial", None) is not None
        }
        if self._prized is not None and self._last_prize_count is not None and prize_total < self._last_prize_count:
            taken = self._last_prize_count - prize_total
            card_ids = self._prize_to_hand(obs_dict, player_index)
            if len(card_ids) != taken:
                card_ids = [cid for serial, cid in hand_by_serial.items() if serial not in self._last_hand_by_serial]
            if len(card_ids) != taken or not self._remove(card_ids):
                self._prized = None
        self._last_prize_count = prize_total
        self._last_hand_by_serial = hand_by_serial
        if self._prized is not None:
            return
        visible_deck = getattr(obs.select, "deck", None) if obs.select is not None else None
        if visible_deck is None or len(visible_deck) != player.deckCount:
            return
        inferred = self._deduce(obs, player, player_index)
        if inferred is not None:
            self._prized = inferred

    def _deduce(self, obs, player, player_index):
        remaining = Counter(self._decklist)

        def subtract(card):
            if card is not None:
                remaining[card.id] -= 1

        for card in obs.select.deck:
            subtract(card)
        for card in player.hand or []:
            subtract(card)
        for pokemon in list(player.active or []) + list(player.bench or []):
            if pokemon is None:
                continue
            subtract(pokemon)
            for card in getattr(pokemon, "preEvolution", None) or []:
                subtract(card)
            for card in getattr(pokemon, "energyCards", None) or []:
                subtract(card)
            for card in getattr(pokemon, "tools", None) or []:
                subtract(card)
        for card in player.discard or []:
            subtract(card)
        for card in obs.current.stadium or []:
            if card is not None and getattr(card, "playerIndex", None) == player_index:
                remaining[card.id] -= 1
        effect = getattr(obs.select, "effect", None)
        if effect is not None and getattr(effect, "playerIndex", None) == player_index and remaining.get(effect.id, 0) > 0:
            remaining[effect.id] -= 1
        if any(value < 0 for value in remaining.values()):
            return None
        inferred = Counter({card_id: value for card_id, value in remaining.items() if value > 0})
        return inferred if sum(inferred.values()) == len(player.prize) else None

    def _remove(self, card_ids):
        requested = Counter(card_ids)
        if any(self._prized.get(card_id, 0) < count for card_id, count in requested.items()):
            return False
        self._prized.subtract(requested)
        self._prized += Counter()
        return True

    @staticmethod
    def _prize_to_hand(obs_dict, player_index):
        if not isinstance(obs_dict, dict):
            return []
        return [
            log["cardId"] for log in obs_dict.get("logs", [])
            if log.get("playerIndex") == player_index
            and log.get("fromArea") in (6, "PRIZE", "Prize")
            and log.get("toArea") in (2, "HAND", "Hand")
            and log.get("cardId") is not None
        ]

    def prized_count(self, card_id):
        return None if self._prized is None else self._prized.get(card_id, 0)


class BasePolicy(ABC):
    ENERGY_TYPES = set()
    ATTACKER_IDS = set()

    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
        self.tracker = None
        self.field = defaultdict(int)
        self.hand = defaultdict(int)
        self.discard = defaultdict(int)
        for pokemon in self.my_board():
            if pokemon is not None:
                self.field[pokemon.id] += 1
        for card in self.me.hand:
            self.hand[card.id] += 1
        for card in self.me.discard:
            self.discard[card.id] += 1

    def my_board(self):
        return list(self.me.active or []) + list(self.me.bench or [])

    def is_energy(self, card_id):
        data = card_table.get(card_id)
        return card_id in self.ENERGY_TYPES or (
            data is not None and data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
        )

    def energy_count(self, pokemon):
        return len(pokemon.energies or []) if pokemon is not None else 0

    @staticmethod
    def can_pay(attached, cost):
        have = Counter(attached)
        colorless = 0
        for required in cost:
            if required == EnergyType.COLORLESS:
                colorless += 1
            elif have.get(required, 0) > 0:
                have[required] -= 1
            else:
                return False
        return sum(have.values()) >= colorless

    def payable_attacks(self, pokemon):
        data = card_table.get(getattr(pokemon, "id", None))
        if data is None:
            return []
        attached = list(getattr(pokemon, "energies", None) or [])
        return [
            attack_id for attack_id in (data.attacks or [])
            if attack_id in ATTACK_COST_ENERGIES and self.can_pay(attached, ATTACK_COST_ENERGIES[attack_id])
        ]

    def can_attack(self, pokemon):
        return bool(self.payable_attacks(pokemon))

    def effect_prevented(self, target):
        if target is None or target.id in EFFECT_PREVENT_SELF:
            return target is not None and target.id in EFFECT_PREVENT_SELF
        return any(getattr(card, "id", None) in EFFECT_PREVENT_ENERGY for card in (getattr(target, "energyCards", None) or []))

    def rank(self):
        if not self.select.option or self.select.maxCount == 0:
            return [], []
        scores = [self.score(option) for option in self.select.option]
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return ranked, scores

    def choose(self):
        ranked, scores = self.rank()
        return normalize_selection(ranked, scores, self.select)

    def score(self, option):
        option_type = option.type
        if self.context == SelectContext.IS_FIRST:
            return 100 if (option_type == OptionType.YES) == bool(self.go_first()) else 0
        if self.context == SelectContext.MULLIGAN:
            return 0 if option_type == OptionType.YES else 100
        if option_type == OptionType.NUMBER:
            return option.number if option.number is not None else 0
        if option_type == OptionType.YES:
            return 1
        if option_type == OptionType.NO:
            return 0
        if option_type == OptionType.CARD:
            return self.score_card(option)
        if option_type == OptionType.PLAY:
            return self.score_play(option)
        if option_type in (OptionType.ENERGY, OptionType.ATTACH):
            return self.score_attach(option)
        if option_type == OptionType.EVOLVE:
            return self.score_evolve(option)
        if option_type == OptionType.ABILITY:
            return self.score_ability(option)
        if option_type == OptionType.RETREAT:
            return self.score_retreat()
        if option_type == OptionType.ATTACK:
            return self.score_attack(option)
        if option_type == OptionType.END:
            return 0
        return 0

    def score_play(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None:
            return 0
        return self.score_play_poke(card) if data.cardType == CardType.POKEMON else self.score_play_trainer(card)

    def score_card(self, option):
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        context = self.context
        if option.playerIndex == self.op_index and not isinstance(card, Pokemon):
            return self.score_opp_card(option, card)
        if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self.score_active_choice(option, card)
        if context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self.score_setup_active(card)
        if context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return self.score_to_bench(card)
        if context == SelectContext.TO_HAND:
            return self.score_to_hand(card)
        if context in (SelectContext.EVOLVES_TO, SelectContext.EVOLVES_FROM):
            return self.score_evolves_choice(card)
        if context == SelectContext.ATTACH_TO:
            if isinstance(card, Pokemon):
                return (200 if card.id in self.ATTACKER_IDS else 50) + self.energy_count(card) * 10
            return 100 if self.is_energy(card.id) else 10
        if context in (SelectContext.ATTACH_FROM, SelectContext.TO_HAND_ENERGY):
            return 100 if self.is_energy(card.id) else 10
        if context in (
            SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
            SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD,
        ):
            return self.score_discard(card)
        if context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.DAMAGE):
            if isinstance(card, Pokemon) and option.playerIndex == self.op_index:
                return self.score_spread_target(card)
            return 0
        if context in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE):
            return self.score_putback(card)
        return 0

    def score_opp_card(self, option, card):
        data = card_table.get(card.id)
        if data is not None and data.cardType in (CardType.SPECIAL_ENERGY, CardType.BASIC_ENERGY):
            return 500 if getattr(option, "inPlayArea", None) == AreaType.ACTIVE else 300
        return 50

    def score_evolves_choice(self, card):
        return 2000 if card is not None and card.id in self.ATTACKER_IDS else 1000

    def score_discard(self, card):
        if card is None:
            return 0
        if self.is_energy(card.id):
            return 20 if self.hand[card.id] >= 3 else -40
        return 60 if self.hand[card.id] >= 2 else 0

    def score_putback(self, card):
        if card is None:
            return 0
        return 60 if self.hand[card.id] >= 2 else 10

    def score_active_choice(self, option, card):
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex == self.op_index:
            return self.gust_value(card)
        return (200 if card.id in self.ATTACKER_IDS else 0) + self.energy_count(card) * 10 + max(1, card.hp // 30)

    def gust_value(self, card):
        return prize_count(card) * 1000 - int(getattr(card, "hp", 0) or 0) // 10

    def score_setup_active(self, card):
        return 30 if card is not None and card.id in self.ATTACKER_IDS else 5

    def score_to_bench(self, card):
        return 100 - 20 * self.field[card.id] if card is not None else 0

    def score_to_hand(self, card):
        return 200 - 40 * self.hand[card.id] if card is not None else 0

    def score_spread_target(self, card):
        hp = int(getattr(card, "hp", 0) or 0)
        return 4000 - hp * 12 + prize_count(card) * 200 + (1500 if hp <= 60 else 0)

    def score_ability(self, option):
        return 9000

    def score_retreat(self):
        return -1

    def score_attach(self, option):
        return -1

    @abstractmethod
    def go_first(self):
        raise NotImplementedError

    @abstractmethod
    def score_play_poke(self, card):
        raise NotImplementedError

    @abstractmethod
    def score_play_trainer(self, card):
        raise NotImplementedError

    @abstractmethod
    def score_evolve(self, option):
        raise NotImplementedError

    @abstractmethod
    def score_attack(self, option):
        raise NotImplementedError


def make_agent(policy_cls, deck, diag):
    state = {"turn": -1, "tracker": PrizeTracker(deck)}

    def record_error(error):
        key = type(error).__name__ + ": " + str(error)[:160]
        diag["errors"][key] = diag["errors"].get(key, 0) + 1

    def agent(obs_dict):
        try:
            if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
                diag["deck_returns"] += 1
                return deck
        except Exception:
            pass
        diag["decisions"] += 1
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                diag["deck_returns"] += 1
                diag["decisions"] -= 1
                return deck
            try:
                state["tracker"].update(obs, obs_dict)
            except Exception:
                pass
            try:
                policy = policy_cls(obs)
                policy.tracker = state["tracker"]
                selection = policy.choose()
                diag["policy_ok"] += 1
                return selection
            except Exception as error:
                record_error(error)
                diag["policy_fallback"] += 1
                return legal_fallback(obs.select)
        except Exception as error:
            record_error(error)
            diag["obs_fallback"] += 1
            return legal_fallback_from_dict(obs_dict if isinstance(obs_dict, dict) else {})

    return agent


def new_diag():
    return {
        "decisions": 0,
        "policy_ok": 0,
        "policy_fallback": 0,
        "obs_fallback": 0,
        "deck_returns": 0,
        "errors": {},
    }
