"""Shared, deck-agnostic policy scaffolding for Pokémon TCG AI Battle agents.

This file intentionally contains only mechanics that should be common to every deck:
legal selection normalization, card lookup, type-aware attack payment, conservative energy
attachment, generic sub-selection scoring, diagnostics, and a crash-safe agent wrapper.
Deck strategy belongs in ``main.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Iterable

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)

ALL_CARDS = all_card_data()
CARD_TABLE = {c.cardId: c for c in ALL_CARDS}
ALL_ATTACKS = all_attack()
ATTACK_TABLE = {a.attackId: a for a in ALL_ATTACKS}
ATTACK_COST_ENERGIES = {
    a.attackId: list(getattr(a, "energies", None) or []) for a in ALL_ATTACKS
}
ENERGY_PROVIDES = {
    c.cardId: getattr(c, "energyType", EnergyType.COLORLESS)
    for c in ALL_CARDS
    if getattr(c, "cardType", None) in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
}


def _safe_get(seq: Any, index: int | None) -> Any:
    try:
        if seq is None or index is None or index < 0 or index >= len(seq):
            return None
        return seq[index]
    except Exception:
        return None


def get_card(obs: Observation, area: Any, index: int | None, player_index: int) -> Any:
    """Return the card addressed by an engine option, or ``None`` on malformed input."""
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
    except Exception:
        return None
    return None


def prize_count(pokemon: Any) -> int:
    data = CARD_TABLE.get(getattr(pokemon, "id", None))
    if data is None:
        return 1
    if getattr(data, "megaEx", False):
        return 3
    if getattr(data, "ex", False):
        return 2
    return 1


def legal_fallback(select: Any) -> list[int]:
    try:
        count = min(max(0, int(select.minCount)), len(select.option or []))
        return list(range(count))
    except Exception:
        return []


def legal_fallback_from_dict(obs_dict: dict[str, Any]) -> list[int]:
    try:
        select = obs_dict.get("select") or {}
        options = select.get("option") or []
        count = min(max(0, int(select.get("minCount", 0))), len(options))
        return list(range(count))
    except Exception:
        return []


def normalize_selection(ranked: list[int], scores: list[float], select: Any) -> list[int]:
    """Build a legal list of option indices.

    Required choices are always filled. Optional choices are included only while their score is
    positive, up to ``maxCount``. This prevents optional search effects from taking junk cards.
    """
    try:
        n = len(select.option or [])
        minimum = max(0, min(int(select.minCount), n))
        maximum = max(minimum, min(int(select.maxCount), n))
        chosen: list[int] = []
        for index in ranked:
            if index < 0 or index >= n or index in chosen:
                continue
            if len(chosen) < minimum or (len(chosen) < maximum and scores[index] > 0):
                chosen.append(index)
            if len(chosen) >= maximum:
                break
        if len(chosen) < minimum:
            for index in range(n):
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) >= minimum:
                    break
        return chosen
    except Exception:
        return legal_fallback(select)


def new_diag() -> dict[str, Any]:
    return {
        "decisions": 0,
        "policy_ok": 0,
        "policy_fallback": 0,
        "obs_fallback": 0,
        "deck_returns": 0,
        "errors": {},
    }


def diag_snapshot(diag: dict[str, Any]) -> dict[str, Any]:
    result = {k: (dict(v) if isinstance(v, dict) else v) for k, v in diag.items()}
    result["fallback_rate"] = (
        result.get("policy_fallback", 0) + result.get("obs_fallback", 0)
    ) / max(1, result.get("decisions", 0))
    return result


class BasePolicy(ABC):
    """Mechanics-safe base class. Subclasses provide deck-specific priorities."""

    ENERGY_TYPES: set[int] = set()
    WILD_ENERGY_IDS: set[int] = set()  # special energies that can satisfy any typed requirement
    ATTACKER_IDS: set[int] = set()

    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else None
        self.field: defaultdict[int, int] = defaultdict(int)
        self.hand: defaultdict[int, int] = defaultdict(int)
        self.discard: defaultdict[int, int] = defaultdict(int)
        for p in self.my_board():
            if p is not None:
                self.field[p.id] += 1
        for c in self.me.hand or []:
            self.hand[c.id] += 1
        for c in self.me.discard or []:
            self.discard[c.id] += 1

    def my_board(self) -> list[Any]:
        return list(self.me.active or []) + list(self.me.bench or [])

    def is_energy(self, card_id: int) -> bool:
        data = CARD_TABLE.get(card_id)
        return card_id in self.ENERGY_TYPES or (
            data is not None
            and data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
        )

    def attached_energy_cards(self, pokemon: Any) -> list[Any]:
        cards = list(getattr(pokemon, "energyCards", None) or [])
        if cards:
            return cards
        # Compatibility fallback for simplified test states that only expose ``energies``.
        return [type("EnergyRef", (), {"id": value})() for value in (getattr(pokemon, "energies", None) or [])]

    def can_pay(self, pokemon: Any, cost: Iterable[Any]) -> bool:
        """Type-aware attack payment from attached energy-card IDs.

        Raw competition observations expose values such as ``[1, 15]`` in ``pokemon.energies``;
        those are card IDs, not always EnergyType values. Mapping through card data avoids the
        common bug where Team Rocket's Energy (ID 15) is ignored. Decks may mark special Energy
        as wild through ``WILD_ENERGY_IDS``.
        """
        required_cost = list(cost or [])
        typed = Counter()
        wild = 0
        total = 0
        for card in self.attached_energy_cards(pokemon):
            card_id = getattr(card, "id", card)
            total += 1
            if card_id in self.WILD_ENERGY_IDS:
                wild += 1
            else:
                typed[ENERGY_PROVIDES.get(card_id, card_id)] += 1

        colorless = 0
        for required in required_cost:
            if required == EnergyType.COLORLESS:
                colorless += 1
            elif typed.get(required, 0) > 0:
                typed[required] -= 1
            elif wild > 0:
                wild -= 1
            else:
                return False
        return sum(typed.values()) + wild >= colorless and total >= len(required_cost)

    def payable_attacks(self, pokemon: Any) -> list[int]:
        data = CARD_TABLE.get(getattr(pokemon, "id", None))
        if data is None:
            return []
        return [
            attack_id
            for attack_id in (getattr(data, "attacks", None) or [])
            if attack_id in ATTACK_COST_ENERGIES
            and self.can_pay(pokemon, ATTACK_COST_ENERGIES[attack_id])
        ]

    def can_attack(self, pokemon: Any) -> bool:
        return bool(self.payable_attacks(pokemon))

    def have_ready_attacker(self) -> bool:
        return any(
            p is not None and p.id in self.ATTACKER_IDS and self.can_attack(p)
            for p in self.my_board()
        )

    def bench_attacker_ready(self) -> bool:
        return any(
            p is not None and p.id in self.ATTACKER_IDS and self.can_attack(p)
            for p in (self.me.bench or [])
        )

    def rank(self) -> tuple[list[int], list[float]]:
        if not self.select.option or self.select.maxCount == 0:
            return [], []
        scores = [float(self.score(option)) for option in self.select.option]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked, scores

    def choose(self) -> list[int]:
        ranked, scores = self.rank()
        return normalize_selection(ranked, scores, self.select)

    def score(self, option: Any) -> float:
        t = option.type
        if self.context == SelectContext.IS_FIRST:
            return 100 if (t == OptionType.YES) == bool(self.go_first()) else 0
        if self.context == SelectContext.MULLIGAN:
            return 0 if t == OptionType.YES else 100
        if t == OptionType.NUMBER:
            return getattr(option, "number", 0) or 0
        if t == OptionType.YES:
            return 1
        if t == OptionType.NO:
            return 0
        if t == OptionType.CARD:
            return self.score_card(option)
        if t == OptionType.PLAY:
            return self.score_play(option)
        if t in (OptionType.ENERGY, OptionType.ATTACH):
            return self.score_attach(option)
        if t == OptionType.EVOLVE:
            return self.score_evolve(option)
        if t == OptionType.ABILITY:
            return self.score_ability(option)
        if t == OptionType.RETREAT:
            return self.score_retreat()
        if t == OptionType.ATTACK:
            return self.score_attack(option)
        if t == OptionType.END:
            return self.score_end()
        return 0

    def score_play(self, option: Any) -> float:
        card = get_card(self.obs, AreaType.HAND, getattr(option, "index", None), self.my_index)
        if card is None:
            return -1
        data = CARD_TABLE.get(card.id)
        if isinstance(card, Pokemon) or (data is not None and data.cardType == CardType.POKEMON):
            return self.score_play_poke(card)
        return self.score_play_trainer(card)

    def score_attach(self, option: Any) -> float:
        target = get_card(
            self.obs,
            getattr(option, "inPlayArea", None),
            getattr(option, "inPlayIndex", None),
            self.my_index,
        )
        source = get_card(self.obs, AreaType.HAND, getattr(option, "index", None), self.my_index)
        if not isinstance(target, Pokemon) or source is None:
            return -1
        if not self.is_energy(source.id):
            return -1
        if self.can_attack(target):
            return -1
        return 8_000 if target.id in self.ATTACKER_IDS else 1_000

    def score_card(self, option: Any) -> float:
        card = get_card(
            self.obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", self.my_index),
        )
        if card is None:
            return 0
        context = self.context
        if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self.score_active_choice(option, card)
        if context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self.score_setup_active(card)
        if context in (
            SelectContext.SETUP_BENCH_POKEMON,
            SelectContext.TO_BENCH,
            SelectContext.TO_FIELD,
        ):
            return self.score_to_bench(card)
        if context == SelectContext.TO_HAND:
            return self.score_to_hand(card)
        if context in (SelectContext.EVOLVES_TO, SelectContext.EVOLVES_FROM):
            return self.score_evolves_choice(card)
        if context == SelectContext.ATTACH_TO:
            if isinstance(card, Pokemon):
                return self.score_attach_target(option, card)
            return 100 if self.is_energy(card.id) else 10
        if context in (SelectContext.ATTACH_FROM, SelectContext.TO_HAND_ENERGY):
            return 100 if self.is_energy(card.id) else 0
        discard_contexts = (
            SelectContext.DISCARD,
            SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
            SelectContext.DISCARD_ENERGY,
            SelectContext.DISCARD_ENERGY_CARD,
        )
        if context in discard_contexts:
            return self.score_discard(option, card)
        if context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.DAMAGE):
            return self.score_damage_target(option, card)
        if context in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE):
            return self.score_putback(card)
        return 0

    def score_attach_target(self, option: Any, card: Any) -> float:
        return 8_000 if card.id in self.ATTACKER_IDS and not self.can_attack(card) else 1_000

    def score_active_choice(self, option: Any, card: Any) -> float:
        player_index = getattr(option, "playerIndex", self.my_index)
        if player_index == self.op_index:
            return self.gust_value(card)
        energy = len(getattr(card, "energies", None) or [])
        return energy * 100 + (500 if card.id in self.ATTACKER_IDS else 0) + getattr(card, "hp", 0)

    def gust_value(self, card: Any) -> float:
        return prize_count(card) * 2_000 - getattr(card, "hp", 0)

    def score_damage_target(self, option: Any, card: Any) -> float:
        player_index = getattr(option, "playerIndex", self.my_index)
        if player_index == self.op_index:
            return 5_000 - getattr(card, "hp", 0) + prize_count(card) * 500
        return -getattr(card, "hp", 0)

    def score_setup_active(self, card: Any) -> float:
        return 100 if card.id in self.ATTACKER_IDS else 10

    def score_to_bench(self, card: Any) -> float:
        return 100 - self.field[card.id] * 20

    def score_to_hand(self, card: Any) -> float:
        return 200 - self.hand[card.id] * 40

    def score_evolves_choice(self, card: Any) -> float:
        return 1_000 if card.id in self.ATTACKER_IDS else 500

    def score_discard(self, option: Any, card: Any) -> float:
        if self.is_energy(card.id):
            return 20 if self.hand[card.id] >= 3 else -20
        return 60 if self.hand[card.id] >= 2 else 0

    def score_putback(self, card: Any) -> float:
        return 60 if self.hand[card.id] >= 2 else 10

    def score_retreat(self) -> float:
        active = self.me.active[0] if self.me.active else None
        if active is not None and not self.can_attack(active) and self.bench_attacker_ready():
            return 6_000
        return -1

    def score_end(self) -> float:
        return 0

    @abstractmethod
    def go_first(self) -> bool: ...

    @abstractmethod
    def score_play_poke(self, card: Any) -> float: ...

    @abstractmethod
    def score_play_trainer(self, card: Any) -> float: ...

    @abstractmethod
    def score_evolve(self, option: Any) -> float: ...

    @abstractmethod
    def score_attack(self, option: Any) -> float: ...

    def score_ability(self, option: Any) -> float:
        return 1_000


def make_agent(policy_cls: type[BasePolicy], deck: list[int], diag: dict[str, Any]):
    """Create the competition entrypoint with diagnostics and legal fallbacks."""

    def record_error(exc: Exception) -> None:
        key = f"{type(exc).__name__}: {str(exc)[:160]}"
        diag["errors"][key] = diag["errors"].get(key, 0) + 1

    def agent(obs_dict: dict[str, Any]) -> list[int]:
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
                result = policy_cls(obs).choose()
                diag["policy_ok"] += 1
                return result
            except Exception as exc:
                record_error(exc)
                diag["policy_fallback"] += 1
                return legal_fallback(obs.select)
        except Exception as exc:
            record_error(exc)
            diag["obs_fallback"] += 1
            return legal_fallback_from_dict(obs_dict if isinstance(obs_dict, dict) else {})

    return agent
