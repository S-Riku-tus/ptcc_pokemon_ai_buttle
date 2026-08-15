"""Belief-corrected adaptive search for public Grimmsnarl mirrors.

The v22 ranker remains the champion policy and v25 contributes candidates and
one alternate opponent response model.  Every hidden hypothesis is now drawn
without replacement from the exact 60-card mirror list after subtracting all
public cards.  The cheap pass is still v26's H2.  Only a close, tactically
important position with adequate public-information confidence is replayed to
H3, and only a robust H3 challenger earns more samples (3 -> 5 -> 7 -> 9).

No candidate may be worse than v22 in any sampled world or lose the existing
prize/deck-out safety comparison.  Missing state, an impossible deck count,
opponent-model disagreement, incomplete rollout, budget pressure, or any
exception therefore returns v22 unchanged.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import json
import math
import os
import random
import time
from collections import Counter
from typing import Any

import fallback_policy
import value_features
from ml_runtime import _prepare, _resolve, tree_score


MAIN_CONTEXT = 0
MIN_TURN = 5
TOP_K = 3
MAX_RANK_MARGIN = 3.0
DETERMINIZATIONS = 3
H3_SAMPLE_STAGES = (3, 5, 7, 9)
MAX_ROLLOUT_STEPS = 192
MIN_MEAN_VALUE_GAIN = 0.04
H3_PROBE_GAIN = 0.015
H2_AMBIGUITY_MARGIN = 0.06
MIN_BELIEF_CONFIDENCE = 0.55
ADAPTIVE_BRANCH_COST_SECONDS = 1.6

# actTimeout is zero and the episode bank is 600 seconds.  v26 reserves a
# third of it and never spends more than 180 seconds itself.
OVERAGE_RESERVE_SECONDS = 200.0
MAX_GAME_SEARCH_SECONDS = 180.0
SEARCH_COST_PRIOR_SECONDS = 10.0
SEARCH_COST_SAFETY = 1.5


class SearchBudget:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.spent = 0.0
        self.searches = 0
        self.stops = 0
        self.last_remaining: float | None = None
        self.min_remaining: float | None = None

    def note(self, observation: dict[str, Any]) -> None:
        raw = observation.get("remainingOverageTime")
        remaining = float(raw) if isinstance(raw, (int, float)) else None
        if remaining is None:
            self.last_remaining = None
            return
        if (
            self.last_remaining is not None
            and remaining > self.last_remaining + 30.0
        ):
            self.reset()
        self.last_remaining = remaining
        self.min_remaining = (
            remaining if self.min_remaining is None
            else min(self.min_remaining, remaining)
        )

    @property
    def mean_cost(self) -> float:
        if not self.searches:
            return SEARCH_COST_PRIOR_SECONDS
        return max(self.spent / self.searches, 0.05)

    def allowed(self) -> bool:
        return self.available_seconds() >= self.mean_cost * SEARCH_COST_SAFETY

    def available_seconds(self) -> float:
        own = MAX_GAME_SEARCH_SECONDS - self.spent
        bank = (
            float("inf") if self.last_remaining is None
            else self.last_remaining - OVERAGE_RESERVE_SECONDS
        )
        return max(0.0, min(own, bank))

    def charge(self, seconds: float) -> None:
        self.spent += max(0.0, float(seconds))
        self.searches += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "search_seconds": round(self.spent, 3),
            "search_seconds_mean": round(self.mean_cost, 4),
            "budget_searches": self.searches,
            "budget_stops": self.stops,
            "overage_remaining_last": (
                round(self.last_remaining, 2)
                if self.last_remaining is not None else -1.0
            ),
            "overage_remaining_min": (
                round(self.min_remaining, 2)
                if self.min_remaining is not None else -1.0
            ),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item) for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


class _SearchApi:
    """Official Kaggle wrapper, with the repository's low-level local shim."""

    def __init__(self) -> None:
        from cg import api

        self.api = api
        self.high_level = all(hasattr(api, name) for name in (
            "search_begin", "search_step", "search_end",
        ))
        self.pointer = None
        self.lib = None
        if not self.high_level:
            self._bind_low_level()

    def _bind_low_level(self) -> None:
        from cg.sim import lib

        pointer = ctypes.POINTER(ctypes.c_int)
        lib.AgentStart.restype = ctypes.c_void_p
        lib.AgentStart.argtypes = []
        lib.SearchBegin.restype = ctypes.c_char_p
        lib.SearchBegin.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            pointer, pointer, pointer, pointer, pointer, pointer,
            ctypes.c_int,
        ]
        lib.SearchStep.restype = ctypes.c_char_p
        lib.SearchStep.argtypes = [
            ctypes.c_void_p, ctypes.c_int, pointer, ctypes.c_int,
        ]
        lib.SearchEnd.restype = None
        lib.SearchEnd.argtypes = [ctypes.c_void_p]
        self.lib = lib
        self.pointer = lib.AgentStart()
        if not self.pointer:
            raise RuntimeError("AgentStart failed")

    @staticmethod
    def _state(result: Any) -> dict[str, Any]:
        if hasattr(result, "error") and int(result.error or 0) != 0:
            raise RuntimeError(f"search error {int(result.error)}")
        if isinstance(result, dict):
            if int(result.get("error", 0) or 0) != 0:
                raise RuntimeError(f"search error {result.get('error')}")
            state = result.get("state", result)
        else:
            state = getattr(result, "state", result)
        plain = _plain(state)
        if not isinstance(plain, dict) or not plain:
            raise RuntimeError("search returned no state")
        return plain

    def begin(self, observation: dict[str, Any], hidden: list[list[int]]) -> dict:
        if self.high_level:
            try:
                result = self.api.search_begin(observation, *hidden, False)
            except (AttributeError, TypeError):
                converted = self.api.to_observation_class(observation)
                result = self.api.search_begin(converted, *hidden, False)
            return self._state(result)
        arrays = [(ctypes.c_int * len(values))(*values) for values in hidden]
        encoded = str(observation["search_begin_input"]).encode("ascii")
        raw = self.lib.SearchBegin(
            self.pointer, encoded, len(encoded), *arrays, 0,
        )
        if not raw:
            raise RuntimeError("SearchBegin returned null")
        return self._state(json.loads(raw))

    def step(self, search_id: int, selection: list[int]) -> dict:
        if self.high_level:
            return self._state(self.api.search_step(search_id, selection))
        array = (ctypes.c_int * len(selection))(*selection)
        raw = self.lib.SearchStep(
            self.pointer, int(search_id), array, len(selection),
        )
        if not raw:
            raise RuntimeError("SearchStep returned null")
        return self._state(json.loads(raw))

    def end(self) -> None:
        if self.high_level:
            self.api.search_end()
        else:
            self.lib.SearchEnd(self.pointer)


class _ValueModel:
    def __init__(self, path: str = "value_model.json") -> None:
        with open(_resolve(path), encoding="utf-8") as handle:
            self.model = json.load(handle)
        for tree in self.model["trees"]:
            _prepare(tree)
        self.names = list(self.model["feature_names"])

    def probability(self, observation: dict[str, Any], perspective: int) -> float:
        row = value_features.vector(observation, self.names, perspective)
        raw = tree_score(row, self.model)
        if raw >= 0:
            return 1.0 / (1.0 + math.exp(-min(raw, 50.0)))
        exponent = math.exp(max(raw, -50.0))
        return exponent / (1.0 + exponent)


def _card_ids(value: Any):
    if isinstance(value, dict):
        card_id = value.get("id")
        if isinstance(card_id, int):
            yield card_id
        for nested in value.values():
            yield from _card_ids(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _card_ids(nested)


def _owned_stadium_ids(current: dict[str, Any], player: int):
    for card in current.get("stadium") or []:
        if not isinstance(card, dict):
            continue
        owner = card.get("playerIndex")
        if owner is not None and int(owner) == player:
            card_id = card.get("id")
            if isinstance(card_id, int):
                yield card_id


def _known_prize_ids(prizes: list[Any]) -> list[int]:
    return [
        int(card["id"])
        for card in prizes
        if isinstance(card, dict) and isinstance(card.get("id"), int)
    ]


def _subtract_known(
    remaining: Counter[int], card_ids, *, label: str,
) -> None:
    """Subtract visible cards strictly; an impossible mirror must not search."""
    for raw in card_ids:
        card_id = int(raw)
        if remaining[card_id] <= 0:
            raise ValueError(f"{label}: card {card_id} exceeds mirror deck count")
        remaining[card_id] -= 1


def _pool(remaining: Counter[int]) -> list[int]:
    result: list[int] = []
    for card_id, count in remaining.items():
        if count < 0:
            raise ValueError(f"negative remaining count for card {card_id}")
        result.extend([int(card_id)] * int(count))
    return result


def _fill_prizes(
    prizes: list[Any], pool: list[int], cursor: int,
) -> tuple[list[int], int]:
    values: list[int] = []
    for card in prizes:
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            values.append(int(card["id"]))
        else:
            if cursor >= len(pool):
                raise ValueError("not enough cards for unknown prizes")
            values.append(pool[cursor])
            cursor += 1
    return values, cursor


def _hidden_state(
    observation: dict[str, Any], perspective: int, salt: int = 0,
) -> list[list[int]]:
    """Sample one legal mirror world from a single without-replacement pool.

    v26 independently repeated the 60-card list for the opponent's hand, deck,
    and prizes.  That could create eight Impidimp or sixteen Darkness Energy.
    Here both players' hidden zones exactly conserve the known deck multiset.
    Any observation that cannot be reconciled with the mirror list raises and
    is handled by the caller's fail-closed search path.
    """
    current = observation.get("current") or {}
    players = current.get("players") or []
    if perspective not in (0, 1) or len(players) < 2:
        raise ValueError("two valid players are required")
    opponent = 1 - perspective
    me = players[perspective]
    them = players[opponent]
    if not isinstance(me, dict) or not isinstance(them, dict):
        raise ValueError("player state is missing")

    deck = list(fallback_policy.MY_DECK)
    if len(deck) != 60:
        raise ValueError(f"mirror deck has {len(deck)} cards, expected 60")
    seed = (
        int(current.get("turn", 0) or 0) * 1009
        + int(current.get("turnActionCount", 0) or 0) * 97
        + perspective * 17
        + salt * 104_729
    )

    # Our hand and board are known.  Known prize identities are fixed while
    # only the remaining prize slots and deck are sampled from one pool.
    own_remaining: Counter[int] = Counter(deck)
    for zone in ("hand", "discard", "active", "bench"):
        _subtract_known(
            own_remaining, _card_ids(me.get(zone) or []),
            label=f"own {zone}",
        )
    _subtract_known(
        own_remaining, _owned_stadium_ids(current, perspective),
        label="own stadium",
    )
    own_prizes_raw = list(me.get("prize") or [])
    _subtract_known(
        own_remaining, _known_prize_ids(own_prizes_raw),
        label="own known prize",
    )
    own_pool = _pool(own_remaining)
    own_unknown_prizes = len(own_prizes_raw) - len(
        _known_prize_ids(own_prizes_raw)
    )
    own_deck_count = int(me.get("deckCount", 0) or 0)
    if len(own_pool) != own_deck_count + own_unknown_prizes:
        raise ValueError(
            "own hidden-zone count mismatch: "
            f"pool={len(own_pool)} deck={own_deck_count} "
            f"unknown_prize={own_unknown_prizes}"
        )
    random.Random(seed ^ 0x51A7).shuffle(own_pool)
    own_prize, cursor = _fill_prizes(own_prizes_raw, own_pool, 0)
    own_deck = own_pool[cursor:]
    if len(own_deck) != own_deck_count:
        raise ValueError("own sampled deck length mismatch")

    # The opponent uses the same operation.  Public play/discard/evolution
    # stacks, attached energy, tools, stadium and any explicitly revealed hand
    # or prize identities are removed before mutually exclusive allocation.
    opponent_remaining: Counter[int] = Counter(deck)
    for zone in ("discard", "active", "bench"):
        _subtract_known(
            opponent_remaining, _card_ids(them.get(zone) or []),
            label=f"opponent {zone}",
        )
    _subtract_known(
        opponent_remaining, _owned_stadium_ids(current, opponent),
        label="opponent stadium",
    )
    known_hand = list(_card_ids(them.get("hand") or []))
    _subtract_known(opponent_remaining, known_hand, label="opponent known hand")
    opponent_prizes_raw = list(them.get("prize") or [])
    known_opponent_prizes = _known_prize_ids(opponent_prizes_raw)
    _subtract_known(
        opponent_remaining, known_opponent_prizes,
        label="opponent known prize",
    )
    active = list(them.get("active") or [])
    unknown_active_count = sum(card is None for card in active)
    if unknown_active_count > 1:
        raise ValueError("multiple hidden Active cards are unsupported")
    opponent_hand_count = int(them.get("handCount", 0) or 0)
    unknown_hand_count = opponent_hand_count - len(known_hand)
    unknown_prize_count = len(opponent_prizes_raw) - len(known_opponent_prizes)
    opponent_deck_count = int(them.get("deckCount", 0) or 0)
    if unknown_hand_count < 0 or unknown_prize_count < 0:
        raise ValueError("known hidden-zone identities exceed zone count")

    opponent_pool = _pool(opponent_remaining)
    expected = (
        opponent_deck_count + unknown_prize_count
        + unknown_hand_count + unknown_active_count
    )
    if len(opponent_pool) != expected:
        raise ValueError(
            "opponent hidden-zone count mismatch: "
            f"pool={len(opponent_pool)} expected={expected}"
        )
    random.Random(seed ^ 0xA93D).shuffle(opponent_pool)

    opponent_active: list[int] = []
    if unknown_active_count:
        # A facedown setup Active must be a Basic from this exact list.
        basic_ids = {112, 646, 860}
        slot = next(
            (index for index, card_id in enumerate(opponent_pool)
             if card_id in basic_ids),
            None,
        )
        if slot is None:
            raise ValueError("no legal Basic remains for hidden Active")
        opponent_active.append(opponent_pool.pop(slot))

    cursor = 0
    opponent_hand = list(known_hand)
    opponent_hand.extend(
        opponent_pool[cursor:cursor + unknown_hand_count]
    )
    cursor += unknown_hand_count
    opponent_prize, cursor = _fill_prizes(
        opponent_prizes_raw, opponent_pool, cursor
    )
    opponent_deck = opponent_pool[cursor:]
    if len(opponent_deck) != opponent_deck_count:
        raise ValueError("opponent sampled deck length mismatch")
    return [
        own_deck, own_prize, opponent_deck, opponent_prize,
        opponent_hand, opponent_active,
    ]


def _public_card_count(current: dict[str, Any], player: int) -> int:
    players = current.get("players") or []
    if len(players) <= player or not isinstance(players[player], dict):
        return 0
    state = players[player]
    count = sum(
        1
        for zone in ("active", "bench", "discard")
        for _ in _card_ids(state.get(zone) or [])
    )
    count += sum(1 for _ in _owned_stadium_ids(current, player))
    return count


def _belief_confidence(observation: dict[str, Any], perspective: int) -> float:
    """Public, monotone proxy for whether a deeper mirror world is credible."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    opponent = 1 - perspective
    if len(players) <= opponent or not isinstance(players[opponent], dict):
        return 0.0
    them = players[opponent]
    turn = max(0, int(current.get("turn", 0) or 0))
    public = min(30, _public_card_count(current, opponent))
    hand = max(0, int(them.get("handCount", 0) or 0))
    deck_count = max(0, int(them.get("deckCount", 0) or 0))
    confidence = (
        0.34
        + public / 50.0
        + min(turn, 12) * 0.015
        + max(0, 6 - hand) * 0.025
        - max(0, hand - 6) * 0.035
        - (0.08 if deck_count > 35 else 0.0)
    )
    return max(0.0, min(1.0, confidence))


def _tactical_h3_position(
    observation: dict[str, Any], perspective: int,
) -> bool:
    """Whether another exchange can expose delayed Shadow/Adrena value."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    if len(players) < 2:
        return False
    me = players[perspective]
    them = players[1 - perspective]
    if not isinstance(me, dict) or not isinstance(them, dict):
        return False
    own_prizes = len(me.get("prize") or [])
    opponent_prizes = len(them.get("prize") or [])
    turn = int(current.get("turn", 0) or 0)
    public_ids = list(_card_ids(me.get("active") or []))
    public_ids.extend(_card_ids(me.get("bench") or []))
    public_ids.extend(_card_ids(them.get("active") or []))
    public_ids.extend(_card_ids(them.get("bench") or []))
    grim_count = sum(card_id == 648 for card_id in public_ids)
    munkidori_count = sum(card_id == 112 for card_id in public_ids)
    damaged_bench = any(
        isinstance(card, dict)
        and int(card.get("maxHp", card.get("hp", 0)) or 0)
        > int(card.get("hp", 0) or 0)
        for player in (me, them)
        for card in (player.get("bench") or [])
    )
    return bool(
        min(own_prizes, opponent_prizes) <= 3
        or turn >= 9
        or grim_count >= 2
        or munkidori_count >= 2
        or damaged_bench
    )


def _simple_legal(select: dict[str, Any]) -> list[int]:
    count = len(select.get("option") or [])
    maximum = min(int(select.get("maxCount", 0) or 0), count)
    minimum = min(int(select.get("minCount", 0) or 0), maximum)
    return list(range(max(minimum, maximum)))


def _terminal_value(observation: dict[str, Any], perspective: int) -> float | None:
    result = int((observation.get("current") or {}).get("result", -1))
    if result < 0:
        return None
    if result == perspective:
        return 1.0
    if result in (0, 1):
        return 0.0
    return 0.5


def _safety(observation: dict[str, Any], perspective: int) -> tuple[int, int, int]:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    if len(players) < 2:
        return (-99, 0, 0)
    me = players[perspective]
    opponent = players[1 - perspective]
    own_prizes = len(me.get("prize") or [])
    opponent_prizes = len(opponent.get("prize") or [])
    prize_lead = opponent_prizes - own_prizes
    return prize_lead, int(me.get("deckCount", 0) or 0), opponent_prizes


_AUTO = object()


class H2SearchPlanner:
    def __init__(self, search_api: Any = _AUTO, value_model: Any = _AUTO) -> None:
        self.disabled = os.environ.get("GRIMMSNARL_H2_DISABLE") == "1"
        self.load_error: str | None = None
        self.search = None
        self.value = None
        if not self.disabled:
            try:
                self.value = (
                    _ValueModel() if value_model is _AUTO else value_model
                )
                self.search = (
                    _SearchApi() if search_api is _AUTO else search_api
                )
            except Exception as error:  # engine/model absence is a safe veto
                self.load_error = f"{type(error).__name__}: {error}"
                self.disabled = True
        self.budget = SearchBudget()
        self.reset()

    def reset(self) -> None:
        self._searched_turns: set[tuple[int, int]] = set()
        self._records: list[dict[str, Any]] = []
        self.budget.reset()
        self.stats: Counter[str] = Counter()

    def _branch_choice(
        self, observation: dict[str, Any], perspective: int,
        ranker, rule_agent, opponent_ranker=None,
        opponent_policy: str = "rule",
    ) -> list[int]:
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        actor = int(current.get("yourIndex", -1))
        if actor == perspective and ranker.is_scorable(select):
            index = ranker.choose(observation)
            if index is not None:
                ranker.commit(index)
                return [index]
        if (
            actor != perspective
            and opponent_policy == "peer"
            and opponent_ranker is not None
            and opponent_ranker.is_scorable(select)
        ):
            index = opponent_ranker.choose(observation)
            if index is not None:
                opponent_ranker.commit(index)
                return [index]
        try:
            choice = list(rule_agent(observation))
        except Exception:
            choice = _simple_legal(select)
        if actor == perspective and len(choice) == 1:
            ranker.observe_external(observation, choice[0])
        elif (
            actor != perspective
            and opponent_policy == "peer"
            and opponent_ranker is not None
            and len(choice) == 1
        ):
            opponent_ranker.observe_external(observation, choice[0])
        return choice

    def _rollout(
        self, root_id: int, candidate: int, perspective: int,
        ranker, ranker_state: dict[str, Any], *,
        future_own_turns: int = 1, opponent_ranker=None,
        opponent_state: dict[str, Any] | None = None,
        opponent_policy: str = "rule",
    ) -> tuple[dict, float] | None:
        ranker.restore_dynamic_state(ranker_state)
        ranker.teacher_forced = False
        ranker.commit(candidate)
        if opponent_ranker is not None and opponent_state is not None:
            opponent_ranker.restore_dynamic_state(opponent_state)
            opponent_ranker.teacher_forced = False
        branch_diag = fallback_policy._fresh_diag()
        rule_agent = fallback_policy.make_agent(
            fallback_policy.GrimmsnarlPolicy,
            fallback_policy.MY_DECK,
            branch_diag,
        )
        state = self.search.step(root_id, [candidate])
        saw_opponent = False
        future_own_started = 0
        last_actor = perspective
        for _ in range(MAX_ROLLOUT_STEPS):
            observation = state.get("observation") or {}
            terminal = _terminal_value(observation, perspective)
            if terminal is not None:
                return observation, terminal
            current = observation.get("current") or {}
            actor = int(current.get("yourIndex", -1))
            select = observation.get("select") or {}
            if (
                future_own_started >= future_own_turns
                and actor != perspective
                and int(select.get("context", -1)) == MAIN_CONTEXT
            ):
                # Resolve promotions/prize/effect selections caused by our
                # second attack first.  The leaf is the opponent's first real
                # MAIN decision, where both Active slots and every prize taken
                # by the H2 line are already represented.
                return observation, self.value.probability(
                    observation, perspective
                )
            if actor != perspective:
                saw_opponent = True
            elif saw_opponent and last_actor != perspective:
                future_own_started += 1
            selection = self._branch_choice(
                observation, perspective, ranker, rule_agent,
                opponent_ranker, opponent_policy,
            )
            state = self.search.step(int(state["searchId"]), selection)
            last_actor = actor
        return None

    @staticmethod
    def _safe_against(
        candidate_leaf: dict[str, Any], base_leaf: dict[str, Any], perspective: int,
    ) -> bool:
        candidate_terminal = _terminal_value(candidate_leaf, perspective)
        base_terminal = _terminal_value(base_leaf, perspective)
        if candidate_terminal == 1.0:
            return True
        if base_terminal == 1.0:
            return False
        candidate = _safety(candidate_leaf, perspective)
        base = _safety(base_leaf, perspective)
        # Never accept a worse two-turn prize exchange.  Also reject a line
        # which reaches an empty deck when v22 retained a card.
        return (
            candidate[0] >= base[0]
            and not (candidate[1] <= 0 < base[1])
        )

    def _candidates(self, proposed: int, peer: int | None,
                    scores: dict[int, float], option_count: int) -> list[int]:
        result: list[int] = []
        for index in (proposed, peer):
            if isinstance(index, int) and 0 <= index < option_count:
                if index not in result:
                    result.append(index)
        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
        top_score = scores.get(ordered[0], 0.0) if ordered else 0.0
        for index in ordered:
            if len(result) >= TOP_K:
                break
            if not 0 <= index < option_count:
                continue
            if top_score - scores[index] > MAX_RANK_MARGIN:
                continue
            if index not in result:
                result.append(index)
        return result

    def _h2_outcome(
        self, candidates: list[int], proposed: int,
        values: dict[int, list[float]], leaves: dict[int, list[dict[str, Any]]],
        scores: dict[int, float], perspective: int,
    ) -> tuple[int | None, str, list[float]]:
        winners = [
            max(
                candidates,
                key=lambda index: (
                    values[index][sample], scores.get(index, float("-inf"))
                ),
            )
            for sample in range(DETERMINIZATIONS)
        ]
        if len(set(winners)) != 1:
            return None, "determinization_disagreement", []
        best = winners[0]
        gains = [
            values[best][sample] - values[proposed][sample]
            for sample in range(DETERMINIZATIONS)
        ]
        if best == proposed or min(gains) < -1e-9:
            return None, "not_dominant", gains
        if not all(
            self._safe_against(
                leaves[best][sample], leaves[proposed][sample], perspective
            )
            for sample in range(DETERMINIZATIONS)
        ):
            return None, "prize_safety", gains
        if sum(gains) / len(gains) < MIN_MEAN_VALUE_GAIN:
            return None, "small_gain", gains
        return best, "accept", gains

    @staticmethod
    def _h2_is_ambiguous(
        candidates: list[int], values: dict[int, list[float]],
    ) -> bool:
        winners = [
            max(candidates, key=lambda index: values[index][sample])
            for sample in range(DETERMINIZATIONS)
        ]
        means = sorted(
            (sum(values[index]) / len(values[index]) for index in candidates),
            reverse=True,
        )
        return bool(
            len(set(winners)) != 1
            or (len(means) >= 2 and means[0] - means[1] <= H2_AMBIGUITY_MARGIN)
        )

    @staticmethod
    def _h3_challenger(
        candidates: list[int], proposed: int,
        values: dict[int, list[float]], scores: dict[int, float],
    ) -> int | None:
        alternatives = [index for index in candidates if index != proposed]
        if not alternatives:
            return None
        return max(
            alternatives,
            key=lambda index: (
                min(values[index]),
                sum(values[index]) / len(values[index]),
                scores.get(index, float("-inf")),
            ),
        )

    def _h3_status(
        self, proposed: int, challenger: int,
        values: dict[int, list[float]], leaves: dict[int, list[dict[str, Any]]],
        perspective: int,
    ) -> tuple[str, list[float]]:
        sample_count = len(values.get(challenger, []))
        if sample_count != len(values.get(proposed, [])) or sample_count == 0:
            return "incomplete", []
        gains = [
            values[challenger][sample] - values[proposed][sample]
            for sample in range(sample_count)
        ]
        # Worst-case protection is intentionally inherited from v26: more
        # samples may reduce uncertainty, but they may never vote away a world
        # in which the challenger is worse than v22.
        if min(gains) < -1e-9:
            return "not_dominant", gains
        if not all(
            self._safe_against(
                leaves[challenger][sample], leaves[proposed][sample], perspective
            )
            for sample in range(sample_count)
        ):
            return "prize_safety", gains
        mean_gain = sum(gains) / sample_count
        if sample_count >= 5 and mean_gain >= MIN_MEAN_VALUE_GAIN:
            return "accept", gains
        if sample_count < H3_SAMPLE_STAGES[-1] and mean_gain >= H3_PROBE_GAIN:
            return "continue", gains
        return "small_gain", gains

    def _can_afford_adaptive(
        self, began_at: float, branch_count: int,
    ) -> bool:
        elapsed = time.perf_counter() - began_at
        expected = (
            max(1, branch_count)
            * ADAPTIVE_BRANCH_COST_SECONDS
            * SEARCH_COST_SAFETY
        )
        return elapsed + expected <= self.budget.available_seconds()

    def adjust(
        self, observation: dict[str, Any], proposed: int,
        scores: dict[int, float], ranker, peer_choice: int | None,
        *, is_mirror: bool, opponent_ranker=None,
        allow_non_top: bool = False,
    ) -> int:
        self.budget.note(observation)
        if self.disabled or self.search is None or self.value is None:
            self.stats["skip_disabled"] += 1
            return proposed
        if not is_mirror:
            self.stats["skip_non_mirror"] += 1
            return proposed
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != MAIN_CONTEXT:
            self.stats["skip_non_main"] += 1
            return proposed
        self.stats["considered"] += 1
        current = observation.get("current") or {}
        turn = int(current.get("turn", 0) or 0)
        perspective = int(current.get("yourIndex", 0) or 0)
        key = (turn, perspective)
        if turn < MIN_TURN:
            self.stats["skip_early"] += 1
            return proposed
        if key in self._searched_turns:
            self.stats["skip_already_searched_turn"] += 1
            return proposed
        if not observation.get("search_begin_input"):
            self.stats["skip_no_search_state"] += 1
            return proposed
        ordered = sorted(scores, key=lambda index: scores[index], reverse=True)
        if not ordered or (proposed != ordered[0] and not allow_non_top):
            # A deterministic one-ply invariant has already changed v22.
            self.stats["skip_planner_override"] += 1
            return proposed
        margin = (
            scores[ordered[0]] - scores[ordered[1]]
            if len(ordered) >= 2 else float("inf")
        )
        if (
            not allow_non_top
            and peer_choice == proposed
            and margin > MAX_RANK_MARGIN
        ):
            self.stats["skip_consensus_confident"] += 1
            return proposed
        options = list(select.get("option") or [])
        candidates = self._candidates(
            proposed, peer_choice, scores, len(options)
        )
        if len(candidates) < 2:
            self.stats["skip_one_candidate"] += 1
            return proposed
        if not self.budget.allowed():
            self.budget.stops += 1
            self.stats["skip_budget"] += 1
            return proposed

        ranker_state = ranker.save_dynamic_state()
        opponent_state = (
            opponent_ranker.save_dynamic_state()
            if opponent_ranker is not None else None
        )
        diag_state = copy.deepcopy(fallback_policy.DIAG)
        immunity_state = copy.deepcopy(fallback_policy.TEMP_IMMUNITY)
        values: dict[int, list[float]] = {index: [] for index in candidates}
        leaves: dict[int, list[dict[str, Any]]] = {
            index: [] for index in candidates
        }
        started_any = False
        began_at = time.perf_counter()
        h2_choice: int | None = None
        h2_reason = "incomplete"
        h2_gains: list[float] = []
        h3_choice: int | None = None
        h3_reason = "not_considered"
        h3_gains: list[float] = []
        h3_values: dict[int, list[float]] = {}
        h3_leaves: dict[int, list[dict[str, Any]]] = {}
        h3_candidates: list[int] = []

        def collect(
            salts, branch_candidates: list[int], future_own_turns: int,
            target_values: dict[int, list[float]],
            target_leaves: dict[int, list[dict[str, Any]]],
            *, adaptive: bool,
        ) -> bool:
            nonlocal started_any
            for salt in salts:
                if adaptive and not self._can_afford_adaptive(
                    began_at, len(branch_candidates)
                ):
                    self.budget.stops += 1
                    self.stats["skip_adaptive_budget"] += 1
                    return False
                started = False
                before = {
                    index: len(target_values[index])
                    for index in branch_candidates
                }
                try:
                    hidden = _hidden_state(observation, perspective, salt)
                    root = self.search.begin(observation, hidden)
                    started = True
                    started_any = True
                    root_id = int(root["searchId"])
                    opponent_policy = (
                        "peer"
                        if opponent_ranker is not None and salt % 2 == 1
                        else "rule"
                    )
                    for candidate in branch_candidates:
                        self.stats["branches"] += 1
                        self.stats[
                            "h3_branches" if future_own_turns == 2
                            else "h2_branches"
                        ] += 1
                        self.stats[f"opponent_policy_{opponent_policy}"] += 1
                        result = self._rollout(
                            root_id, candidate, perspective, ranker,
                            ranker_state,
                            future_own_turns=future_own_turns,
                            opponent_ranker=opponent_ranker,
                            opponent_state=opponent_state,
                            opponent_policy=opponent_policy,
                        )
                        if result is None:
                            self.stats["incomplete_branches"] += 1
                            continue
                        leaf, value = result
                        target_leaves[candidate].append(leaf)
                        target_values[candidate].append(float(value))
                except ValueError:
                    self.stats["belief_rejections"] += 1
                except Exception:
                    self.stats["branch_errors"] += 1
                finally:
                    if started:
                        try:
                            self.search.end()
                        except Exception:
                            self.stats["branch_errors"] += 1
                if any(
                    len(target_values[index]) != before[index] + 1
                    for index in branch_candidates
                ):
                    return False
                if future_own_turns == 2:
                    self.stats["h3_worlds"] += 1
            return True

        try:
            complete_h2 = collect(
                range(DETERMINIZATIONS), candidates, 1, values, leaves,
                adaptive=False,
            )
            if complete_h2:
                h2_choice, h2_reason, h2_gains = self._h2_outcome(
                    candidates, proposed, values, leaves, scores, perspective
                )

            if (
                complete_h2
                and h2_choice is None
                and self._h2_is_ambiguous(candidates, values)
            ):
                confidence = _belief_confidence(observation, perspective)
                self.stats["belief_confidence_milli"] += int(
                    round(confidence * 1000.0)
                )
                if confidence < MIN_BELIEF_CONFIDENCE:
                    h3_reason = "low_belief_confidence"
                    self.stats["skip_h3_low_confidence"] += 1
                elif not _tactical_h3_position(observation, perspective):
                    h3_reason = "not_tactical"
                    self.stats["skip_h3_not_tactical"] += 1
                else:
                    challenger = self._h3_challenger(
                        candidates, proposed, values, scores
                    )
                    if challenger is None:
                        h3_reason = "no_challenger"
                    else:
                        h3_candidates = [proposed, challenger]
                        h3_values = {index: [] for index in h3_candidates}
                        h3_leaves = {index: [] for index in h3_candidates}
                        self.stats["h3_considered"] += 1
                        complete_h3 = collect(
                            range(H3_SAMPLE_STAGES[0]), h3_candidates, 2,
                            h3_values, h3_leaves, adaptive=True,
                        )
                        if complete_h3:
                            h3_reason, h3_gains = self._h3_status(
                                proposed, challenger, h3_values, h3_leaves,
                                perspective,
                            )
                        else:
                            h3_reason = "incomplete"
                        for target in H3_SAMPLE_STAGES[1:]:
                            if h3_reason != "continue":
                                break
                            start = len(h3_values[challenger])
                            if not collect(
                                range(start, target), h3_candidates, 2,
                                h3_values, h3_leaves, adaptive=True,
                            ):
                                h3_reason = "incomplete"
                                break
                            h3_reason, h3_gains = self._h3_status(
                                proposed, challenger, h3_values, h3_leaves,
                                perspective,
                            )
                        if h3_reason == "accept":
                            h3_choice = challenger
        finally:
            elapsed = time.perf_counter() - began_at
            self._searched_turns.add(key)
            if started_any:
                self.stats["searched"] += 1
                self.budget.charge(elapsed)
            ranker.restore_dynamic_state(ranker_state)
            if opponent_ranker is not None and opponent_state is not None:
                opponent_ranker.restore_dynamic_state(opponent_state)
            fallback_policy.DIAG.clear()
            fallback_policy.DIAG.update(diag_state)
            fallback_policy.TEMP_IMMUNITY.clear()
            fallback_policy.TEMP_IMMUNITY.update(immunity_state)

        if any(len(values[index]) != DETERMINIZATIONS for index in candidates):
            self.stats["skip_incomplete"] += 1
            return proposed
        best = h2_choice if h2_choice is not None else h3_choice
        horizon = "h2" if h2_choice is not None else "h3"
        gains = h2_gains if h2_choice is not None else h3_gains
        if best is None:
            self.stats[f"skip_{h2_reason}"] += 1
            if h3_reason not in (
                "not_considered", "low_belief_confidence", "not_tactical",
            ):
                self.stats[f"skip_h3_{h3_reason}"] += 1
            return proposed
        mean_gain = sum(gains) / len(gains)
        self.stats["overrides"] += 1
        self.stats[f"overrides_{horizon}"] += 1
        self.stats["value_gain_milli"] += int(round(mean_gain * 1000.0))
        if len(self._records) < 24:
            record_values = values if horizon == "h2" else h3_values
            self._records.append({
                "turn": turn,
                "horizon": horizon,
                "from_index": proposed,
                "to_index": best,
                "peer_index": peer_choice,
                "samples": len(record_values[best]),
                "values_from": [
                    round(value, 6) for value in record_values[proposed]
                ],
                "values_to": [
                    round(value, 6) for value in record_values[best]
                ],
                "mean_gain": round(mean_gain, 6),
            })
        return best

    def snapshot(self) -> dict[str, Any]:
        output: dict[str, Any] = dict(self.stats)
        output.update(self.budget.snapshot())
        output.update({
            "enabled": not self.disabled,
            "load_error": self.load_error,
            "min_turn": MIN_TURN,
            "determinizations": DETERMINIZATIONS,
            "adaptive_h3_sample_stages": list(H3_SAMPLE_STAGES),
            "min_belief_confidence": MIN_BELIEF_CONFIDENCE,
            "min_mean_value_gain": MIN_MEAN_VALUE_GAIN,
            "records": list(self._records),
        })
        return output
