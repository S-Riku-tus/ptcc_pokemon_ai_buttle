"""Conservative two-ply search for public Grimmsnarl mirrors.

The v22 ranker remains the policy.  v25 contributes one additional root
candidate, while the real game engine advances every candidate through our
current turn, the opponent reply, and our following turn.  A separately
trained public-state value model scores that H2 leaf.

This module is deliberately narrow.  It searches only a publicly confirmed
same-deck mirror from turn five onward, uses three hidden-zone
determinizations, and accepts an override only when all three select the same
candidate, none values it below v22, and its realised prize exchange is no
worse in every sample.  Any missing API, timeout, tie, incomplete rollout, or
exception returns v22 unchanged.
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
MAX_ROLLOUT_STEPS = 112
MIN_MEAN_VALUE_GAIN = 0.04

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
        own = MAX_GAME_SEARCH_SECONDS - self.spent
        bank = (
            float("inf") if self.last_remaining is None
            else self.last_remaining - OVERAGE_RESERVE_SECONDS
        )
        return min(own, bank) >= self.mean_cost * SEARCH_COST_SAFETY

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


def _repeat(values: list[int], count: int, offset: int = 0) -> list[int]:
    source = values or list(fallback_policy.MY_DECK)
    return [source[(offset + index) % len(source)] for index in range(count)]


def _hidden_state(
    observation: dict[str, Any], perspective: int, salt: int = 0,
) -> list[list[int]]:
    """One deterministic legal hidden-zone hypothesis under a mirror prior."""
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    opponent = 1 - perspective
    me = players[perspective]
    them = players[opponent]
    deck = list(fallback_policy.MY_DECK)
    remaining = Counter(deck)
    for zone in ("hand", "discard", "active", "bench"):
        for card_id in _card_ids(me.get(zone) or []):
            if remaining[card_id] > 0:
                remaining[card_id] -= 1
    for card in me.get("prize") or []:
        if isinstance(card, dict):
            card_id = card.get("id")
            if isinstance(card_id, int) and remaining[card_id] > 0:
                remaining[card_id] -= 1

    pool: list[int] = []
    for card_id, count in remaining.items():
        pool.extend([card_id] * max(int(count), 0))
    seed = (
        int(current.get("turn", 0) or 0) * 1009
        + int(current.get("turnActionCount", 0) or 0) * 97
        + perspective * 17
        + salt * 104_729
    )
    random.Random(seed).shuffle(pool)
    prize_count = len(me.get("prize") or [])
    deck_count = int(me.get("deckCount", 0) or 0)
    needed = deck_count + prize_count
    if len(pool) < needed:
        pool.extend(_repeat(deck, needed - len(pool), seed % len(deck)))

    prize_values: list[int] = []
    cursor = 0
    for card in me.get("prize") or []:
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            prize_values.append(int(card["id"]))
        else:
            prize_values.append(pool[cursor])
            cursor += 1
    own_deck = pool[cursor:cursor + deck_count]
    if len(own_deck) < deck_count:
        own_deck.extend(_repeat(deck, deck_count - len(own_deck), seed))

    offset = (seed + sum(_card_ids(them.get("discard") or []))) % len(deck)
    opponent_deck = _repeat(deck, int(them.get("deckCount", 0) or 0), offset)
    opponent_prize = _repeat(deck, len(them.get("prize") or []), offset + 11)
    opponent_hand = _repeat(deck, int(them.get("handCount", 0) or 0), offset + 23)
    active = them.get("active") or []
    opponent_active = [646] if active and active[0] is None else []
    return [
        own_deck, prize_values, opponent_deck, opponent_prize,
        opponent_hand, opponent_active,
    ]


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

    def _branch_choice(self, observation: dict[str, Any], perspective: int,
                       ranker, rule_agent) -> list[int]:
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        actor = int(current.get("yourIndex", -1))
        if actor == perspective and ranker.is_scorable(select):
            index = ranker.choose(observation)
            if index is not None:
                ranker.commit(index)
                return [index]
        try:
            choice = list(rule_agent(observation))
        except Exception:
            choice = _simple_legal(select)
        if actor == perspective and len(choice) == 1:
            ranker.observe_external(observation, choice[0])
        return choice

    def _rollout(self, root_id: int, candidate: int, perspective: int,
                 ranker, ranker_state: dict[str, Any]) -> tuple[dict, float] | None:
        ranker.restore_dynamic_state(ranker_state)
        ranker.teacher_forced = False
        ranker.commit(candidate)
        branch_diag = fallback_policy._fresh_diag()
        rule_agent = fallback_policy.make_agent(
            fallback_policy.GrimmsnarlPolicy,
            fallback_policy.MY_DECK,
            branch_diag,
        )
        state = self.search.step(root_id, [candidate])
        saw_opponent = False
        saw_next_own = False
        for _ in range(MAX_ROLLOUT_STEPS):
            observation = state.get("observation") or {}
            terminal = _terminal_value(observation, perspective)
            if terminal is not None:
                return observation, terminal
            current = observation.get("current") or {}
            actor = int(current.get("yourIndex", -1))
            select = observation.get("select") or {}
            if (
                saw_next_own
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
            elif saw_opponent:
                saw_next_own = True
            selection = self._branch_choice(
                observation, perspective, ranker, rule_agent
            )
            state = self.search.step(int(state["searchId"]), selection)
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

    def adjust(self, observation: dict[str, Any], proposed: int,
               scores: dict[int, float], ranker, peer_choice: int | None,
               *, is_mirror: bool) -> int:
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
        if not ordered or proposed != ordered[0]:
            # A deterministic one-ply invariant has already changed v22.
            self.stats["skip_planner_override"] += 1
            return proposed
        margin = (
            scores[ordered[0]] - scores[ordered[1]]
            if len(ordered) >= 2 else float("inf")
        )
        if peer_choice == proposed and margin > MAX_RANK_MARGIN:
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
        diag_state = copy.deepcopy(fallback_policy.DIAG)
        immunity_state = copy.deepcopy(fallback_policy.TEMP_IMMUNITY)
        values: dict[int, list[float]] = {index: [] for index in candidates}
        leaves: dict[int, list[dict[str, Any]]] = {
            index: [] for index in candidates
        }
        started_any = False
        began_at = time.perf_counter()
        try:
            for salt in range(DETERMINIZATIONS):
                started = False
                try:
                    hidden = _hidden_state(observation, perspective, salt)
                    root = self.search.begin(observation, hidden)
                    started = True
                    started_any = True
                    root_id = int(root["searchId"])
                    for candidate in candidates:
                        self.stats["branches"] += 1
                        result = self._rollout(
                            root_id, candidate, perspective, ranker,
                            ranker_state,
                        )
                        if result is None:
                            self.stats["incomplete_branches"] += 1
                            continue
                        leaf, value = result
                        leaves[candidate].append(leaf)
                        values[candidate].append(float(value))
                except Exception:
                    self.stats["branch_errors"] += 1
                finally:
                    if started:
                        try:
                            self.search.end()
                        except Exception:
                            self.stats["branch_errors"] += 1
        finally:
            elapsed = time.perf_counter() - began_at
            if started_any:
                self._searched_turns.add(key)
                self.stats["searched"] += 1
                self.budget.charge(elapsed)
            ranker.restore_dynamic_state(ranker_state)
            fallback_policy.DIAG.clear()
            fallback_policy.DIAG.update(diag_state)
            fallback_policy.TEMP_IMMUNITY.clear()
            fallback_policy.TEMP_IMMUNITY.update(immunity_state)

        if any(len(values[index]) != DETERMINIZATIONS for index in candidates):
            self.stats["skip_incomplete"] += 1
            return proposed
        winners = []
        for sample in range(DETERMINIZATIONS):
            winners.append(max(
                candidates,
                key=lambda index: (
                    values[index][sample], scores.get(index, float("-inf"))
                ),
            ))
        if len(set(winners)) != 1:
            self.stats["skip_determinization_disagreement"] += 1
            return proposed
        best = winners[0]
        gains = [
            values[best][sample] - values[proposed][sample]
            for sample in range(DETERMINIZATIONS)
        ]
        if best == proposed or min(gains) < -1e-9:
            self.stats["skip_not_dominant"] += 1
            return proposed
        if not all(
            self._safe_against(
                leaves[best][sample], leaves[proposed][sample], perspective
            )
            for sample in range(DETERMINIZATIONS)
        ):
            self.stats["skip_prize_safety"] += 1
            return proposed
        mean_gain = sum(gains) / len(gains)
        if mean_gain < MIN_MEAN_VALUE_GAIN:
            self.stats["skip_small_gain"] += 1
            return proposed

        self.stats["overrides"] += 1
        self.stats["value_gain_milli"] += int(round(mean_gain * 1000.0))
        if len(self._records) < 24:
            self._records.append({
                "turn": turn,
                "from_index": proposed,
                "to_index": best,
                "peer_index": peer_choice,
                "values_from": [round(value, 6) for value in values[proposed]],
                "values_to": [round(value, 6) for value in values[best]],
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
            "min_mean_value_gain": MIN_MEAN_VALUE_GAIN,
            "records": list(self._records),
        })
        return output
