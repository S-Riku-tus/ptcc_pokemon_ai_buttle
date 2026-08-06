"""Counterfactual full-turn value search for Grimmsnarl v7.

The v6 imitation ranker remains the candidate generator.  Only low-margin
MAIN decisions from turn five onward are eligible.  Each top candidate is
advanced with cabt's real Search API through the rest of the turn.  A public
state-value head scores the resulting next-turn board from our original seat.
At most one such comparison is made per turn.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import json
import math
import os
import random
from collections import Counter
from typing import Any

import fallback_policy
import value_features
from ml_runtime import _prepare, _resolve, tree_score


MAIN_CONTEXT = 0
MIN_TURN = 5
TOP_K = 3
MAX_RANK_MARGIN = 3.0
MIN_VALUE_GAIN = 0.06
MAX_ROLLOUT_STEPS = 48


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
    """Use the official wrapper on Kaggle and the low-level local shim here."""

    def __init__(self) -> None:
        from cg import api

        self.api = api
        self.high_level = all(hasattr(api, name) for name in (
            "search_begin", "search_step", "search_end"
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
            ctypes.c_void_p, ctypes.c_int, pointer, ctypes.c_int
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
            argument: Any = observation
            try:
                result = self.api.search_begin(argument, *hidden, False)
            except (AttributeError, TypeError):
                argument = self.api.to_observation_class(observation)
                result = self.api.search_begin(argument, *hidden, False)
            return self._state(result)

        arrays = [(ctypes.c_int * len(values))(*values) for values in hidden]
        encoded = str(observation["search_begin_input"]).encode("ascii")
        raw = self.lib.SearchBegin(
            self.pointer, encoded, len(encoded), *arrays, 0
        )
        if not raw:
            raise RuntimeError("SearchBegin returned null")
        return self._state(json.loads(raw))

    def step(self, search_id: int, selection: list[int]) -> dict:
        if self.high_level:
            return self._state(self.api.search_step(search_id, selection))
        array = (ctypes.c_int * len(selection))(*selection)
        raw = self.lib.SearchStep(
            self.pointer, int(search_id), array, len(selection)
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
        exp = math.exp(max(raw, -50.0))
        return exp / (1.0 + exp)


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


def _hidden_state(observation: dict[str, Any], perspective: int) -> list[list[int]]:
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
    pool = []
    for card_id, count in remaining.items():
        pool.extend([card_id] * max(int(count), 0))
    seed = (
        int(current.get("turn", 0) or 0) * 1009
        + int(current.get("turnActionCount", 0) or 0) * 97
        + perspective * 17
    )
    random.Random(seed).shuffle(pool)
    needed = int(me.get("deckCount", 0) or 0) + len(me.get("prize") or [])
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
    own_deck = pool[cursor:cursor + int(me.get("deckCount", 0) or 0)]
    if len(own_deck) < int(me.get("deckCount", 0) or 0):
        own_deck.extend(_repeat(
            deck, int(me.get("deckCount", 0) or 0) - len(own_deck), seed
        ))

    # Opponent identities affect only effects which explicitly access their
    # hidden zones.  A mirror prior is deterministic and always engine-legal;
    # public board identities still come from search_begin_input itself.
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


class SearchPlanner:
    def __init__(self) -> None:
        self.disabled = os.environ.get("GRIMMSNARL_VALUE_DISABLE") == "1"
        self.value = _ValueModel()
        self.search = None if self.disabled else _SearchApi()
        self.reset()

    def reset(self) -> None:
        self._searched_turns: set[tuple[int, int]] = set()
        self._override_records: list[dict[str, Any]] = []
        self.stats = {
            "considered": 0,
            "searched": 0,
            "branches": 0,
            "overrides": 0,
            "branch_errors": 0,
            "incomplete_branches": 0,
            "skip_early": 0,
            "skip_confident": 0,
            "skip_already_searched_turn": 0,
            "skip_planner_guard": 0,
            "skip_small_gain": 0,
            "value_gain_sum": 0.0,
            "positive_gain_count": 0,
            "positive_gain_sum": 0.0,
            "best_gain_max": 0.0,
            "nonzero_value_spread_count": 0,
            "value_spread_max": 0.0,
        }

    def _branch_choice(self, observation: dict[str, Any], ranker, rule_agent):
        select = observation.get("select") or {}
        try:
            rule_choice = list(rule_agent(observation))
        except Exception:
            rule_choice = _simple_legal(select)
        if ranker.is_scorable(select):
            index = ranker.choose(observation)
            if index is not None:
                ranker.commit(index)
                return [index]
        chosen = rule_choice[0] if len(rule_choice) == 1 else None
        if chosen is not None:
            ranker.observe_external(observation, chosen)
        return rule_choice

    def _rollout(
        self,
        root_id: int,
        candidate: int,
        root_turn: int,
        perspective: int,
        ranker,
        root_ranker_state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, float | None]:
        ranker.restore_dynamic_state(root_ranker_state)
        # Teacher-forced evaluation suppresses the live commit, but a branch
        # still needs its hypothetical action in the turn-history columns.
        ranker.teacher_forced = False
        ranker.commit(candidate)
        branch_diag = fallback_policy._fresh_diag()
        rule_agent = fallback_policy.make_agent(
            fallback_policy.GrimmsnarlPolicy,
            fallback_policy.MY_DECK,
            branch_diag,
        )
        state = self.search.step(root_id, [candidate])
        for _ in range(MAX_ROLLOUT_STEPS + 1):
            observation = state.get("observation") or {}
            current = observation.get("current") or {}
            result = int(current.get("result", -1))
            if result >= 0:
                return observation, (
                    1.0 if result == perspective
                    else 0.0 if result in (0, 1) else 0.5
                )
            if (
                int(current.get("turn", -1)) != root_turn
                or int(current.get("yourIndex", -1)) != perspective
            ):
                return observation, self.value.probability(
                    observation, perspective
                )
            select = observation.get("select") or {}
            selection = self._branch_choice(observation, ranker, rule_agent)
            state = self.search.step(int(state["searchId"]), selection)
        return None, None

    def adjust(
        self,
        observation: dict[str, Any],
        proposed: int,
        scores: dict[int, float],
        ranker,
    ) -> int:
        if self.disabled or self.search is None or len(scores) < 2:
            return proposed
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != MAIN_CONTEXT:
            return proposed
        self.stats["considered"] += 1
        current = observation.get("current") or {}
        turn = int(current.get("turn", 0) or 0)
        perspective = int(current.get("yourIndex", 0) or 0)
        turn_key = (turn, perspective)
        if turn_key in self._searched_turns:
            self.stats["skip_already_searched_turn"] += 1
            return proposed
        if turn < MIN_TURN:
            self.stats["skip_early"] += 1
            return proposed
        ordered = sorted(scores, key=lambda index: scores[index], reverse=True)
        if proposed != ordered[0]:
            # The arithmetic planner has already proved the ranker's argmax
            # dominated.  A statistical value head must not undo that guard.
            self.stats["skip_planner_guard"] += 1
            return proposed
        margin = scores[ordered[0]] - scores[ordered[1]]
        if margin > MAX_RANK_MARGIN:
            self.stats["skip_confident"] += 1
            return proposed
        options = list(select.get("option") or [])
        if proposed >= len(options):
            return proposed
        candidates = [
            index for index in ordered[:TOP_K]
            if index < len(options)
            and scores[ordered[0]] - scores[index] <= MAX_RANK_MARGIN
        ]
        if len(candidates) < 2 or not observation.get("search_begin_input"):
            return proposed

        ranker_state = ranker.save_dynamic_state()
        diag_state = copy.deepcopy(fallback_policy.DIAG)
        immunity_state = copy.deepcopy(fallback_policy.TEMP_IMMUNITY)
        values: dict[int, float] = {}
        started = False
        try:
            hidden = _hidden_state(
                observation, int(current.get("yourIndex", 0) or 0)
            )
            root = self.search.begin(observation, hidden)
            started = True
            self._searched_turns.add(turn_key)
            self.stats["searched"] += 1
            root_id = int(root["searchId"])
            for candidate in candidates:
                self.stats["branches"] += 1
                try:
                    _leaf, value = self._rollout(
                        root_id, candidate, turn,
                        int(current.get("yourIndex", 0) or 0), ranker,
                        ranker_state,
                    )
                    if value is None:
                        self.stats["incomplete_branches"] += 1
                    else:
                        values[candidate] = value
                except Exception:
                    self.stats["branch_errors"] += 1
        except Exception:
            self.stats["branch_errors"] += 1
        finally:
            if started:
                try:
                    self.search.end()
                except Exception:
                    self.stats["branch_errors"] += 1
            ranker.restore_dynamic_state(ranker_state)
            fallback_policy.DIAG.clear()
            fallback_policy.DIAG.update(diag_state)
            fallback_policy.TEMP_IMMUNITY.clear()
            fallback_policy.TEMP_IMMUNITY.update(immunity_state)

        if proposed not in values or len(values) < 2:
            return proposed
        spread = max(values.values()) - min(values.values())
        if spread > 1e-9:
            self.stats["nonzero_value_spread_count"] += 1
            self.stats["value_spread_max"] = max(
                self.stats["value_spread_max"], spread
            )
        best = max(values, key=lambda index: (values[index], scores[index]))
        gain = values[best] - values[proposed]
        if gain > 0:
            self.stats["positive_gain_count"] += 1
            self.stats["positive_gain_sum"] += gain
            self.stats["best_gain_max"] = max(
                self.stats["best_gain_max"], gain
            )
        if best == proposed or gain < MIN_VALUE_GAIN:
            self.stats["skip_small_gain"] += 1
            return proposed
        self.stats["overrides"] += 1
        self.stats["value_gain_sum"] += gain
        pending = ranker_state.get("_pending") or []
        before = pending[proposed] if proposed < len(pending) else {}
        after = pending[best] if best < len(pending) else {}
        if len(self._override_records) < 24:
            self._override_records.append({
                "turn": turn,
                "from_index": proposed,
                "to_index": best,
                "from_action_type": int(before.get("action_type_id", -1)),
                "to_action_type": int(after.get("action_type_id", -1)),
                "from_card_id": int(before.get("candidate_card_id", -1)),
                "to_card_id": int(after.get("candidate_card_id", -1)),
                "from_value": round(float(values[proposed]), 6),
                "to_value": round(float(values[best]), 6),
                "gain": round(float(gain), 6),
            })
        return best

    def snapshot(self) -> dict[str, Any]:
        output = dict(self.stats)
        for name in (
            "value_gain_sum", "positive_gain_sum", "best_gain_max",
            "value_spread_max",
        ):
            output[name] = round(float(output[name]), 6)
        output["enabled"] = not self.disabled
        output["min_turn"] = MIN_TURN
        output["max_rank_margin"] = MAX_RANK_MARGIN
        output["min_value_gain"] = MIN_VALUE_GAIN
        output["override_records"] = list(self._override_records)
        return output
