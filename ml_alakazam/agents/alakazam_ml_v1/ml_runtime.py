from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Any

from ml_features import action_type, candidate_features, state_features


FIRST_STAGE_ACTIONS = {"attack", "ability", "evolve"}
SEARCH_CONTEXTS = {7, 9, 10, 18, 19, 24, 25, 26, 27, 29, 31, 32}


def _model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "ranker_model.json"),
        "ranker_model.json",
        "/kaggle_simulations/agent/ranker_model.json",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("ranker_model.json not found")


def _tree_score(features: list[float], model: dict[str, Any]) -> float:
    total = 0.0
    for tree in model["trees"]:
        node = tree
        while "v" not in node:
            value = features[node["f"]]
            if value != value:
                go_left = node.get("x", True)
            elif node.get("d", "<=") == "<=":
                go_left = value <= node["t"]
            else:
                go_left = value == node["t"]
            node = node["l"] if go_left else node["r"]
        total += node["v"]
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


def _softmax_confidence(scores: list[float]) -> float:
    if not scores:
        return 0.0
    peak = max(scores)
    values = [math.exp(max(-50.0, min(50.0, score - peak))) for score in scores]
    return max(values) / max(sum(values), 1e-12)


def _legal(action: list[int], select: dict[str, Any]) -> bool:
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    maximum = int(select.get("maxCount") if select.get("maxCount") is not None else len(options))
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
    )


def _body_count(observation: dict[str, Any]) -> int:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your_index = int(current.get("yourIndex") or 0)
    player = players[your_index] if your_index < len(players) else {}
    return len(player.get("active") or []) + len(player.get("bench") or [])


class HybridRanker:
    def __init__(self, attacks: dict[int, dict[str, Any]] | None = None, threshold: float = 0.58):
        self.threshold = threshold
        self.attacks = attacks or {}
        self.model_error = ""
        self.model: dict[str, Any] | None = None
        self.diag = Counter()
        try:
            with open(_model_path(), encoding="utf-8") as handle:
                model = json.load(handle)
            if model.get("format") != "lightgbm_tree_v1":
                raise ValueError("unsupported model format")
            self.model = model
        except Exception as exc:
            self.model_error = f"{type(exc).__name__}: {exc}"

    def reset(self) -> None:
        self.diag.clear()

    def snapshot(self) -> dict[str, Any]:
        decisions = max(1, self.diag["decisions"])
        return {
            **dict(self.diag),
            "model_loaded": self.model is not None,
            "model_error": self.model_error,
            "model_rate": self.diag["model_selected"] / decisions,
            "fallback_rate": self.diag["fallback"] / decisions,
            "low_confidence_rate": self.diag["low_confidence"] / decisions,
            "average_inference_ms": self.diag["inference_us"] / decisions / 1000.0,
        }

    def _fallback(self, fallback_action: list[int], reason: str) -> list[int]:
        self.diag["fallback"] += 1
        self.diag[f"fallback_{reason}"] += 1
        return list(fallback_action)

    def choose(self, observation: dict[str, Any], fallback_action: list[int]) -> list[int]:
        self.diag["decisions"] += 1
        started = time.perf_counter()
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        if self.model is None or not options:
            return self._fallback(fallback_action, "model_unavailable")
        if float(observation.get("remainingOverageTime") or 600) < 2.0:
            return self._fallback(fallback_action, "time_guard")
        try:
            state = state_features(observation)
            runtime_options = list(options)
            none_index = None
            if int(select.get("minCount") or 0) == 0:
                none_index = len(runtime_options)
                runtime_options.append({"type": "__none__"})
            rows = []
            identities = []
            safe = []
            body_count = _body_count(observation)
            current = observation.get("current") or {}
            players = current.get("players") or [{}, {}]
            your_index = int(current.get("yourIndex") or 0)
            player = players[your_index] if your_index < len(players) else {}
            deck_count = int(player.get("deckCount") or 0)
            for index, raw in enumerate(runtime_options):
                option = raw if isinstance(raw, dict) else {"type": str(raw)}
                candidate, identity = candidate_features(
                    observation, option, index, cards=None, attacks=self.attacks
                )
                values = {**state, **candidate}
                rows.append([float(values.get(name, 0.0)) for name in self.model["feature_names"]])
                identities.append(identity)
                option_action = action_type(identity)
                is_safe = True
                if option_action == "ability" and int(identity.get("card_id") or 0) == 66:
                    is_safe = body_count > 1 and deck_count > 3
                safe.append(is_safe)

            # Preserve a baseline-confirmed immediate KO rather than spending the hand first.
            for index in fallback_action:
                if 0 <= index < len(rows):
                    candidate, _ = candidate_features(
                        observation,
                        options[index] if isinstance(options[index], dict) else {"type": str(options[index])},
                        index, cards=None, attacks=self.attacks,
                    )
                    if candidate["ko_possible"]:
                        return self._fallback(fallback_action, "lethal_guard")

            scores = [
                _tree_score(row, self.model) if safe[index] else float("-inf")
                for index, row in enumerate(rows)
            ]
            context = select.get("context")
            if context == 0:
                if not fallback_action:
                    return self._fallback(fallback_action, "outside_scope")
                fallback_type = action_type(identities[fallback_action[0]])
                if fallback_type not in FIRST_STAGE_ACTIONS:
                    return self._fallback(fallback_action, "outside_scope")
                scope = [action_type(identity) == fallback_type for identity in identities]
            elif context in SEARCH_CONTEXTS:
                scope = [
                    action_type(identity) in {"card", "none", "yes", "no"}
                    for identity in identities
                ]
            else:
                return self._fallback(fallback_action, "outside_scope")
            for index in range(len(scores)):
                if not scope[index]:
                    scores[index] = float("-inf")
            finite = [score for score in scores if math.isfinite(score)]
            if not finite:
                return self._fallback(fallback_action, "all_filtered")
            ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            top = ranked[0]
            confidence = _softmax_confidence(finite)
            self.diag["inference_us"] += int((time.perf_counter() - started) * 1_000_000)
            if confidence < self.threshold:
                self.diag["low_confidence"] += 1
                return self._fallback(fallback_action, "low_confidence")
            if top == none_index:
                action: list[int] = []
            else:
                desired = max(int(select.get("minCount") or 0), len(fallback_action))
                desired = min(desired or 1, int(select.get("maxCount") or 1))
                action = [index for index in ranked if index != none_index and safe[index]][:desired]
            if not _legal(action, select):
                return self._fallback(fallback_action, "legality")
            self.diag["model_selected"] += 1
            if action != list(fallback_action):
                self.diag["model_override"] += 1
            return action
        except Exception:
            self.diag["runtime_error"] += 1
            return self._fallback(fallback_action, "exception")
