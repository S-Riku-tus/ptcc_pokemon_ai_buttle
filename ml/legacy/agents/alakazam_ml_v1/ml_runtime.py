from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Any

from ml_features import candidate_card, option_features


HARD_FALLBACK_ACTIONS = {"boss", "retreat", "xerosic", "hammer"}


def _model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "ranker_model.json"),
        "ranker_model.json",
        "/kaggle_simulations/agent/ranker_model.json",
    ):
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
            elif node.get("d", "<=") == "==":
                go_left = int(round(value)) in node.get("c", [])
            else:
                go_left = value <= node["t"]
            node = node["l"] if go_left else node["r"]
        total += node["v"]
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


def _probabilities(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    scaled = [score / max(temperature, 1e-6) for score in scores]
    peak = max(scaled)
    values = [math.exp(max(-50.0, min(50.0, score - peak))) for score in scaled]
    total = max(sum(values), 1e-12)
    return [value / total for value in values]


def _legal(action: list[int], select: dict[str, Any]) -> bool:
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    maximum = int(select.get("maxCount") if select.get("maxCount") is not None else len(options))
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
    )


def _body_and_deck(observation: dict[str, Any]) -> tuple[int, int]:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex") or 0)
    player = players[your] if your < len(players) else {}
    return len(player.get("active") or []) + len(player.get("bench") or []), int(player.get("deckCount") or 0)


class HybridRanker:
    """Conservative legal-option ranker layered over the proven v12 fallback."""

    def __init__(self, attacks: dict[int, dict[str, Any]] | None = None, threshold: float = 0.55):
        del attacks
        self.threshold_override = float(threshold)
        self.model_error = ""
        self.model: dict[str, Any] | None = None
        self.diag = Counter()
        try:
            with open(_model_path(), encoding="utf-8") as handle:
                model = json.load(handle)
            if model.get("format") not in {"lightgbm_tree_v1", "lightgbm_tree_v2"}:
                raise ValueError("unsupported model format")
            if not model.get("trees") or not model.get("feature_names"):
                raise ValueError("empty distilled model")
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
        # The expanded corpus deliberately trains only ACTIVE MAIN decisions
        # with one required option. Nested search/target selections remain with
        # the deterministic fallback.
        if int(select.get("type", -1)) != 0 or int(select.get("context", -1)) != 0:
            return self._fallback(fallback_action, "outside_training_scope")
        if int(select.get("minCount") or 0) != 1 or int(select.get("maxCount") or 0) != 1:
            return self._fallback(fallback_action, "multi_select")
        if len(options) < 2:
            return self._fallback(fallback_action, "forced")
        try:
            current = observation.get("current") or {}
            action_map = {str(k): int(v) for k, v in (self.model.get("action_type_map") or {}).items()}
            rows: list[list[float]] = []
            actions: list[str] = []
            safe: list[bool] = []
            body_count, deck_count = _body_and_deck(observation)
            for option in options:
                feature = option_features(current, select, option)
                action = str(feature.get("action_type") or "other")
                actions.append(action)
                feature["action_type"] = action_map.get(action, -1)
                rows.append([float(feature.get(name, -1)) for name in self.model["feature_names"]])
                card = candidate_card(current, option) or {}
                card_id = int(card.get("id", -1))
                safe.append(not (action == "ability" and card_id == 66 and (body_count <= 1 or deck_count <= 3)))

            # A proven fallback immediate KO must never be spent for extra setup.
            if len(fallback_action) == 1 and 0 <= fallback_action[0] < len(options):
                fallback_feature = option_features(current, select, options[fallback_action[0]])
                if int(fallback_feature.get("attack_lethal_estimate", 0)):
                    return self._fallback(fallback_action, "lethal_guard")

            scores = [
                _tree_score(row, self.model) if safe[index] else float("-inf")
                for index, row in enumerate(rows)
            ]
            finite_indices = [index for index, score in enumerate(scores) if math.isfinite(score)]
            if not finite_indices:
                return self._fallback(fallback_action, "all_filtered")
            finite_scores = [scores[index] for index in finite_indices]
            probs = _probabilities(finite_scores, float(self.model.get("temperature", 1.0)))
            order_local = sorted(range(len(finite_indices)), key=lambda i: finite_scores[i], reverse=True)
            top_local = order_local[0]
            top = finite_indices[top_local]
            confidence = probs[top_local]
            second_probability = probs[order_local[1]] if len(order_local) > 1 else 0.0
            margin = confidence - second_probability
            predicted_action = actions[top]

            if predicted_action in HARD_FALLBACK_ACTIONS:
                return self._fallback(fallback_action, f"hard_{predicted_action}")
            thresholds = self.model.get("action_type_thresholds") or {}
            base_threshold = max(
                float(self.model.get("fallback_probability", 0.55)),
                self.threshold_override,
            )
            probability_threshold = max(base_threshold, float(thresholds.get(predicted_action, base_threshold)))
            margin_threshold = float(self.model.get("fallback_margin", 0.12))
            if confidence < probability_threshold or margin < margin_threshold:
                self.diag["low_confidence"] += 1
                return self._fallback(fallback_action, "low_confidence")

            action = [top]
            if not _legal(action, select):
                return self._fallback(fallback_action, "legality")
            self.diag["inference_us"] += int((time.perf_counter() - started) * 1_000_000)
            self.diag["model_selected"] += 1
            self.diag[f"model_{predicted_action}"] += 1
            if action != list(fallback_action):
                self.diag["model_override"] += 1
            return action
        except Exception as exc:
            self.diag["runtime_error"] += 1
            self.diag[f"runtime_{type(exc).__name__}"] += 1
            return self._fallback(fallback_action, "exception")
