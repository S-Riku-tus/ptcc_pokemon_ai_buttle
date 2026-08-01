"""Standard-library inference for the compact Lopunny imitation models."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Any

from imitation_features import (
    decision_features,
    encode_action_type,
    observation_features,
    option_features,
    state_features,
)


HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str) -> dict[str, Any]:
    with open(os.path.join(HERE, name), encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("format") != "lightgbm_tree_v2" or not model.get("trees"):
        raise ValueError(f"unsupported or empty model: {name}")
    return model


def _tree_score(row: list[float], model: dict[str, Any]) -> float:
    total = 0.0
    for tree in model["trees"]:
        node = tree
        while "v" not in node:
            value = row[node["f"]]
            if value != value:
                go_left = bool(node.get("x", True))
            elif node.get("d", "<=") == "==":
                go_left = int(round(value)) in node.get("c", ())
            else:
                go_left = value <= node["t"]
            node = node["l"] if go_left else node["r"]
        total += float(node["v"])
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


def _clamped_bounds(select: dict[str, Any], size: int) -> tuple[int, int]:
    minimum = max(0, min(int(select.get("minCount") or 0), size))
    maximum = max(
        minimum, min(int(select.get("maxCount") or 0), size)
    )
    return minimum, maximum


def _legal_fallback(select: dict[str, Any]) -> list[int]:
    options = list(select.get("option") or [])
    minimum, _ = _clamped_bounds(select, len(options))
    return list(range(minimum))


def _simple_scores(options: list[dict[str, Any]]) -> list[float]:
    """Low-time deterministic fallback; it can never emit an illegal index."""
    priority = {
        13: 900.0,  # attack
        10: 800.0,  # ability
        9: 700.0,   # evolve
        8: 600.0,   # attach
        7: 500.0,   # play
        12: 250.0,  # retreat
        1: 100.0,   # yes
        2: 90.0,    # no
        14: 0.0,    # end
    }
    return [priority.get(int(option.get("type", -1)), 50.0) for option in options]


class ImitationRuntime:
    def __init__(self) -> None:
        self.ranker: dict[str, Any] | None = None
        self.count_model: dict[str, Any] | None = None
        self.errors: dict[str, str] = {}
        self.diag: Counter[str] = Counter()
        for attribute, filename in (
            ("ranker", "ranker_model.json"),
            ("count_model", "count_model.json"),
        ):
            try:
                setattr(self, attribute, _load(filename))
            except Exception as exc:  # submission must still return a legal move
                self.errors[attribute] = f"{type(exc).__name__}: {exc}"

    def reset(self) -> None:
        self.diag.clear()

    def snapshot(self) -> dict[str, Any]:
        decisions = max(1, self.diag["decisions"])
        return {
            **dict(self.diag),
            "runtime_scope": "all_select_contexts",
            "ranker_loaded": self.ranker is not None,
            "count_model_loaded": self.count_model is not None,
            "ranker_trees": len((self.ranker or {}).get("trees") or []),
            "count_trees": len((self.count_model or {}).get("trees") or []),
            "average_inference_ms": self.diag["inference_us"] / decisions / 1000.0,
            "errors": dict(self.errors),
        }

    def _model_rows(
        self,
        observation: dict[str, Any],
        options: list[dict[str, Any]],
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        if self.ranker is None:
            return [], []
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        base = state_features(current)
        base.update(observation_features(observation))
        rows: list[list[float]] = []
        raw_features: list[dict[str, Any]] = []
        for position, option in enumerate(options):
            feature = option_features(
                current,
                select,
                option,
                base_state=base,
                option_position=position,
            )
            feature["action_type"] = encode_action_type(
                str(feature.get("action_type") or "other")
            )
            raw_features.append(feature)
            rows.append([
                float(feature.get(name, -1))
                for name in self.ranker["feature_names"]
            ])
        return rows, raw_features

    def _pick_count(
        self,
        observation: dict[str, Any],
        minimum: int,
        maximum: int,
    ) -> int:
        if minimum == maximum:
            return minimum
        if self.count_model is None:
            self.diag["count_model_fallback"] += 1
            return maximum
        feature = decision_features(observation)
        row = [
            float(feature.get(name, -1))
            for name in self.count_model["feature_names"]
        ]
        prediction = _tree_score(row, self.count_model)
        if not math.isfinite(prediction):
            self.diag["count_nonfinite"] += 1
            return maximum
        count = int(round(prediction))
        return max(minimum, min(maximum, count))

    def choose(self, observation: dict[str, Any]) -> list[int]:
        started = time.perf_counter()
        self.diag["decisions"] += 1
        select = observation.get("select") or {}
        options = [
            option for option in (select.get("option") or [])
            if isinstance(option, dict)
        ]
        minimum, maximum = _clamped_bounds(select, len(options))
        if not options or maximum == 0:
            self.diag["empty_or_zero"] += 1
            return []
        if minimum == maximum == len(options):
            self.diag["all_mandatory"] += 1
            return list(range(len(options)))

        try:
            low_time = float(observation.get("remainingOverageTime") or 600) < 2.0
            if self.ranker is None or low_time:
                scores = _simple_scores(options)
                raw_features: list[dict[str, Any]] = []
                self.diag["time_fallback" if low_time else "model_fallback"] += 1
            else:
                rows, raw_features = self._model_rows(observation, options)
                scores = [_tree_score(row, self.ranker) for row in rows]
                self.diag["model_selected"] += 1
            count = self._pick_count(observation, minimum, maximum)
            order = sorted(range(len(options)), key=lambda index: (-scores[index], index))

            # Strength-preserving shell.  Mega Lopunny's two attacks always do
            # at least 60/160, so a visible lethal is never deferred and END is
            # never chosen while the active Lopunny can attack.  In normal
            # teacher-like states these agree with the learned top choice.
            if int(select.get("context", -1)) == 0 and count == 1 and raw_features:
                lethal = [
                    index for index, feature in enumerate(raw_features)
                    if int(feature.get("lopunny_attack_lethal_estimate", 0)) == 1
                ]
                if lethal:
                    best = max(lethal, key=lambda index: (scores[index], -index))
                    if order[0] != best:
                        self.diag["lethal_guard"] += 1
                    order = [best] + [index for index in order if index != best]
                elif (
                    int(options[order[0]].get("type", -1)) == 14
                    and int(raw_features[order[0]].get("has_ready_active_lopunny", 0)) == 1
                ):
                    attacks = [
                        index for index, option in enumerate(options)
                        if int(option.get("type", -1)) == 13
                    ]
                    if attacks:
                        best = max(attacks, key=lambda index: (scores[index], -index))
                        order = [best] + [index for index in order if index != best]
                        self.diag["attack_continuity_guard"] += 1

            action = order[:count]
            if (
                len(action) < minimum
                or len(action) > maximum
                or len(set(action)) != len(action)
                or any(index < 0 or index >= len(options) for index in action)
            ):
                self.diag["legality_fallback"] += 1
                return _legal_fallback(select)
            self.diag["inference_us"] += int(
                (time.perf_counter() - started) * 1_000_000
            )
            return action
        except Exception as exc:
            self.diag["runtime_error"] += 1
            self.diag[f"runtime_{type(exc).__name__}"] += 1
            return _legal_fallback(select)
