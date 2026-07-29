from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Any

import fallback_policy
from ml_features import candidate_card, candidate_target, option_features


def _artifact_path(name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, name),
        name,
        f"/kaggle_simulations/agent/{name}",
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{name} not found")


def _load_model(name: str) -> dict[str, Any]:
    with open(_artifact_path(name), encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("format") not in {"lightgbm_tree_v1", "lightgbm_tree_v2"}:
        raise ValueError(f"unsupported model format: {name}")
    if not model.get("trees") or not model.get("feature_names"):
        raise ValueError(f"empty model: {name}")
    return model


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
    values = [
        math.exp(max(-50.0, min(50.0, score - peak)))
        for score in scaled
    ]
    total = max(sum(values), 1e-12)
    return [value / total for value in values]


def _legal(action: list[int], select: dict[str, Any]) -> bool:
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    maximum = int(
        select.get("maxCount")
        if select.get("maxCount") is not None
        else len(options)
    )
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(
            isinstance(index, int) and 0 <= index < len(options)
            for index in action
        )
    )


def _context_from_feature(
    option: dict[str, Any],
    feature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "option": option,
        "action_type": str(feature.get("_action_name") or "other"),
        "card_id": int(feature.get("candidate_card_id", -1)),
        "target_id": int(feature.get("candidate_target_id", -1)),
        "breaks_current_ko": bool(
            feature.get("breaks_current_ko_estimate", 0)
        ),
        "attack_lethal": bool(feature.get("attack_lethal_estimate", 0)),
    }


def _feature_semantic_key(feature: dict[str, Any]) -> tuple[Any, ...]:
    """The exact copy-collapsing label used by v29 teacher training."""
    return (
        int(feature.get("option_type", -1)),
        int(feature.get("candidate_card_id", -1)),
        int(feature.get("candidate_attack_id", -1)),
        int(feature.get("candidate_target_id", -1)),
        int(feature.get("candidate_target_hp", -1)),
        int(feature.get("candidate_target_max_hp", -1)),
        int(feature.get("candidate_target_energy", -1)),
        int(feature.get("candidate_target_special_energy", -1)),
        int(feature.get("candidate_inplay_area", -1)),
    )


def _rank_positions(scores: list[float]) -> tuple[list[int], list[int]]:
    order = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )
    positions = [0] * len(scores)
    for position, index in enumerate(order):
        positions[index] = position
    return order, positions


def _fallback_policy_scores(
    observation: dict[str, Any],
    option_count: int,
) -> list[float]:
    try:
        parsed = fallback_policy.to_observation_class(observation)
        policy = fallback_policy.AlakazamPolicy(parsed)
        scores = [
            float(policy._score(option))
            for option in parsed.select.option
        ]
        if len(scores) == option_count:
            return scores
    except Exception:
        pass
    return [0.0] * option_count


def _candidate_safety_reason(
    context: dict[str, Any],
    fallback_context: dict[str, Any],
    state: dict[str, Any],
    *,
    attack_is_available: bool,
) -> str | None:
    """Keep only small, evidence-backed hard guards above teacher imitation."""
    action = context["action_type"]
    if action == "other":
        return "unmodelled_other"
    if context["breaks_current_ko"]:
        return "breaks_current_ko"

    # Nested target selection remains with fallback, so a fallback MAIN Boss
    # action owns a complete deterministic terminal/prize route.
    if fallback_context["action_type"] == "boss" and action != "boss":
        return "preserve_fallback_boss_route"

    # Never decline an already available Alakazam attack by ending the turn.
    if (
        action == "end"
        and attack_is_available
        and int(state.get("has_ready_active_alakazam", 0)) == 1
    ):
        return "end_with_ready_attack"

    # v28 ladder autopsy confirmed that cycling the last Dudunsparce out of a
    # two-body board creates avoidable board-out losses.
    if (
        action == "ability"
        and context["card_id"] == 66
        and int(state.get("self_board_count", 0)) <= 2
    ):
        return "dudunsparce_body_floor"

    return None


class HybridRanker:
    """Rank-weighted v29 teacher policy with deterministic safety fallbacks."""

    def __init__(
        self,
        attacks: dict[int, dict[str, Any]] | None = None,
        threshold: float = 0.20,
    ):
        del attacks
        self.threshold_override = float(threshold)
        self.enable_override = (
            os.environ.get("ALAKAZAM_ML_V29_ENABLE_OVERRIDE", "1") == "1"
        )
        self.model_error = ""
        self.legacy_model_error = ""
        self.model: dict[str, Any] | None = None
        self.legacy_model: dict[str, Any] | None = None
        self.diag = Counter()
        try:
            self.model = _load_model("ranker_model.json")
        except Exception as exc:
            self.model_error = f"{type(exc).__name__}: {exc}"
        try:
            self.legacy_model = _load_model("legacy_ranker_model.json")
        except Exception as exc:
            self.legacy_model_error = f"{type(exc).__name__}: {exc}"

    def reset(self) -> None:
        self.diag.clear()

    def snapshot(self) -> dict[str, Any]:
        decisions = max(1, self.diag["decisions"])
        return {
            **dict(self.diag),
            "runtime_scope": "v29_rank_weighted_residual_main_policy",
            "override_enabled": self.enable_override,
            "model_loaded": self.model is not None,
            "legacy_model_loaded": self.legacy_model is not None,
            "model_error": self.model_error,
            "legacy_model_error": self.legacy_model_error,
            "model_rate": self.diag["model_selected"] / decisions,
            "fallback_rate": self.diag["fallback"] / decisions,
            "low_confidence_rate": self.diag["low_confidence"] / decisions,
            "average_inference_ms": (
                self.diag["inference_us"] / decisions / 1000.0
            ),
        }

    def _fallback(
        self,
        fallback_action: list[int],
        reason: str,
    ) -> list[int]:
        self.diag["fallback"] += 1
        self.diag[f"fallback_{reason}"] += 1
        return list(fallback_action)

    def choose(
        self,
        observation: dict[str, Any],
        fallback_action: list[int],
    ) -> list[int]:
        self.diag["decisions"] += 1
        started = time.perf_counter()
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        if self.model is None or self.legacy_model is None or not options:
            return self._fallback(fallback_action, "model_unavailable")
        if float(observation.get("remainingOverageTime") or 600) < 2.0:
            return self._fallback(fallback_action, "time_guard")
        if (
            int(select.get("type", -1)) != 0
            or int(select.get("context", -1)) != 0
        ):
            return self._fallback(fallback_action, "outside_training_scope")
        if (
            int(select.get("minCount") or 0) != 1
            or int(select.get("maxCount") or 0) != 1
        ):
            return self._fallback(fallback_action, "multi_select")
        if len(options) < 2:
            return self._fallback(fallback_action, "forced")
        if (
            len(fallback_action) != 1
            or not 0 <= fallback_action[0] < len(options)
        ):
            return self._fallback(fallback_action, "fallback_unresolved")

        try:
            current = observation.get("current") or {}
            action_map = {
                str(key): int(value)
                for key, value in (
                    self.model.get("action_type_map") or {}
                ).items()
            }
            features: list[dict[str, Any]] = []
            contexts: list[dict[str, Any]] = []
            for option in options:
                feature = dict(option_features(current, select, option))
                action_name = str(feature.get("action_type") or "other")
                feature["_action_name"] = action_name
                feature["action_type"] = action_map.get(action_name, -1)
                features.append(feature)
                contexts.append(_context_from_feature(option, feature))

            fallback_index = fallback_action[0]
            fallback_context = contexts[fallback_index]
            fallback_key = _feature_semantic_key(features[fallback_index])

            # A proven immediate attack win always dominates imitation.
            if fallback_context["attack_lethal"]:
                return self._fallback(fallback_action, "lethal_guard")

            policy_scores = _fallback_policy_scores(
                observation, len(options)
            )
            policy_order, policy_positions = _rank_positions(policy_scores)
            policy_peak = policy_scores[policy_order[0]]

            legacy_scores: list[float] = []
            for feature in features:
                legacy_row = [
                    float(feature.get(name, -1))
                    for name in self.legacy_model["feature_names"]
                ]
                legacy_scores.append(
                    _tree_score(legacy_row, self.legacy_model)
                )
            legacy_order, legacy_positions = _rank_positions(legacy_scores)
            legacy_peak = legacy_scores[legacy_order[0]]
            fallback_action_type = int(
                features[fallback_index]["action_type"]
            )
            fallback_card_id = int(
                features[fallback_index].get("candidate_card_id", -1)
            )
            fallback_legacy_agree = int(
                fallback_index == legacy_order[0]
            )

            rows: list[list[float]] = []
            for index, feature in enumerate(features):
                feature.update({
                    "fallback_selected": int(
                        _feature_semantic_key(feature) == fallback_key
                    ),
                    "fallback_action_type": fallback_action_type,
                    "fallback_card_id": fallback_card_id,
                    "fallback_policy_score": max(
                        -10_000_000.0,
                        min(10_000_000.0, policy_scores[index]),
                    ),
                    "fallback_policy_score_gap": max(
                        -10_000_000.0,
                        min(
                            10_000_000.0,
                            policy_scores[index] - policy_peak,
                        ),
                    ),
                    "fallback_policy_rank": policy_positions[index],
                    "legacy_ranker_score": legacy_scores[index],
                    "legacy_ranker_score_gap": (
                        legacy_scores[index] - legacy_peak
                    ),
                    "legacy_ranker_rank": legacy_positions[index],
                    "legacy_ranker_selected": int(
                        index == legacy_order[0]
                    ),
                    "fallback_legacy_agree": fallback_legacy_agree,
                })
                rows.append([
                    float(feature.get(name, -1))
                    for name in self.model["feature_names"]
                ])

            scores = [_tree_score(row, self.model) for row in rows]
            order = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )
            probabilities = _probabilities(
                scores, float(self.model.get("temperature", 1.0))
            )
            top = order[0]
            confidence = probabilities[top]
            second_probability = (
                probabilities[order[1]] if len(order) > 1 else 0.0
            )
            margin = confidence - second_probability
            probability_threshold = max(
                float(self.model.get("fallback_probability", 0.20)),
                self.threshold_override,
            )
            margin_threshold = float(
                self.model.get("fallback_margin", 0.0)
            )
            if (
                confidence < probability_threshold
                or margin < margin_threshold
            ):
                self.diag["low_confidence"] += 1
                return self._fallback(fallback_action, "low_confidence")

            state = features[top]
            attack_is_available = any(
                context["action_type"] == "attack"
                for context in contexts
            )
            safety_reason = _candidate_safety_reason(
                contexts[top],
                fallback_context,
                state,
                attack_is_available=attack_is_available,
            )
            if safety_reason is not None:
                self.diag[f"candidate_blocked_{safety_reason}"] += 1
                return self._fallback(
                    fallback_action,
                    f"safety_{safety_reason}",
                )

            action = [top]
            if not _legal(action, select):
                return self._fallback(fallback_action, "legality")
            self.diag["inference_us"] += int(
                (time.perf_counter() - started) * 1_000_000
            )
            self.diag["model_selected"] += 1
            predicted_action = contexts[top]["action_type"]
            self.diag[f"model_{predicted_action}"] += 1
            if not self.enable_override:
                self.diag["shadow_evaluated"] += 1
                return self._fallback(fallback_action, "shadow_only")
            if _feature_semantic_key(features[top]) != fallback_key:
                self.diag["model_override"] += 1
                self.diag[f"model_override_{predicted_action}"] += 1
                if predicted_action != fallback_context["action_type"]:
                    self.diag["model_override_action_type"] += 1
                    self.diag[
                        "model_override_"
                        f"{fallback_context['action_type']}_to_"
                        f"{predicted_action}"
                    ] += 1
            else:
                self.diag["model_agrees_fallback"] += 1
            return action
        except Exception as exc:
            self.diag["runtime_error"] += 1
            self.diag[f"runtime_{type(exc).__name__}"] += 1
            return self._fallback(fallback_action, "exception")
