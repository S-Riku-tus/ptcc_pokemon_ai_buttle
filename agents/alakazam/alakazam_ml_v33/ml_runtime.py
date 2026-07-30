from __future__ import annotations

import json
import math
import os
import time
import zlib
from collections import Counter
from typing import Any

from ml_features import (
    observation_features,
    option_features,
    state_features,
)
from teacher_memory import (
    resolve_semantic_action,
    teacher_memory_keys,
)
from v29_runtime import (
    _candidate_safety_reason,
    _context_from_feature,
    _fallback_policy_scores,
    _feature_semantic_key,
    _legal,
    _load_model,
    _probabilities,
    _rank_positions,
    _tree_score,
)


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


def _load_memory() -> dict[str, Any]:
    with open(_artifact_path("teacher_memory.bin"), "rb") as handle:
        payload = json.loads(zlib.decompress(handle.read()))
    if payload.get("format") != "v30_teacher_memory_v1":
        raise ValueError("unsupported teacher memory")
    return payload


def _selector_tree_score(
    features: list[float],
    model: dict[str, Any],
) -> float:
    """Score v33 artifacts with LightGBM's missing-category semantics."""
    total = 0.0
    for tree in model["trees"]:
        node = tree
        while "v" not in node:
            value = features[node["f"]]
            if value != value:
                go_left = node.get("x", True)
            elif node.get("d", "<=") == "==":
                go_left = (
                    node.get("x", True)
                    if value < 0
                    else int(round(value)) in node.get("c", [])
                )
            else:
                go_left = value <= node["t"]
            node = node["l"] if go_left else node["r"]
        total += node["v"]
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


class HybridRanker:
    """v31 safety/memory shell plus the v33 OOF selector challenger."""

    def __init__(
        self,
        attacks: dict[int, dict[str, Any]] | None = None,
        threshold: float = 0.20,
    ):
        del attacks
        self.threshold_override = float(threshold)
        self.enable_override = (
            os.environ.get(
                "ALAKAZAM_ML_V33_ENABLE_OVERRIDE",
                os.environ.get(
                    "ALAKAZAM_ML_V32_ENABLE_OVERRIDE",
                    os.environ.get(
                        "ALAKAZAM_ML_V31_ENABLE_OVERRIDE",
                        "1",
                    ),
                ),
            )
            == "1"
        )
        self.enable_memory = (
            os.environ.get(
                "ALAKAZAM_ML_V33_ENABLE_MEMORY",
                os.environ.get(
                    "ALAKAZAM_ML_V32_ENABLE_MEMORY",
                    os.environ.get(
                        "ALAKAZAM_ML_V31_ENABLE_MEMORY",
                        "1",
                    ),
                ),
            )
            == "1"
        )
        self.model: dict[str, Any] | None = None
        self.numeric_model: dict[str, Any] | None = None
        self.v29_model: dict[str, Any] | None = None
        self.legacy_model: dict[str, Any] | None = None
        self.memory: dict[str, Any] = {}
        self.selector_model: dict[str, Any] | None = None
        self.selector_bases: list[dict[str, Any]] = []
        self.errors: dict[str, str] = {}
        self.diag = Counter()
        for attribute, artifact in (
            ("model", "ranker_model.json"),
            ("v29_model", "v29_ranker_model.json"),
            ("legacy_model", "legacy_ranker_model.json"),
        ):
            try:
                setattr(self, attribute, _load_model(artifact))
            except Exception as exc:
                self.errors[attribute] = f"{type(exc).__name__}: {exc}"
        try:
            self.memory = _load_memory()
        except Exception as exc:
            self.errors["memory"] = f"{type(exc).__name__}: {exc}"
        try:
            selector = _load_model("selector_model.json")
            selector_enabled = bool(selector.get("enabled"))
            env_enabled = (
                os.environ.get("ALAKAZAM_ML_V33_ENABLE_SELECTOR", "1")
                == "1"
            )
            if selector_enabled and env_enabled:
                artifacts = list(selector.get("base_artifacts") or [])
                bases = [_load_model(str(name)) for name in artifacts]
                if (
                        len(bases) != len(selector.get("model_order") or [])
                        or not bases):
                    raise ValueError("selector base-model count mismatch")
                self.selector_model = selector
                self.selector_bases = bases
        except Exception as exc:
            self.errors["selector"] = f"{type(exc).__name__}: {exc}"

    def reset(self) -> None:
        self.diag.clear()

    def snapshot(self) -> dict[str, Any]:
        decisions = max(1, self.diag["decisions"])
        return {
            **dict(self.diag),
            "runtime_scope": (
                "v33_oof_selector_plus_measured_rebuild_logic"
            ),
            "override_enabled": self.enable_override,
            "memory_enabled": self.enable_memory,
            "model_loaded": self.model is not None,
            "numeric_model_loaded": self.numeric_model is not None,
            "v29_model_loaded": self.v29_model is not None,
            "legacy_model_loaded": self.legacy_model is not None,
            "memory_loaded": bool(self.memory),
            "selector_enabled": self.selector_model is not None,
            "selector_base_models": len(self.selector_bases),
            "errors": dict(self.errors),
            "memory_rate": self.diag["memory_selected"] / decisions,
            "model_rate": self.diag["model_selected"] / decisions,
            "fallback_rate": self.diag["fallback"] / decisions,
            "average_inference_ms": (
                self.diag["inference_us"] / decisions / 1000.0
            ),
        }

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        mean = sum(values) / max(1, len(values))
        variance = sum(
            (value - mean) ** 2 for value in values
        ) / max(1, len(values))
        scale = max(math.sqrt(variance), 1e-5)
        return [(value - mean) / scale for value in values]

    def _selector_scores(
        self,
        features: list[dict[str, Any]],
        representatives: list[int],
    ) -> list[float]:
        selector = self.selector_model
        if selector is None or not self.selector_bases:
            raise RuntimeError("selector unavailable")
        model_names = list(selector["model_order"])
        base_score_sets: list[list[float]] = []
        for model in self.selector_bases:
            rows = [
                [
                    float(features[index].get(name, -1))
                    for name in model["feature_names"]
                ]
                for index in representatives
            ]
            base_score_sets.append(self._normalize([
                _selector_tree_score(row, model) for row in rows
            ]))

        candidate_count = len(representatives)
        ranks: list[list[int]] = []
        tops: list[int] = []
        votes = [0] * candidate_count
        for scores in base_score_sets:
            order = sorted(
                range(candidate_count),
                key=lambda index: (-scores[index], index),
            )
            positions = [0] * candidate_count
            for position, index in enumerate(order):
                positions[index] = position
            ranks.append(positions)
            tops.append(order[0])
            votes[order[0]] += 1
        unique_top_count = len(set(tops))
        selector_rows = []
        raw_names = list(selector.get("raw_feature_names") or [])
        for local_index, representative in enumerate(representatives):
            row: dict[str, float] = {
                f"raw__{name}": float(
                    features[representative].get(name, -1)
                )
                for name in raw_names
            }
            candidate_scores = []
            candidate_ranks = []
            for model_index, name in enumerate(model_names):
                score = base_score_sets[model_index][local_index]
                top_score = max(base_score_sets[model_index])
                rank = ranks[model_index][local_index]
                row.update({
                    f"{name}__score": score,
                    f"{name}__gap": score - top_score,
                    f"{name}__rank": float(rank),
                    f"{name}__selected": float(
                        local_index == tops[model_index]
                    ),
                })
                candidate_scores.append(score)
                candidate_ranks.append(rank)
            score_mean = sum(candidate_scores) / len(candidate_scores)
            score_variance = sum(
                (score - score_mean) ** 2
                for score in candidate_scores
            ) / len(candidate_scores)
            row.update({
                "model_score_mean": score_mean,
                "model_score_std": math.sqrt(score_variance),
                "model_score_min": min(candidate_scores),
                "model_score_max": max(candidate_scores),
                "model_score_range": (
                    max(candidate_scores) - min(candidate_scores)
                ),
                "model_rank_mean": (
                    sum(candidate_ranks) / len(candidate_ranks)
                ),
                "model_rank_min": float(min(candidate_ranks)),
                "model_vote_count": float(votes[local_index]),
                "model_vote_fraction": (
                    votes[local_index] / len(base_score_sets)
                ),
                "model_unique_top_count": float(unique_top_count),
            })
            selector_rows.append([
                float(row.get(name, -1))
                for name in selector["feature_names"]
            ])
        return [
            _selector_tree_score(row, selector)
            for row in selector_rows
        ]

    def _fallback(
        self,
        baseline_action: list[int],
        reason: str,
    ) -> list[int]:
        self.diag["fallback"] += 1
        self.diag[f"fallback_{reason}"] += 1
        return list(baseline_action)

    def _memory_choice(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
    ) -> tuple[list[int] | None, str]:
        if not self.enable_memory or not self.memory:
            return None, ""
        exact_key, canonical_key = teacher_memory_keys(observation)
        for name, key in (
            ("exact", exact_key),
            ("canonical", canonical_key),
        ):
            table_name = (
                "exact" if name == "exact" else "canonical_repeated"
            )
            semantic = (self.memory.get(table_name) or {}).get(key)
            if semantic is None:
                continue
            action = resolve_semantic_action(observation, semantic)
            if action is not None and _legal(action, select):
                return action, name
            self.diag[f"memory_{name}_unresolved"] += 1
        return None, ""

    def recall(
        self,
        observation: dict[str, Any],
    ) -> list[int] | None:
        """Fast path used before the substantially costlier v29 baseline."""
        started = time.perf_counter()
        select = observation.get("select") or {}
        action, kind = self._memory_choice(observation, select)
        if action is None:
            return None
        self.diag["decisions"] += 1
        self.diag["memory_selected"] += 1
        self.diag[f"memory_{kind}_selected"] += 1
        self.diag["inference_us"] += int(
            (time.perf_counter() - started) * 1_000_000
        )
        return action

    def choose(
        self,
        observation: dict[str, Any],
        baseline_action: list[int],
        deterministic_action: list[int] | None = None,
        *,
        memory_checked: bool = False,
    ) -> list[int]:
        self.diag["decisions"] += 1
        started = time.perf_counter()
        select = observation.get("select") or {}
        options = list(select.get("option") or [])

        if not memory_checked:
            memory_action, memory_kind = self._memory_choice(
                observation,
                select,
            )
            if memory_action is not None:
                self.diag["memory_selected"] += 1
                self.diag[f"memory_{memory_kind}_selected"] += 1
                self.diag["inference_us"] += int(
                    (time.perf_counter() - started) * 1_000_000
                )
                return memory_action

        if (
            self.model is None
            or self.v29_model is None
            or self.legacy_model is None
            or not options
        ):
            return self._fallback(baseline_action, "model_unavailable")
        if float(observation.get("remainingOverageTime") or 600) < 2.0:
            return self._fallback(baseline_action, "time_guard")
        if (
            int(select.get("type", -1)) != 0
            or int(select.get("context", -1)) != 0
        ):
            return self._fallback(baseline_action, "outside_training_scope")
        if (
            int(select.get("minCount") or 0) != 1
            or int(select.get("maxCount") or 0) != 1
        ):
            return self._fallback(baseline_action, "multi_select")
        if len(options) < 2:
            return self._fallback(baseline_action, "forced")
        if (
            len(baseline_action) != 1
            or not 0 <= baseline_action[0] < len(options)
        ):
            return self._fallback(baseline_action, "baseline_unresolved")
        deterministic = (
            list(deterministic_action)
            if deterministic_action is not None
            else list(baseline_action)
        )
        if (
            len(deterministic) != 1
            or not 0 <= deterministic[0] < len(options)
        ):
            return self._fallback(baseline_action, "deterministic_unresolved")

        try:
            current = observation.get("current") or {}
            action_map = {
                str(key): int(value)
                for key, value in (
                    self.model.get("action_type_map") or {}
                ).items()
            }
            base_state = state_features(current)
            base_state.update(observation_features(observation))
            features: list[dict[str, Any]] = []
            contexts: list[dict[str, Any]] = []
            for option_position, option in enumerate(options):
                feature = dict(option_features(
                    current,
                    select,
                    option,
                    base_state=base_state,
                    option_position=option_position,
                ))
                action_name = str(feature.get("action_type") or "other")
                feature["_action_name"] = action_name
                feature["action_type"] = action_map.get(action_name, -1)
                features.append(feature)
                contexts.append(_context_from_feature(option, feature))

            baseline_index = baseline_action[0]
            deterministic_index = deterministic[0]
            baseline_key = _feature_semantic_key(features[baseline_index])
            deterministic_key = _feature_semantic_key(
                features[deterministic_index]
            )
            baseline_context = contexts[baseline_index]
            if baseline_context["attack_lethal"]:
                return self._fallback(baseline_action, "lethal_guard")

            policy_scores = _fallback_policy_scores(
                observation,
                len(options),
            )
            policy_order, policy_positions = _rank_positions(policy_scores)
            policy_peak = policy_scores[policy_order[0]]

            legacy_scores = []
            for feature in features:
                row = [
                    float(feature.get(name, -1))
                    for name in self.legacy_model["feature_names"]
                ]
                legacy_scores.append(_tree_score(row, self.legacy_model))
            legacy_order, legacy_positions = _rank_positions(legacy_scores)
            legacy_peak = legacy_scores[legacy_order[0]]
            fallback_action_type = int(
                features[deterministic_index]["action_type"]
            )
            fallback_card_id = int(
                features[deterministic_index].get(
                    "candidate_card_id",
                    -1,
                )
            )
            fallback_legacy_agree = int(
                deterministic_index == legacy_order[0]
            )
            for index, feature in enumerate(features):
                feature.update({
                    "fallback_selected": int(
                        _feature_semantic_key(feature) == deterministic_key
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

            v29_scores = []
            for feature in features:
                row = [
                    float(feature.get(name, -1))
                    for name in self.v29_model["feature_names"]
                ]
                v29_scores.append(_tree_score(row, self.v29_model))
            v29_order, v29_positions = _rank_positions(v29_scores)
            v29_peak = v29_scores[v29_order[0]]
            for index, feature in enumerate(features):
                feature.update({
                    "v29_selected": int(
                        _feature_semantic_key(feature) == baseline_key
                    ),
                    "v29_ranker_score": v29_scores[index],
                    "v29_ranker_score_gap": v29_scores[index] - v29_peak,
                    "v29_ranker_rank": v29_positions[index],
                    "v29_ranker_raw_selected": int(
                        index == v29_order[0]
                    ),
                    "v29_deterministic_agree": int(
                        baseline_key == deterministic_key
                    ),
                })

            representatives = []
            seen = set()
            for index, feature in enumerate(features):
                key = _feature_semantic_key(feature)
                if key in seen:
                    continue
                seen.add(key)
                representatives.append(index)
            rows = [
                [
                    float(features[index].get(name, -1))
                    for name in self.model["feature_names"]
                ]
                for index in representatives
            ]
            if self.selector_model is not None:
                scores = self._selector_scores(
                    features,
                    representatives,
                )
                self.diag["selector_evaluated"] += 1
            else:
                primary_scores = [
                    _tree_score(row, self.model) for row in rows
                ]
                primary_normalized = self._normalize(primary_scores)
                numeric_weight = (
                    float(self.numeric_model.get("ensemble_weight", 0.0))
                    if self.numeric_model is not None
                    else 0.0
                )
                if numeric_weight:
                    numeric_rows = [
                        [
                            float(features[index].get(name, -1))
                            for name in self.numeric_model["feature_names"]
                        ]
                        for index in representatives
                    ]
                    numeric_scores = [
                        _tree_score(row, self.numeric_model)
                        for row in numeric_rows
                    ]
                    numeric_normalized = self._normalize(numeric_scores)
                    scores = [
                        primary + numeric_weight * numeric
                        for primary, numeric in zip(
                            primary_normalized,
                            numeric_normalized,
                        )
                    ]
                else:
                    scores = primary_normalized
            order = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )
            probabilities = _probabilities(
                scores,
                float(self.model.get("temperature", 1.0)),
            )
            local_top = order[0]
            top = representatives[local_top]
            confidence = probabilities[local_top]
            second = probabilities[order[1]] if len(order) > 1 else 0.0
            threshold = max(
                float(self.model.get("fallback_probability", 0.20)),
                self.threshold_override,
            )
            margin_threshold = float(
                self.model.get("fallback_margin", 0.0)
            )
            if (
                confidence < threshold
                or confidence - second < margin_threshold
            ):
                self.diag["low_confidence"] += 1
                return self._fallback(
                    baseline_action,
                    "low_confidence",
                )

            attack_available = any(
                context["action_type"] == "attack"
                for context in contexts
            )
            safety_reason = _candidate_safety_reason(
                contexts[top],
                baseline_context,
                features[top],
                attack_is_available=attack_available,
            )
            if safety_reason is not None:
                self.diag[f"candidate_blocked_{safety_reason}"] += 1
                return self._fallback(
                    baseline_action,
                    f"safety_{safety_reason}",
                )

            if _feature_semantic_key(features[top]) == baseline_key:
                self.diag["model_selected"] += 1
                self.diag["model_agrees_v29"] += 1
                return list(baseline_action)
            action = [top]
            if not _legal(action, select):
                return self._fallback(baseline_action, "legality")
            self.diag["inference_us"] += int(
                (time.perf_counter() - started) * 1_000_000
            )
            self.diag["model_selected"] += 1
            predicted = contexts[top]["action_type"]
            self.diag["model_override"] += 1
            self.diag[f"model_override_{predicted}"] += 1
            if not self.enable_override:
                self.diag["shadow_evaluated"] += 1
                return self._fallback(baseline_action, "shadow_only")
            return action
        except Exception as exc:
            self.diag["runtime_error"] += 1
            self.diag[f"runtime_{type(exc).__name__}"] += 1
            return self._fallback(baseline_action, "runtime_error")
