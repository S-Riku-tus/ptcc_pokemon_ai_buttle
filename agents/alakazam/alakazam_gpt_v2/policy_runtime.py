from __future__ import annotations

import json
import math
import os
import time
import zlib
from collections import Counter, deque
from typing import Any

from policy_features import (
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


class HybridRanker:
    """v31 safety/memory shell plus the v32 Yushin ranker."""

    def __init__(
        self,
        attacks: dict[int, dict[str, Any]] | None = None,
        threshold: float = 0.20,
    ):
        del attacks
        self.threshold_override = float(threshold)
        self.enable_override = (
            os.environ.get(
                "ALAKAZAM_ML_V2_ENABLE_OVERRIDE",
                os.environ.get(
                    "ALAKAZAM_ML_V1_ENABLE_OVERRIDE",
                    os.environ.get(
                        "ALAKAZAM_ML_V32_ENABLE_OVERRIDE",
                        os.environ.get("ALAKAZAM_ML_V31_ENABLE_OVERRIDE", "1"),
                    ),
                ),
            )
            == "1"
        )
        self.enable_memory = (
            os.environ.get(
                "ALAKAZAM_ML_V2_ENABLE_MEMORY",
                os.environ.get(
                    "ALAKAZAM_ML_V1_ENABLE_MEMORY",
                    os.environ.get(
                        "ALAKAZAM_ML_V32_ENABLE_MEMORY",
                        os.environ.get("ALAKAZAM_ML_V31_ENABLE_MEMORY", "1"),
                    ),
                ),
            )
            == "1"
        )
        self.model: dict[str, Any] | None = None
        self.type_model: dict[str, Any] | None = None
        self.type_spec: dict[str, Any] | None = None
        self.numeric_model: dict[str, Any] | None = None
        self.v29_model: dict[str, Any] | None = None
        self.legacy_model: dict[str, Any] | None = None
        self.memory: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.diag = Counter()
        self._reset_sequence_state()
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
            with open(_artifact_path("type_model.json"), encoding="utf-8") as handle:
                self.type_model = json.load(handle)
            with open(_artifact_path("type_runtime_spec.json"), encoding="utf-8") as handle:
                self.type_spec = json.load(handle)
        except Exception as exc:
            self.errors["type_model"] = f"{type(exc).__name__}: {exc}"

    def _reset_sequence_state(self) -> None:
        self.sequence_history = deque(maxlen=4)
        self.sequence_counts = Counter()
        self.sequence_decision_index = 0
        self.sequence_turn = None
        self.sequence_same_turn_index = 0
        self.sequence_last_turn_by_kind = {
            "attack": -999,
            "energy": -999,
            "evolve": -999,
            "trainer": -999,
        }

    def reset(self) -> None:
        self.diag.clear()
        self._reset_sequence_state()

    def _ensure_sequence_turn(self, current: dict[str, Any]) -> int:
        turn = int(current.get("turn", 0))
        if self.sequence_turn is not None and turn < self.sequence_turn:
            self._reset_sequence_state()
        if turn != self.sequence_turn:
            self.sequence_turn = turn
            self.sequence_same_turn_index = 0
        return turn

    def _sequence_features(self, current: dict[str, Any]) -> dict[str, int]:
        turn = self._ensure_sequence_turn(current)
        history = list(self.sequence_history)
        out: dict[str, int] = {}
        for offset in range(1, 5):
            item = history[-offset] if offset <= len(history) else (-1, -1, -1, -1)
            out[f"seq_prev_{offset}_action_type"] = int(item[0])
            out[f"seq_prev_{offset}_card_id"] = int(item[1])
            out[f"seq_prev_{offset}_attack_id"] = int(item[2])
            out[f"seq_prev_{offset}_target_id"] = int(item[3])
        for name in (
            "ability", "attack", "bench", "boss", "end", "energy",
            "evolve", "hammer", "other", "retreat", "trainer", "xerosic",
        ):
            out[f"seq_count_{name}"] = int(self.sequence_counts[name])
        out["seq_decision_index"] = int(self.sequence_decision_index)
        out["seq_same_turn_decision_index"] = int(self.sequence_same_turn_index)
        for name in ("attack", "energy", "evolve", "trainer"):
            previous = int(self.sequence_last_turn_by_kind[name])
            out[f"seq_last_{name}_turn_gap"] = (
                min(99, max(0, turn - previous)) if previous >= 0 else 99
            )
        return out

    def record_choice(
        self, observation: dict[str, Any], action: list[int]
    ) -> None:
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        if (
            int(select.get("type", -1)) != 0
            or int(select.get("context", -1)) != 0
            or int(select.get("minCount") or 0) != 1
            or int(select.get("maxCount") or 0) != 1
            or len(options) < 2
            or len(action) != 1
            or not 0 <= action[0] < len(options)
        ):
            return
        current = observation.get("current") or {}
        turn = self._ensure_sequence_turn(current)
        option = options[action[0]]
        feature = dict(option_features(
            current,
            select,
            option,
            base_state=state_features(current),
            observation=observation,
            option_position=action[0],
        ))
        action_name = str(feature.get("action_type") or "other")
        action_map = {
            str(key): int(value)
            for key, value in ((self.model or {}).get("action_type_map") or {}).items()
        }
        action_type_id = action_map.get(action_name, -1)
        chosen = (
            action_type_id,
            int(feature.get("candidate_card_id", -1)),
            int(feature.get("candidate_attack_id", -1)),
            int(feature.get("candidate_target_id", -1)),
        )
        self.sequence_history.append(chosen)
        self.sequence_counts[action_name] += 1
        if action_name in self.sequence_last_turn_by_kind:
            self.sequence_last_turn_by_kind[action_name] = turn
        if action_name in {"boss", "xerosic", "hammer"}:
            self.sequence_last_turn_by_kind["trainer"] = turn
        self.sequence_decision_index += 1
        self.sequence_same_turn_index += 1


    @staticmethod
    def _type_softmax(values: list[float]) -> list[float]:
        peak = max(values)
        exp_values = [math.exp(max(-50.0, min(50.0, value - peak))) for value in values]
        total = sum(exp_values) or 1.0
        return [value / total for value in exp_values]

    def _type_meta_row(
        self,
        features: list[dict[str, Any]],
        representatives: list[int],
        primary_scores: list[float],
    ) -> list[float]:
        spec = self.type_spec or {}
        state_names = list(spec.get("state_names") or [])
        candidate_fields = list(spec.get("candidate_fields") or [])
        first = features[representatives[0]]
        row = [float(first.get(name, -1)) for name in state_names]
        type_peaks: list[float] = []
        for action_type in range(12):
            local = [
                pos for pos, index in enumerate(representatives)
                if int(features[index].get("action_type_id", -1)) == action_type
            ]
            if local:
                values = [primary_scores[pos] for pos in local]
                ranked = sorted(local, key=lambda pos: primary_scores[pos], reverse=True)
                top = ranked[0]
                top_value = primary_scores[top]
                second = primary_scores[ranked[1]] if len(ranked) > 1 else top_value - 5.0
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                values_out = [
                    float(len(local)), top_value, second, top_value - second,
                    mean, variance ** 0.5, float(top),
                    float(sum(value > top_value - 0.25 for value in values)),
                ]
                top_feature = features[representatives[top]]
                values_out.extend(float(top_feature.get(name, -1)) for name in candidate_fields)
                type_peaks.append(top_value)
            else:
                values_out = [0.0, -99.0, -99.0, 0.0, -99.0, 0.0, -1.0, 0.0]
                values_out.extend([-1.0] * len(candidate_fields))
                type_peaks.append(-99.0)
            row.extend(values_out)
        ordered = sorted(range(12), key=lambda index: type_peaks[index], reverse=True)
        probabilities = self._type_softmax(type_peaks)
        row.extend([
            float(ordered[0]), float(ordered[1]), type_peaks[ordered[0]],
            type_peaks[ordered[1]], type_peaks[ordered[0]] - type_peaks[ordered[1]],
            -sum(value * math.log(value + 1e-9) for value in probabilities),
            float(sum(value > -90.0 for value in type_peaks)),
            float(features[representatives[max(range(len(primary_scores)), key=lambda i: primary_scores[i])]].get("action_type_id", -1)),
        ])
        return row

    def _predict_action_type(self, row: list[float]) -> tuple[int, float] | None:
        model = self.type_model
        if not model:
            return None
        selected = [row[int(index)] for index in model.get("cols", [])]
        scores = [float(value) for value in model.get("base", [])]
        for tree in model.get("trees", []):
            node = 0
            left = tree["l"]
            while left[node] != -1:
                feature_value = selected[int(tree["f"][node])]
                if math.isnan(feature_value):
                    go_left = bool(tree["d"][node])
                else:
                    go_left = feature_value < float(tree["v"][node])
                node = left[node] if go_left else tree["r"][node]
            scores[int(tree["c"])] += float(tree["w"][node])
        probabilities = self._type_softmax(scores)
        best = max(range(len(scores)), key=lambda index: scores[index])
        classes = model.get("classes") or list(range(len(scores)))
        return int(classes[best]), float(probabilities[best])

    def snapshot(self) -> dict[str, Any]:
        decisions = max(1, self.diag["decisions"])
        return {
            **dict(self.diag),
            "runtime_scope": "v1_v31_safety_memory_plus_sequence_state_ranker",
            "override_enabled": self.enable_override,
            "memory_enabled": self.enable_memory,
            "model_loaded": self.model is not None,
            "numeric_model_loaded": self.numeric_model is not None,
            "v29_model_loaded": self.v29_model is not None,
            "legacy_model_loaded": self.legacy_model is not None,
            "memory_loaded": bool(self.memory),
            "errors": dict(self.errors),
            "memory_rate": self.diag["memory_selected"] / decisions,
            "model_rate": self.diag["model_selected"] / decisions,
            "fallback_rate": self.diag["fallback"] / decisions,
            "average_inference_ms": (
                self.diag["inference_us"] / decisions / 1000.0
            ),
        }

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
            base_state.update(self._sequence_features(current))
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
                feature["action_type_id"] = action_map.get(action_name, -1)
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
            primary_scores = [
                _tree_score(row, self.model) for row in rows
            ]
            predicted_type = None
            predicted_type_confidence = 0.0
            if self.type_model is not None and self.type_spec is not None:
                type_prediction = self._predict_action_type(
                    self._type_meta_row(features, representatives, primary_scores)
                )
                if type_prediction is not None:
                    predicted_type, predicted_type_confidence = type_prediction
                    self.diag["type_predictions"] += 1
                    self.diag[f"type_predicted_{predicted_type}"] += 1

            def normalized(values):
                mean = sum(values) / max(1, len(values))
                variance = sum(
                    (value - mean) ** 2 for value in values
                ) / max(1, len(values))
                scale = max(variance ** 0.5, 1e-5)
                return [(value - mean) / scale for value in values]

            primary_normalized = normalized(primary_scores)
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
                numeric_normalized = normalized(numeric_scores)
                scores = [
                    primary + numeric_weight * numeric
                    for primary, numeric in zip(
                        primary_normalized,
                        numeric_normalized,
                    )
                ]
            else:
                scores = primary_normalized
            type_threshold = float((self.type_model or {}).get("threshold", 0.5))
            eligible = list(range(len(scores)))
            if predicted_type is not None and predicted_type_confidence >= type_threshold:
                typed = [
                    local for local, index in enumerate(representatives)
                    if int(features[index].get("action_type_id", -1)) == predicted_type
                ]
                if typed:
                    eligible = typed
                    self.diag["type_gate_applied"] += 1
            order = sorted(
                eligible,
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
