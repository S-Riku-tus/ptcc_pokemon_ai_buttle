from __future__ import annotations

import json
import os
import time
import zlib
from collections import Counter
from typing import Any

from ml_features import (
    action_type as _option_action_type,
    candidate_card,
    candidate_target,
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


def _flag(name: str, default: str) -> str:
    """v33 setting with v32 and v31 names accepted for compatibility."""
    return os.environ.get(
        f"ALAKAZAM_ML_V33_{name}",
        os.environ.get(
            f"ALAKAZAM_ML_V32_{name}",
            os.environ.get(f"ALAKAZAM_ML_V31_{name}", default),
        ),
    )


TURN_FEATURE_NAMES = (
    "turn_decision_index",
    "turn_candidate_offer_count",
    "turn_candidate_passed_over",
    "turn_candidate_offered_previous",
    "turn_candidate_first_offer_index",
    "turn_class_passed_over",
    "turn_class_offer_count",
    "turn_new_candidate",
)


def _turn_option_keys(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
    action_map: dict[str, int],
) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Semantic and coarse class keys used by the intra-turn history.

    Derived straight from the raw option so the tracker stays cheap on the
    memory fast path, and identical to the columns the corpus builder writes.
    """
    card = candidate_card(current, option, select) or {}
    target = candidate_target(current, option) or {}
    card_id = int(card.get("id", -1))
    target_id = int(target.get("id", -1))
    semantic = (
        int(option.get("type", -1)),
        card_id,
        int(option.get("attackId", -1)),
        target_id,
        int(option.get("inPlayArea", -1)),
    )
    action = str(_option_action_type(current, option, select) or "other")
    return semantic, (action_map.get(action, -1), card_id)


class _TurnHistory:
    """What the acting player has been offered and passed over this turn.

    The v32 ranker scores each candidate as a pure function of the current
    observation, so it cannot tell a card it has just declined twice from one
    it is seeing for the first time. This tracker restores that context from
    the agent's own decision stream.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.turn_key: tuple[int, int] | None = None
        self.candidates: dict[tuple[int, ...], list[int]] = {}
        self.classes: dict[tuple[int, int], list[int]] = {}
        self.previous_offered: set[tuple[int, ...]] = set()
        self.position = 0

    def _sync(self, current: dict[str, Any]) -> None:
        key = (
            int(current.get("turn", -1)),
            int(current.get("yourIndex", -1)),
        )
        if key != self.turn_key:
            self.reset()
            self.turn_key = key

    def columns(
        self,
        current: dict[str, Any],
        keys: list[tuple[tuple[int, ...], tuple[int, int]]],
    ) -> list[dict[str, int]]:
        self._sync(current)
        rows = []
        for semantic, class_key in keys:
            offers, passed, first = self.candidates.get(semantic, (0, 0, -1))
            class_offers, class_passed = self.classes.get(class_key, (0, 0))
            rows.append({
                "turn_decision_index": self.position,
                "turn_candidate_offer_count": offers,
                "turn_candidate_passed_over": passed,
                "turn_candidate_offered_previous": int(
                    semantic in self.previous_offered
                ),
                "turn_candidate_first_offer_index": (
                    first if first >= 0 else self.position
                ),
                "turn_class_passed_over": class_passed,
                "turn_class_offer_count": class_offers,
                "turn_new_candidate": int(offers == 0),
            })
        return rows

    def record(
        self,
        current: dict[str, Any],
        keys: list[tuple[tuple[int, ...], tuple[int, int]]],
        chosen: int,
    ) -> None:
        self._sync(current)
        if not 0 <= chosen < len(keys):
            return
        chosen_semantic, chosen_class = keys[chosen]
        # One increment per distinct semantic candidate, matching the corpus
        # builder. Raw option lists repeat interchangeable copies and would
        # otherwise inflate every counter.
        offered_now: dict[tuple[int, ...], tuple[int, int]] = {}
        for semantic, class_key in keys:
            offered_now.setdefault(semantic, class_key)
        for semantic, class_key in offered_now.items():
            offers, passed, first = self.candidates.get(semantic, (0, 0, -1))
            self.candidates[semantic] = (
                offers + 1,
                passed + int(semantic != chosen_semantic),
                first if first >= 0 else self.position,
            )
            class_offers, class_passed = self.classes.get(class_key, (0, 0))
            self.classes[class_key] = (
                class_offers + 1,
                class_passed + int(class_key != chosen_class),
            )
        self.previous_offered = set(offered_now)
        self.position += 1


class HybridRanker:
    """v31 safety/memory shell plus the v33 turn-order Yushin ranker."""

    def __init__(
        self,
        attacks: dict[int, dict[str, Any]] | None = None,
        threshold: float = 0.20,
    ):
        del attacks
        self.threshold_override = float(threshold)
        self.enable_override = (
            _flag("ENABLE_OVERRIDE", "1") == "1"
        )
        self.enable_memory = (
            _flag("ENABLE_MEMORY", "1") == "1"
        )
        self.model: dict[str, Any] | None = None
        # v31 shipped a separate numeric-ID ranker. v32 rejected it and v33
        # blends through ``self.ensemble`` instead, so this stays None and is
        # only reported so the inherited diagnostics keep their shape.
        self.numeric_model: dict[str, Any] | None = None
        self.v29_model: dict[str, Any] | None = None
        self.legacy_model: dict[str, Any] | None = None
        self.memory: dict[str, Any] = {}
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

        # v33 may deploy a small validation-selected blend. Every member is a
        # plain JSON tree, so the submission runtime stays standard library
        # only. Extra members are optional and silently skipped when absent.
        self.ensemble: list[tuple[dict[str, Any], float]] = []
        if self.model is not None:
            self.ensemble.append((
                self.model,
                float(self.model.get("ensemble_weight", 1.0)),
            ))
        for index in (1, 2):
            name = f"ranker_model_{index}.json"
            try:
                extra = _load_model(name)
            except FileNotFoundError:
                continue
            except Exception as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            self.ensemble.append((
                extra, float(extra.get("ensemble_weight", 1.0))
            ))
        self.uses_turn_features = any(
            bool(member.get("uses_turn_features"))
            for member, _ in self.ensemble
        )
        self.turn_history = _TurnHistory()

        try:
            self.memory = _load_memory()
        except Exception as exc:
            self.errors["memory"] = f"{type(exc).__name__}: {exc}"

    def reset(self) -> None:
        self.diag.clear()
        self.turn_history.reset()

    def snapshot(self) -> dict[str, Any]:
        decisions = max(1, self.diag["decisions"])
        return {
            **dict(self.diag),
            "runtime_scope": "v33_v31_safety_memory_plus_turn_order_ranker",
            "override_enabled": self.enable_override,
            "memory_enabled": self.enable_memory,
            "model_loaded": self.model is not None,
            "ensemble_size": len(self.ensemble),
            "ensemble_weights": [weight for _, weight in self.ensemble],
            "uses_turn_features": self.uses_turn_features,
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

    def note_decision(
        self,
        observation: dict[str, Any],
        action: list[int] | None,
    ) -> None:
        """Record the action actually returned for a MAIN single-choice.

        Called for every scoped decision, including the ones answered from
        teacher memory or by a fallback, so the intra-turn history matches
        what the corpus builder saw.
        """
        if not self.uses_turn_features or not action or len(action) != 1:
            return
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        if (
            int(select.get("type", -1)) != 0
            or int(select.get("context", -1)) != 0
            or int(select.get("minCount") or 0) != 1
            or int(select.get("maxCount") or 0) != 1
            or len(options) < 2
            or not 0 <= action[0] < len(options)
        ):
            return
        current = observation.get("current") or {}
        try:
            keys = self._turn_keys(current, select, options)
            self.turn_history.record(current, keys, action[0])
        except Exception as exc:
            self.diag["turn_history_error"] += 1
            self.diag[f"turn_history_{type(exc).__name__}"] += 1

    def _turn_keys(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
    ) -> list[tuple[tuple[int, ...], tuple[int, int]]]:
        action_map = {
            str(key): int(value)
            for key, value in (
                (self.model or {}).get("action_type_map") or {}
            ).items()
        }
        return [
            _turn_option_keys(current, select, option, action_map)
            for option in options
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
            or not self.ensemble
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

            if self.uses_turn_features:
                keys = self._turn_keys(current, select, options)
                for feature, columns in zip(
                    features, self.turn_history.columns(current, keys)
                ):
                    feature.update(columns)

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

            def normalized(values):
                mean = sum(values) / max(1, len(values))
                variance = sum(
                    (value - mean) ** 2 for value in values
                ) / max(1, len(values))
                scale = max(variance ** 0.5, 1e-5)
                return [(value - mean) / scale for value in values]

            # Each ensemble member is standardised inside the candidate set
            # before weighting, exactly as the blend was selected offline.
            scores = [0.0] * len(representatives)
            for member, weight in self.ensemble:
                if not weight:
                    continue
                member_scores = [
                    _tree_score(
                        [
                            float(features[index].get(name, -1))
                            for name in member["feature_names"]
                        ],
                        member,
                    )
                    for index in representatives
                ]
                for position, value in enumerate(normalized(member_scores)):
                    scores[position] += weight * value
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
