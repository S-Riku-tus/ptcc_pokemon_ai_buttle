from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Any

from ml_features import (
    candidate_card,
    candidate_target,
    option_features,
)

# Strategic and irreversible choices remain with the deterministic fallback.
# The distilled model is evaluated in shadow mode by default because every
# tested live-override scope failed the 200-game promotion gate.
RULE_ONLY_ACTIONS = {
    "ability",
    "end",
    "trainer",
    "energy",
    "boss",
    "retreat",
    "xerosic",
    "hammer",
    "other",
}

# In guarded mode ML is only a local ranker for low-risk board construction.
ML_ALLOWED_ACTIONS = {"bench", "evolve", "attack"}
ML_SAFE_BENCH_IDS = {305, 741}  # Dunsparce, Abra
FEZANDIPITI_EX = 140
SHAYMIN = 343
GENESECT = 142


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


def _option_context(
    current: dict[str, Any],
    select: dict[str, Any],
    options: list[dict[str, Any]],
    action: list[int],
) -> dict[str, Any] | None:
    """Return the semantic context for a single selected option."""
    if len(action) != 1 or not (0 <= action[0] < len(options)):
        return None
    option = options[action[0]]
    feature = option_features(current, select, option)
    card = candidate_card(current, option) or {}
    target = candidate_target(current, option) or {}
    return {
        "index": action[0],
        "option": option,
        "action_type": str(feature.get("action_type") or "other"),
        "card_id": int(card.get("id", -1)),
        "target_id": int(target.get("id", -1)),
        "breaks_current_ko": bool(feature.get("breaks_current_ko_estimate", 0)),
        "attack_lethal": bool(feature.get("attack_lethal_estimate", 0)),
    }


def _fallback_scope_reason(context: dict[str, Any] | None) -> str | None:
    """Return why the whole decision must stay with fallback_v12."""
    if context is None:
        return "fallback_unresolved"
    action = context["action_type"]
    if action in RULE_ONLY_ACTIONS:
        return f"rule_only_{action}"
    return None


def _candidate_scope_reason(
    context: dict[str, Any],
    fallback_context: dict[str, Any],
) -> str | None:
    """Validate a model candidate after fallback has established a safe scope."""
    action = context["action_type"]
    if action not in ML_ALLOWED_ACTIONS:
        return f"candidate_rule_only_{action}"

    # Never spend a fallback-reserved attack on development.
    if fallback_context["action_type"] == "attack" and action != "attack":
        return "preserve_fallback_attack"

    # Role Pokémon depend on KO timing, ACE state, or opposing self-KO Abilities that the
    # current model does not encode.
    if action == "bench" and context["card_id"] not in ML_SAFE_BENCH_IDS:
        if context["card_id"] == FEZANDIPITI_EX:
            return "role_fezandipiti"
        if context["card_id"] == SHAYMIN:
            return "role_shaymin"
        if context["card_id"] == GENESECT:
            return "role_genesect"
        return "bench_not_allowlisted"

    # ML may rank alternatives inside the fallback's strategic intent, but may not replace the
    # first Abra body with Dunsparce or an Alakazam-line evolution with Dudunsparce. This keeps the
    # deterministic route/engine priority while still allowing target choice within the same card.
    if fallback_context["action_type"] == "bench" and action == "bench":
        if context["card_id"] != fallback_context["card_id"]:
            return "preserve_fallback_bench_role"
    if fallback_context["action_type"] == "evolve" and action == "evolve":
        if context["card_id"] != fallback_context["card_id"]:
            return "preserve_fallback_evolution_stage"

    if context["breaks_current_ko"]:
        return "breaks_current_ko"
    return None


class HybridRanker:
    """Guarded legal-option ranker layered over the deterministic fallback.

    ML scores low-risk candidate scopes and records counterfactual disagreements,
    but does not replace the fallback unless ``ALAKAZAM_ML_ENABLE_OVERRIDE=1`` is
    explicitly set for an experiment.  This keeps a trained model and useful
    runtime diagnostics without silently promoting a model that lost the live
    battle ablation.
    """

    def __init__(self, attacks: dict[int, dict[str, Any]] | None = None, threshold: float = 0.55):
        del attacks
        self.threshold_override = float(threshold)
        self.enable_override = os.environ.get("ALAKAZAM_ML_ENABLE_OVERRIDE", "0") == "1"
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
            "runtime_scope": "shadow_guarded_v3_base",
            "override_enabled": self.enable_override,
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
        # The model is trained only on ACTIVE MAIN decisions with one required
        # option. Nested search/target selections remain deterministic.
        if int(select.get("type", -1)) != 0 or int(select.get("context", -1)) != 0:
            return self._fallback(fallback_action, "outside_training_scope")
        if int(select.get("minCount") or 0) != 1 or int(select.get("maxCount") or 0) != 1:
            return self._fallback(fallback_action, "multi_select")
        if len(options) < 2:
            return self._fallback(fallback_action, "forced")

        try:
            current = observation.get("current") or {}
            fallback_context = _option_context(current, select, options, fallback_action)
            scope_reason = _fallback_scope_reason(fallback_context)
            if scope_reason is not None:
                return self._fallback(fallback_action, scope_reason)

            # A proven fallback immediate KO must never be spent for extra setup.
            if fallback_context and fallback_context["attack_lethal"]:
                return self._fallback(fallback_action, "lethal_guard")

            action_map = {str(k): int(v) for k, v in (self.model.get("action_type_map") or {}).items()}
            rows: list[list[float]] = []
            contexts: list[dict[str, Any]] = []
            safe: list[bool] = []
            for option in options:
                feature = option_features(current, select, option)
                action = str(feature.get("action_type") or "other")
                feature["action_type"] = action_map.get(action, -1)
                rows.append([float(feature.get(name, -1)) for name in self.model["feature_names"]])
                card = candidate_card(current, option) or {}
                target = candidate_target(current, option) or {}
                context = {
                    "option": option,
                    "action_type": action,
                    "card_id": int(card.get("id", -1)),
                    "target_id": int(target.get("id", -1)),
                    "breaks_current_ko": bool(feature.get("breaks_current_ko_estimate", 0)),
                    "attack_lethal": bool(feature.get("attack_lethal_estimate", 0)),
                }
                contexts.append(context)
                reason = _candidate_scope_reason(context, fallback_context)
                safe.append(reason is None)
                if reason is not None:
                    self.diag[f"candidate_blocked_{reason}"] += 1

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
            predicted_action = contexts[top]["action_type"]

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
            if not self.enable_override:
                self.diag["shadow_evaluated"] += 1
                if action != list(fallback_action):
                    self.diag["shadow_disagreement"] += 1
                    self.diag[f"shadow_disagreement_{predicted_action}"] += 1
                    if fallback_context and predicted_action != fallback_context["action_type"]:
                        self.diag["shadow_disagreement_action_type"] += 1
                return self._fallback(fallback_action, "shadow_only")
            if action != list(fallback_action):
                self.diag["model_override"] += 1
                self.diag[f"model_override_{predicted_action}"] += 1
                if fallback_context and predicted_action != fallback_context["action_type"]:
                    self.diag["model_override_action_type"] += 1
                    self.diag[
                        f"model_override_{fallback_context['action_type']}_to_{predicted_action}"
                    ] += 1
            return action
        except Exception as exc:
            self.diag["runtime_error"] += 1
            self.diag[f"runtime_{type(exc).__name__}"] += 1
            return self._fallback(fallback_action, "exception")
