"""Runtime side of the Grimmsnarl v3 two-stage imitation policy.

Standard library only. It scores MAIN options with the distilled LightGBM
trees and reproduces the intra-turn history columns exactly as the offline
corpus builder computed them; if the two drifted, the deployed agent would be
reading different features from the ones it was fitted on.

Two safety properties from v2 remain:

* no rule shell over MAIN. The ranker's argmax is returned as-is. A safety
  veto only fires when the ranker cannot produce a legal index at all.
The v3 addition is a guarded multiclass next-action prior. It reads only the
same public observation, complete legal menu and frozen v2 scores. The prior
changes action-family ordering; v2 still resolves the concrete option.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import ml_features

SEMANTIC = (
    "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "candidate_inplay_area",
    "candidate_target_hp", "candidate_target_energy",
    "ctx_card_id", "ctx_area", "ctx_owner_is_self", "ctx_number",
)
CLASS = ("action_type_id", "candidate_card_id")
TURN_FEATURES = (
    "turn_decision_index",
    "turn_candidate_offer_count",
    "turn_candidate_passed_over",
    "turn_candidate_offered_previous",
    "turn_candidate_first_offer_index",
    "turn_class_passed_over",
    "turn_class_offer_count",
    "turn_new_candidate",
)
MAIN_CONTEXT = 0


def _resolve(name: str) -> str:
    candidates = []
    if "__file__" in globals():
        candidates.append(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        )
    candidates.append(name)
    candidates.append(os.path.join("/kaggle_simulations/agent", name))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(name)


def tree_score(row: list[float], model: dict[str, Any]) -> float:
    total = 0.0
    for tree in model["trees"]:
        node = tree
        while "v" not in node:
            value = row[node["f"]]
            if value != value:
                go_left = node.get("x", True)
            elif node.get("d", "<=") == "==":
                go_left = int(round(value)) in node["cs"]
            else:
                go_left = value <= node["t"]
            node = node["l"] if go_left else node["r"]
        total += node["v"]
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


def tree_sum(row: list[float], trees: list[dict[str, Any]]) -> float:
    """Score one class of an interleaved LightGBM multiclass model."""
    total = 0.0
    for tree in trees:
        node = tree
        while "v" not in node:
            value = row[node["f"]]
            if value != value:
                go_left = node.get("x", True)
            elif node.get("d", "<=") == "==":
                go_left = int(round(value)) in node["cs"]
            else:
                go_left = value <= node["t"]
            node = node["l"] if go_left else node["r"]
        total += node["v"]
    return total


def _prepare(node: dict[str, Any]) -> None:
    """Turn categorical value lists into sets once, not per scored row."""
    stack = [node]
    while stack:
        current = stack.pop()
        if "v" in current:
            continue
        if current.get("d") == "==":
            current["cs"] = set(current.get("c", ()))
        stack.append(current["l"])
        stack.append(current["r"])


class Ranker:
    def __init__(self, model_path: str = "ranker_model.json"):
        with open(_resolve(model_path), encoding="utf-8") as handle:
            self.model = json.load(handle)
        for tree in self.model["trees"]:
            _prepare(tree)
        self.names: list[str] = list(self.model["feature_names"])
        # Contexts the export measured as worth routing. Without it every
        # context in SCORABLE_CONTEXTS is scored, which shipped context 8 on
        # 9 held-out decisions at 22% Top-1 - a rule replaced by noise.
        routed = self.model.get("routed_contexts")
        self.contexts = (
            ml_features.SCORABLE_CONTEXTS
            if routed is None
            else frozenset(int(c) for c in routed)
        )
        self.teacher_code = self.model.get("teacher_team_code")
        self.teacher_index = (
            self.names.index("teacher_team_id")
            if "teacher_team_id" in self.names else -1
        )
        if (self.teacher_index >= 0) != (self.teacher_code is not None):
            raise ValueError(
                "model has teacher_team_id but no teacher_team_code (or "
                "vice versa); inference would score an unseen pilot"
            )
        self.action_model: dict[str, Any] | None = None
        self.action_load_error: str | None = None
        try:
            with open(_resolve("action_model.json"), encoding="utf-8") as handle:
                action_model = json.load(handle)
            if (
                action_model.get("format")
                != "lightgbm_multiclass_tree_v1"
                or not action_model.get("class_trees")
                or len(action_model.get("classes") or [])
                != len(action_model["class_trees"])
            ):
                raise ValueError("unsupported or incomplete action model")
            for trees in action_model["class_trees"]:
                for tree in trees:
                    _prepare(tree)
            self.action_model = action_model
        except Exception as error:
            # The base v2.1 ranker remains a complete legal policy. A broken
            # optional prior must degrade to it, not take the submission down.
            self.action_load_error = f"{type(error).__name__}: {error}"
        self.reset()

    def reset(self) -> None:
        # Teacher-forced replay scores our answer but must advance the
        # intra-turn history with the *teacher's* action, or the columns stop
        # describing the turn the corpus described. The evaluator does that via
        # observe_external, so commit has to stand down or the history advances
        # twice per decision and every offer/pass count comes out doubled.
        self.teacher_forced = False
        self._pending: list[dict[str, Any]] | None = None
        self._turn_key: tuple | None = None
        self._seen_candidate: dict[tuple, tuple[int, int, int]] = {}
        self._seen_class: dict[tuple, tuple[int, int]] = {}
        self._previous_offered: set[tuple] = set()
        self._position = 0
        self.stats = {
            "main_decisions": 0,
            "non_main_decisions": 0,
            "ranker_used": 0,
            "feature_errors": 0,
            "score_errors": 0,
            "action_prior_used": 0,
            "action_prior_errors": 0,
        }

    @staticmethod
    def is_main(select: dict[str, Any] | None) -> bool:
        if not select:
            return False
        options = select.get("option") or []
        return (
            int(select.get("context", -1)) == MAIN_CONTEXT
            and int(select.get("minCount") or 0) == 1
            and int(select.get("maxCount") or 0) == 1
            and len(options) >= 2
        )

    def is_scorable(self, select: dict[str, Any] | None) -> bool:
        """Every single-pick select this model was measured as fit to decide.

        v1 answered this for MAIN only and left deck search and damage
        placement to the rule policy, which matched the pinned teacher 39.5%
        and 50-65% of the time. Optional selects count: across 3,655 games the
        teachers never declined one, so there is no decline branch.

        The context must also be in the export's routed list. Routing every
        scorable context unconditionally shipped context 8 on 9 held-out
        decisions at 22% Top-1; a context with no data behind it is better
        left to the rule that already handles it.
        """
        if not select:
            return False
        options = select.get("option") or []
        return (
            int(select.get("context", -1)) in self.contexts
            and int(select.get("minCount") or 0) <= 1
            and int(select.get("maxCount") or 0) == 1
            and len(options) >= 2
        )

    @staticmethod
    def is_corpus_scorable(select: dict[str, Any] | None) -> bool:
        """Whether this decision contributed to intra-turn corpus history.

        A routed-context gate may leave a thin context to the rule policy, but
        the corpus builder still observed that decision before later choices
        in the same turn. Runtime must therefore record the rule's choice even
        when it deliberately does not score that context with the ranker.
        """
        if not select:
            return False
        options = select.get("option") or []
        return (
            int(select.get("context", -1))
            in ml_features.SCORABLE_CONTEXTS
            and int(select.get("minCount") or 0) <= 1
            and int(select.get("maxCount") or 0) == 1
            and len(options) >= 2
        )

    def _rows(self, observation: dict[str, Any]) -> tuple[
        list[dict[str, Any]], list[int]
    ]:
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        base = ml_features.state_features(current)
        base.update(ml_features.observation_features(observation))
        action_map = {
            name: index
            for index, name in enumerate(ml_features.ACTION_TYPES)
        }
        features: list[dict[str, Any]] = []
        for position, option in enumerate(options):
            row = dict(ml_features.option_features(
                current, select, option,
                base_state=base, option_position=position,
            ))
            row["action_type_id"] = action_map.get(
                str(row.pop("action_type", "other")), action_map["other"]
            )
            features.append(row)

        # Same collapse rule as the corpus builder: interchangeable copies are
        # one candidate, and the first occurrence represents the group.
        representatives: list[int] = []
        seen: set[tuple] = set()
        for position, row in enumerate(features):
            key = self._semantic(row)
            if key in seen:
                continue
            seen.add(key)
            representatives.append(position)
        return features, representatives

    @staticmethod
    def _semantic(row: dict[str, Any]) -> tuple:
        return tuple(int(row.get(name, -1)) for name in SEMANTIC)

    @staticmethod
    def _class(row: dict[str, Any]) -> tuple:
        return tuple(int(row.get(name, -1)) for name in CLASS)

    def _turn_state(self, observation: dict[str, Any],
                    features: list[dict[str, Any]]) -> None:
        """Write the eight intra-turn columns into every candidate row."""
        current = observation.get("current") or {}
        key = (int(current.get("turn", -1)),)
        if key != self._turn_key:
            self._turn_key = key
            self._seen_candidate = {}
            self._seen_class = {}
            self._previous_offered = set()
            self._position = 0
        for row in features:
            semantic = self._semantic(row)
            klass = self._class(row)
            offers, passed, first = self._seen_candidate.get(
                semantic, (0, 0, -1)
            )
            class_offers, class_passed = self._seen_class.get(klass, (0, 0))
            values = (
                self._position,
                offers,
                passed,
                int(semantic in self._previous_offered),
                first if first >= 0 else self._position,
                class_passed,
                class_offers,
                int(offers == 0),
            )
            for name, value in zip(TURN_FEATURES, values):
                row[name] = value

    def note_decision(self, features: list[dict[str, Any]],
                      chosen: int) -> None:
        """Advance the intra-turn history with the action we just took."""
        chosen_semantic = self._semantic(features[chosen])
        chosen_class = self._class(features[chosen])
        offered_now: dict[tuple, tuple] = {}
        for row in features:
            offered_now.setdefault(self._semantic(row), self._class(row))
        for semantic, klass in offered_now.items():
            offers, passed, first = self._seen_candidate.get(
                semantic, (0, 0, -1)
            )
            self._seen_candidate[semantic] = (
                offers + 1,
                passed + int(semantic != chosen_semantic),
                first if first >= 0 else self._position,
            )
            class_offers, class_passed = self._seen_class.get(klass, (0, 0))
            self._seen_class[klass] = (
                class_offers + 1,
                class_passed + int(klass != chosen_class),
            )
        self._previous_offered = set(offered_now)
        self._position += 1

    @staticmethod
    def _zscore(values: list[float]) -> list[float]:
        mean = sum(values) / max(1, len(values))
        variance = sum(
            (value - mean) ** 2 for value in values
        ) / max(1, len(values))
        scale = max(math.sqrt(variance), 1e-5)
        return [(value - mean) / scale for value in values]

    def _apply_action_prior(
        self,
        features: list[dict[str, Any]],
        representatives: list[int],
        base_scores: list[float],
    ) -> list[float]:
        """Blend the elite next-action prior with v2.1 candidate scores."""
        model = self.action_model
        if model is None or len(representatives) < 2:
            return base_scores
        actions = [
            int(features[position].get("action_type_id", 10))
            for position in representatives
        ]
        counts = [0.0] * 17
        maxima = [-20.0] * 17
        totals = [0.0] * 17
        for action, score in zip(actions, base_scores):
            if 0 <= action < 17:
                counts[action] += 1.0
                totals[action] += score
                maxima[action] = max(maxima[action], score)
        means = [
            totals[action] / counts[action] if counts[action] else -20.0
            for action in range(17)
        ]
        ordered = sorted(base_scores)
        top = ordered[-1]
        margin = top - ordered[-2]
        shifted = [math.exp(score - top) for score in base_scores]
        denominator = max(sum(shifted), 1e-8)
        probabilities = [value / denominator for value in shifted]
        entropy = -sum(
            value * math.log(value + 1e-8) for value in probabilities
        )

        decision = dict(features[representatives[0]])
        for action in range(17):
            decision[f"menu_action_{action}_count"] = counts[action]
            decision[f"menu_action_{action}_max_v2"] = maxima[action]
            decision[f"menu_action_{action}_mean_v2"] = means[action]
        decision.update({
            "v2_top_score": top,
            "v2_margin": margin,
            "v2_entropy": entropy,
            "menu_size": len(representatives),
        })
        row = [
            float(decision.get(name, -1))
            for name in model["feature_names"]
        ]
        logits = {
            int(action): tree_sum(row, trees)
            for action, trees in zip(
                model["classes"], model["class_trees"]
            )
        }
        alpha = float(model.get("blend_alpha", 0.0))
        base_z = self._zscore(base_scores)
        self.stats["action_prior_used"] += 1
        return [
            score + alpha * logits.get(action, -20.0)
            for score, action in zip(base_z, actions)
        ]

    def choose(self, observation: dict[str, Any]) -> int | None:
        """Index into ``select.option``, or None to defer to the rule policy.

        Scoring and history are separate steps. ``commit`` must be called with
        the action that was actually taken, which in live play is this one but
        under teacher-forced evaluation is the teacher's, so the intra-turn
        columns describe the same turn the offline corpus described.
        """
        select = observation.get("select") or {}
        self._pending = None
        if not self.is_scorable(select):
            return None
        self.stats["main_decisions"] += 1
        if not self.is_main(select):
            self.stats["non_main_decisions"] += 1
        try:
            features, representatives = self._rows(observation)
            self._turn_state(observation, features)
        except Exception:
            self.stats["feature_errors"] += 1
            return None
        if len(representatives) < 2:
            # The corpus builder drops these and never advances the intra-turn
            # history for them, so neither can we or the offer/pass counts
            # drift away from the columns the model was fitted on. Every
            # option is interchangeable here anyway.
            return None
        self._pending = features
        try:
            base_scores: list[float] = []
            for position in representatives:
                row = features[position]
                vector = [
                    float(row.get(name, -1)) for name in self.names
                ]
                if self.teacher_index >= 0:
                    vector[self.teacher_index] = float(self.teacher_code)
                base_scores.append(tree_score(vector, self.model))
            try:
                scores = self._apply_action_prior(
                    features, representatives, base_scores
                )
            except Exception:
                self.stats["action_prior_errors"] += 1
                scores = base_scores
            best_slot = max(
                range(len(representatives)), key=lambda slot: scores[slot]
            )
            best_index = representatives[best_slot]
        except Exception:
            self.stats["score_errors"] += 1
            return None
        self.stats["ranker_used"] += 1
        return best_index

    def commit(self, chosen: int) -> None:
        """Advance the intra-turn history with the action actually taken."""
        features = self._pending
        self._pending = None
        if self.teacher_forced:
            return  # observe_external will advance with the teacher's action
        if features and 0 <= chosen < len(features):
            self.note_decision(features, chosen)

    def observe_external(self, observation: dict[str, Any],
                         chosen: int) -> None:
        """Keep the turn history aligned when the rule policy decided."""
        if not self.is_corpus_scorable(observation.get("select")):
            return
        try:
            features, representatives = self._rows(observation)
            if len(representatives) < 2:
                return
            self._turn_state(observation, features)
            if 0 <= chosen < len(features):
                self.note_decision(features, chosen)
        except Exception:
            self.stats["feature_errors"] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "action_prior_loaded": int(self.action_model is not None),
            "action_prior_load_error": self.action_load_error,
        }
