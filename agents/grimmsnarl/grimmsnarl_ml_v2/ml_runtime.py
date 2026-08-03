"""Runtime side of the Grimmsnarl imitation ranker.

Standard library only. It scores MAIN options with the distilled LightGBM
trees and reproduces the intra-turn history columns exactly as the offline
corpus builder computed them; if the two drifted, the deployed agent would be
reading different features from the ones it was fitted on.

Two things are deliberately absent, because they are what cost the Alakazam
line most of its measured agreement:

* no rule shell over MAIN. The ranker's argmax is returned as-is. A safety
  veto only fires when the ranker cannot produce a legal index at all.
* no residual chain. The model reads the observation, nothing else.
"""

from __future__ import annotations

import json
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
            best_index = representatives[0]
            best_score = None
            for position in representatives:
                row = features[position]
                vector = [
                    float(row.get(name, -1)) for name in self.names
                ]
                if self.teacher_index >= 0:
                    vector[self.teacher_index] = float(self.teacher_code)
                score = tree_score(vector, self.model)
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = position
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

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)
