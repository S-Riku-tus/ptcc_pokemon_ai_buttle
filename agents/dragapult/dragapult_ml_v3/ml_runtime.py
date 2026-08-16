"""Standard-library runtime for the Dragapult teacher-conditioned ranker."""

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


def _resolve(name: str) -> str:
    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), name))
    candidates.extend((name, os.path.join("/kaggle_simulations/agent", name)))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(name)


def _prepare(node: dict[str, Any]) -> None:
    stack = [node]
    while stack:
        current = stack.pop()
        if "v" in current:
            continue
        if current.get("d") == "==":
            current["cs"] = set(current.get("c", ()))
        stack.extend((current["l"], current["r"]))


def tree_score(row: list[float], model: dict[str, Any]) -> float:
    total = 0.0
    for tree in model["trees"]:
        node = tree
        while "v" not in node:
            value = row[node["f"]]
            if value != value:
                left = node.get("x", True)
            elif node.get("d", "<=") == "==":
                left = int(round(value)) in node["cs"]
            else:
                left = value <= node["t"]
            node = node["l"] if left else node["r"]
        total += float(node["v"])
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


class Ranker:
    """Score legal options while preserving the training-time turn history."""

    def __init__(self, model_path: str = "ranker_model.json"):
        with open(_resolve(model_path), encoding="utf-8") as handle:
            self.model = json.load(handle)
        for tree in self.model.get("trees", []):
            _prepare(tree)
        if not self.model.get("trees"):
            raise ValueError("ranker model contains no trees")
        self.names = list(self.model["feature_names"])
        routed = self.model.get("routed_contexts")
        self.contexts = (
            ml_features.SCORABLE_CONTEXTS
            if routed is None
            else frozenset(int(value) for value in routed)
        )
        self.teacher_index = (
            self.names.index("teacher_team_id")
            if "teacher_team_id" in self.names else -1
        )
        self.teacher_code = self.model.get("teacher_team_code")
        if (self.teacher_index >= 0) != (self.teacher_code is not None):
            raise ValueError("teacher-conditioned model is missing its pinned teacher code")
        raw_support = self.model.get("runtime_support") or {}
        self.support = {
            key: frozenset(int(value) for value in values)
            for key, values in raw_support.items()
            if isinstance(values, list)
        }
        self.reset()

    def reset(self) -> None:
        self.teacher_forced = False
        self._pending: list[dict[str, Any]] | None = None
        self._turn_key: tuple[int] | None = None
        self._seen_candidate: dict[tuple, tuple[int, int, int]] = {}
        self._seen_class: dict[tuple, tuple[int, int]] = {}
        self._previous_offered: set[tuple] = set()
        self._position = 0
        self.last_scores: dict[int, float] = {}
        self.stats = {
            "decisions_seen": 0,
            "ranker_used": 0,
            "unrouted": 0,
            "optional_fallback": 0,
            "ood_fallback": 0,
            "single_semantic_fallback": 0,
            "feature_errors": 0,
            "score_errors": 0,
        }

    def is_scorable(self, select: dict[str, Any] | None) -> bool:
        if not select:
            return False
        options = select.get("option") or []
        context = int(select.get("context", -1))
        if context not in self.contexts:
            return False
        # Optional selections need an explicit decline candidate.  The v1
        # ranker models only played options, so it must never suppress a legal
        # decline merely because accepted examples were present in the log.
        return (
            int(select.get("minCount") or 0) == 1
            and int(select.get("maxCount") or 0) == 1
            and len(options) >= 2
        )

    @staticmethod
    def is_corpus_scorable(select: dict[str, Any] | None) -> bool:
        if not select:
            return False
        return (
            int(select.get("context", -1)) in ml_features.SCORABLE_CONTEXTS
            and int(select.get("minCount") or 0) <= 1
            and int(select.get("maxCount") or 0) == 1
            and len(select.get("option") or []) >= 2
        )

    @staticmethod
    def _semantic(row: dict[str, Any]) -> tuple:
        return tuple(int(row.get(name, -1)) for name in SEMANTIC)

    @staticmethod
    def _class(row: dict[str, Any]) -> tuple:
        return tuple(int(row.get(name, -1)) for name in CLASS)

    def _rows(self, observation: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        base = ml_features.state_features(current)
        base.update(ml_features.observation_features(observation))
        action_map = {name: index for index, name in enumerate(ml_features.ACTION_TYPES)}
        features: list[dict[str, Any]] = []
        for position, option in enumerate(select.get("option") or []):
            row = dict(ml_features.option_features(
                current, select, option, base_state=base, option_position=position,
            ))
            row["action_type_id"] = action_map.get(
                str(row.pop("action_type", "other")), action_map["other"]
            )
            features.append(row)
        representatives: list[int] = []
        seen: set[tuple] = set()
        for position, row in enumerate(features):
            key = self._semantic(row)
            if key not in seen:
                seen.add(key)
                representatives.append(position)
        return features, representatives

    def _supported(self, select: dict[str, Any], rows: list[dict[str, Any]], reps: list[int]) -> bool:
        checks = (
            ("select_context", int(select.get("context", -1))),
        )
        for key, value in checks:
            allowed = self.support.get(key)
            if allowed is not None and value not in allowed:
                return False
        for position in reps:
            row = rows[position]
            for key in ("option_type", "candidate_card_id", "candidate_attack_id"):
                value = int(row.get(key, -1))
                allowed = self.support.get(key)
                # -1 means the option has no member of that category.  A real
                # unseen candidate identity is out-of-distribution and falls
                # back for the whole comparison.
                if allowed is not None and value >= 0 and value not in allowed:
                    return False
        return True

    def _turn_state(self, observation: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        current = observation.get("current") or {}
        key = (int(current.get("turn", -1)),)
        if key != self._turn_key:
            self._turn_key = key
            self._seen_candidate = {}
            self._seen_class = {}
            self._previous_offered = set()
            self._position = 0
        for row in rows:
            semantic = self._semantic(row)
            klass = self._class(row)
            offers, passed, first = self._seen_candidate.get(semantic, (0, 0, -1))
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

    def note_decision(self, rows: list[dict[str, Any]], chosen: int) -> None:
        chosen_semantic = self._semantic(rows[chosen])
        chosen_class = self._class(rows[chosen])
        offered: dict[tuple, tuple] = {}
        for row in rows:
            offered.setdefault(self._semantic(row), self._class(row))
        for semantic, klass in offered.items():
            offers, passed, first = self._seen_candidate.get(semantic, (0, 0, -1))
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
        self._previous_offered = set(offered)
        self._position += 1

    def choose(self, observation: dict[str, Any]) -> int | None:
        select = observation.get("select") or {}
        self.stats["decisions_seen"] += 1
        self._pending = None
        self.last_scores = {}
        if int(select.get("minCount") or 0) == 0:
            self.stats["optional_fallback"] += 1
        if not self.is_scorable(select):
            self.stats["unrouted"] += 1
            return None
        try:
            rows, representatives = self._rows(observation)
            self._turn_state(observation, rows)
        except Exception:
            self.stats["feature_errors"] += 1
            return None
        if len(representatives) < 2:
            self.stats["single_semantic_fallback"] += 1
            return None
        if not self._supported(select, rows, representatives):
            self.stats["ood_fallback"] += 1
            return None
        self._pending = rows
        try:
            best = representatives[0]
            best_score: float | None = None
            for position in representatives:
                vector = [float(rows[position].get(name, -1)) for name in self.names]
                if self.teacher_index >= 0:
                    vector[self.teacher_index] = float(self.teacher_code)
                score = tree_score(vector, self.model)
                self.last_scores[position] = score
                if best_score is None or score > best_score:
                    best, best_score = position, score
        except Exception:
            self.stats["score_errors"] += 1
            self.last_scores = {}
            return None
        self.stats["ranker_used"] += 1
        return best

    def commit(self, chosen: int) -> None:
        rows = self._pending
        self._pending = None
        if self.teacher_forced:
            return
        if rows and 0 <= chosen < len(rows):
            self.note_decision(rows, chosen)

    def observe_external(self, observation: dict[str, Any], chosen: int) -> None:
        if not self.is_corpus_scorable(observation.get("select")):
            return
        try:
            rows, representatives = self._rows(observation)
            if len(representatives) < 2:
                return
            self._turn_state(observation, rows)
            if 0 <= chosen < len(rows):
                self.note_decision(rows, chosen)
        except Exception:
            self.stats["feature_errors"] += 1

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)

