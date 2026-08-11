"""Standard-library runtime for v20's attack-continuity ranker.

The single teacher-unconditioned model uses 842 public-observation features.
Twenty v20 columns expose the current and backup Grimmsnarl-line ETA and how a
candidate changes that chain.  Training upweights observable hard states, not
eventual wins, so neither replay outcome nor matchup identity is available to
the fitted policy at training-weight selection or at inference.
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

# ----- conditional teacher escalation ---------------------------------------
#
# ``teacher_team_id`` is fed to the trees as a dense code, not as the Kaggle
# team id: LightGBM allocates categorical bins over the raw value range, so the
# corpus builder maps the 21 same-deck pilots onto 0..20 in ascending team-id
# order. The deployed pin is code 16 (team 16494330) and code 0 is team
# 16371703, the 1220.2-rated pilot. Both are asserted against the corpus in
# ``tests/test_v6_teacher_escalation.py``; if the corpus is ever rebuilt with a
# different team set these codes move and that test is what catches it.
ESCALATION_TEACHER_TEAM = 16371703
ESCALATION_TEACHER_CODE = 0
# A decision class is a context plus one column that identifies the option the
# class is about. ``froslass_evolve`` is the only class v6 enables.
#
# ``petrel_stamp`` is defined and measured but *not* enabled. It is the gap
# with the strongest evidence behind it - taking an Unfair Stamp that cannot be
# played this turn runs with pilot rating at Spearman rho -0.626, p = 0.0029
# over the 21 pilots, where the Froslass rate only reaches -0.355, p = 0.116 -
# and the deployed pin is the worst pilot in the corpus on it, 71.3% against a
# weighted field 56.0%. It is held back so one ladder run measures one change;
# see experiments/grimmsnarl_ml_v6/RESULTS.md.
ESCALATION_CLASSES = (
    {"name": "froslass_evolve", "context": MAIN_CONTEXT,
     "column": "evolve_froslass", "value": 1},
)
AVAILABLE_ESCALATION_CLASSES = ESCALATION_CLASSES + (
    {"name": "petrel_stamp", "context": 7,
     "column": "candidate_card_id", "value": 1080},
)
# "class"   - every select in the class is scored as the escalation pilot. This
#             is the shipped mode: the class belongs to that pilot, so the
#             escalation can refuse *and* take the action on the boards where
#             they would, which keeps the policy state-sensitive.
# "confirm" - the pin still chooses, and the escalation pilot is asked only
#             when the pin's own argmax is the class's action. Strictly
#             narrower and one-directional; measured as the control and found
#             to produce the identical behaviour on 66 stored games.
# "off"     - v5 exactly.
# v20's refreshed model is teacher-unconditioned. Re-applying v6's legacy
# class escalation would splice an old 2026-08-05 pilot into a model that has
# no teacher feature or defined pin, so the legacy path remains disabled.
ESCALATION_MODE = "off"

# v16: the escalation stands down on mirror boards.
#
# v6 picked 16371703 for this class because the pinned teacher takes the
# Froslass evolve on 95.7% of its own turns and that pilot on 80.5%, so the
# escalation is a *reduction* and the target rate is the escalation pilot's.
# Off the mirror the deployed agent lands where it should: 85.6% uptake over
# the 110 v15 ladder games.  On mirror boards it lands nowhere either pilot
# stands. Replaying all 104 stored mirror decisions that offered the evolve
# through the shipped v15:
#
#     escalation on   4 / 104 would evolve
#     escalation off 33 / 104 would evolve
#     v15 actually    6 evolutions, 6 of 20 offering turns taken (30%)
#     the mirror opponents, same 60 cards, 12 of 12 offering turns (100%)
#
# Fisher exact on 6/20 against 12/12 is p = 0.000112, and no pilot in the
# corpus plays at 30%. An escalation whose whole justification is "copy a
# better pilot on this class" is not doing that here, so it is suspended for
# the one matchup where it is measurably off, and left exactly as v15 for
# every other board. ``GRIMMSNARL_ESCALATION_MIRROR=on`` restores v15.
ESCALATION_MIRROR_DEFAULT = "off"


def escalation_mode() -> str:
    """Deployed mode, overridable for the A/B probes only."""
    mode = os.environ.get("GRIMMSNARL_ESCALATION", ESCALATION_MODE)
    return mode if mode in ("class", "confirm", "off") else ESCALATION_MODE


def escalation_in_mirror() -> bool:
    """Whether the escalation is allowed to fire on a mirror board."""
    value = os.environ.get(
        "GRIMMSNARL_ESCALATION_MIRROR", ESCALATION_MIRROR_DEFAULT
    )
    return value == "on"


def escalation_classes() -> tuple[dict[str, Any], ...]:
    """Enabled classes. The env override exists for the probes only."""
    names = os.environ.get("GRIMMSNARL_ESCALATION_CLASSES")
    if not names:
        return ESCALATION_CLASSES
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    chosen = tuple(
        spec for spec in AVAILABLE_ESCALATION_CLASSES
        if spec["name"] in wanted
    )
    return chosen or ESCALATION_CLASSES


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
        self.default_teacher_code = self.teacher_code
        self.teacher_index = (
            self.names.index("teacher_team_id")
            if "teacher_team_id" in self.names else -1
        )
        if (self.teacher_index >= 0) != (self.teacher_code is not None):
            raise ValueError(
                "model has teacher_team_id but no teacher_team_code (or "
                "vice versa); inference would score an unseen pilot"
            )
        self.escalation_mode = escalation_mode()
        self.escalation_in_mirror = escalation_in_mirror()
        # A class whose column the model never saw cannot be detected, so it is
        # dropped rather than left to read a missing feature as -1.
        self.escalation_classes = tuple(
            spec for spec in escalation_classes()
            if spec["column"] in self.names
        )
        # Every one of these has to hold or the escalation is a no-op that
        # would silently look like it fired: no pin column to rewrite, no class
        # left to detect, or an escalation code equal to the pin.
        self.escalation_code = (
            ESCALATION_TEACHER_CODE
            if (
                self.escalation_mode != "off"
                and self.teacher_index >= 0
                and self.escalation_classes
                and ESCALATION_TEACHER_CODE != self.teacher_code
            )
            else None
        )
        self.reset()

    def reset(self) -> None:
        # Teacher-forced replay scores our answer but must advance the
        # intra-turn history with the *teacher's* action, or the columns stop
        # describing the turn the corpus described. The evaluator does that via
        # observe_external, so commit has to stand down or the history advances
        # twice per decision and every offer/pass count comes out doubled.
        self.teacher_forced = False
        # ``getattr`` keeps the small unit-test rankers (constructed without
        # loading a model) compatible with the historical escalation tests.
        self.default_teacher_code = getattr(
            self, "default_teacher_code", getattr(self, "teacher_code", None)
        )
        self.teacher_code = self.default_teacher_code
        # Set by main from the public-information router, once per observation.
        # False means "not a mirror", which is v15's behaviour everywhere.
        self.suspend_escalation = False
        self._pending: list[dict[str, Any]] | None = None
        self._turn_key: tuple | None = None
        self._seen_candidate: dict[tuple, tuple[int, int, int]] = {}
        self._seen_class: dict[tuple, tuple[int, int]] = {}
        self._previous_offered: set[tuple] = set()
        self._position = 0
        # Score per scored option index, so a planner can keep the ranker's
        # ordering as its tie-break instead of picking an arbitrary member of
        # the set it allows.
        self.last_scores: dict[int, float] = {}
        self.stats = {
            "main_decisions": 0,
            "non_main_decisions": 0,
            "ranker_used": 0,
            "feature_errors": 0,
            "score_errors": 0,
            # The escalation is the only thing v6 changes, so its firing rate
            # and its effect are counted rather than assumed. The Alakazam line
            # lost 5.16 points of agreement to an unmeasured safety shell.
            "escalation_offered": 0,
            "escalation_scored": 0,
            "escalation_moved": 0,
            "escalation_refused_trigger": 0,
            # v16: the mirror boards where v15 would have escalated and this
            # version does not. The whole change is this counter.
            "escalation_suspended_mirror": 0,
        }
        for spec in getattr(self, "escalation_classes", ()):
            self.stats[f"escalation_offered_{spec['name']}"] = 0

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
        self.last_scores = {}
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
        spec = self._escalated_class(select, features, representatives)
        if spec is not None:
            self.stats["escalation_offered"] += 1
            self.stats[f"escalation_offered_{spec['name']}"] += 1
        try:
            if spec is not None and self.escalation_mode == "class":
                self.stats["escalation_scored"] += 1
                best_index, scores = self._score(
                    features, representatives, self.escalation_code
                )
                pinned_index, _ = self._score(
                    features, representatives, self.teacher_code
                )
                # Counted, not used: the pin's answer is what v5 would have
                # played here, so the difference is the deployed change rate
                # of the whole version and it must be a measured number.
                if pinned_index != best_index:
                    self.stats["escalation_moved"] += 1
                    if self._in_class(features[pinned_index], spec):
                        self.stats["escalation_refused_trigger"] += 1
            else:
                best_index, scores = self._score(
                    features, representatives, self.teacher_code
                )
                if spec is not None and self._in_class(
                    features[best_index], spec
                ):
                    # "confirm": the pin wants the class's action, so the
                    # escalation pilot gets a veto over that one action and
                    # nothing else.
                    self.stats["escalation_scored"] += 1
                    moved, escalated_scores = self._score(
                        features, representatives, self.escalation_code
                    )
                    if moved != best_index:
                        self.stats["escalation_moved"] += 1
                        self.stats["escalation_refused_trigger"] += 1
                        best_index, scores = moved, escalated_scores
            self.last_scores = scores
        except Exception:
            self.stats["score_errors"] += 1
            self.last_scores = {}
            return None
        self.stats["ranker_used"] += 1
        return best_index

    def _score(
        self,
        features: list[dict[str, Any]],
        representatives: list[int],
        teacher_code: Any,
    ) -> tuple[int, dict[int, float]]:
        """Argmax and every score, all as the same pilot.

        One teacher code per argmax is the invariant: scores from two different
        pilots are two different functions, and comparing them inside one
        comparison would be comparing two scales.
        """
        best_index = representatives[0]
        best_score = None
        scores: dict[int, float] = {}
        for position in representatives:
            row = features[position]
            vector = [float(row.get(name, -1)) for name in self.names]
            if self.teacher_index >= 0:
                vector[self.teacher_index] = float(teacher_code)
            score = tree_score(vector, self.model)
            scores[position] = score
            if best_score is None or score > best_score:
                best_score = score
                best_index = position
        return best_index, scores

    @staticmethod
    def _in_class(row: dict[str, Any], spec: dict[str, Any]) -> bool:
        return int(row.get(spec["column"], -1)) == spec["value"]

    def _escalated_class(
        self,
        select: dict[str, Any],
        features: list[dict[str, Any]],
        representatives: list[int],
    ) -> dict[str, Any] | None:
        """The class this select belongs to, or None.

        Only the scored representatives count: they are the set the argmax runs
        over, so an option the collapse rule dropped cannot define the class.
        """
        if self.escalation_code is None:
            return None
        context = int(select.get("context", -1))
        for spec in self.escalation_classes:
            if spec["context"] != context:
                continue
            if not any(
                self._in_class(features[position], spec)
                for position in representatives
            ):
                continue
            if self.suspend_escalation and not self.escalation_in_mirror:
                # Counted where it would have fired, so the size of the v16
                # change is a measured number rather than an assumed one.
                self.stats["escalation_suspended_mirror"] += 1
                return None
            return spec
        return None

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
        result: dict[str, Any] = dict(self.stats)
        result.update({
            "teacher_code": self.teacher_code,
            "default_teacher_code": self.default_teacher_code,
        })
        return result
