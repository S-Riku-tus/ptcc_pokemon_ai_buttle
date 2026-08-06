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

v6 adds one thing and it is not a shell either: **the pilot is chosen per
decision class rather than once for the whole agent.**

``teacher_team_id`` is a categorical column, so the pin is a knob at inference
time, not a property of the weights. v1-v5 set it once - team 16494330, rated
1077.6 - and that pilot turns out to be the field's *worst* teacher on exactly
the behaviours the v5 ladder run still had open. Measured over their own games
in the 4,097-game same-deck corpus:

                              Froslass evolve, own turns    dead Unfair Stamp
    16494330  1077.6  (our pin)          95.7%                   71.3%
    16371703  1220.2                     80.5%                   50.7%
    16561259  1126.3                     72.6%                   33.0%
    field median                          ~96% .. 73%             ~52%

v5 plays the Froslass evolve on 100% of the turns it is offered. It is not
diverging from its teacher there; it is copying them faithfully. So the fix is
not another feature column - v3 and v4 spent 91 of them failing to move a MAIN
preference - it is to ask a *different* pilot about this one decision class.

Which pilot is a measurement, not a rating lookup: fidelity runs inversely to
rating on this deck, so the strongest pilot is not automatically the one whose
policy the model can actually reproduce. Per-team Top-1 on the refreshed test
block is 0.797 for 16371703, 0.813 for 16422241 and 0.839 for 16561259 against
0.928 for the incumbent pin, and the escalation teacher is picked by the
deployed behaviour that combination produces on stored boards, not by either
number alone. See ``experiments/grimmsnarl_ml_v6``.

The escalation is deliberately kept to a decision *class*: a MAIN select that
offers a Froslass evolve is scored end to end as the escalation pilot, so every
score inside one argmax comes from one teacher code and the comparison stays on
a single scale. Every other decision in the game is byte-for-byte v5.
"""

from __future__ import annotations

import copy
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
# class is about.  The new v7 deliberately restores v6's routing table.  Its
# broader change is the state-value search layer, not a second teacher pin.
ESCALATION_CLASSES = (
    {"name": "froslass_evolve", "context": MAIN_CONTEXT,
     "column": "evolve_froslass", "value": 1,
     "teacher_team": ESCALATION_TEACHER_TEAM,
     "teacher_code": ESCALATION_TEACHER_CODE},
)
AVAILABLE_ESCALATION_CLASSES = ESCALATION_CLASSES
# "class"   - every select in the class is scored as the escalation pilot. This
#             is the shipped mode: the class belongs to that pilot, so the
#             escalation can refuse *and* take the action on the boards where
#             they would, which keeps the policy state-sensitive.
# "confirm" - the pin still chooses, and the escalation pilot is asked only
#             when the pin's own argmax is the class's action. Strictly
#             narrower and one-directional; measured as the control and found
#             to produce the identical behaviour on 66 stored games.
# "off"     - v5 exactly.
ESCALATION_MODE = "class"


def escalation_mode() -> str:
    """Deployed mode, overridable for the A/B probes only."""
    mode = os.environ.get("GRIMMSNARL_ESCALATION", ESCALATION_MODE)
    return mode if mode in ("class", "confirm", "off") else ESCALATION_MODE


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
        }
        for spec in getattr(self, "escalation_classes", ()):
            self.stats[f"escalation_offered_{spec['name']}"] = 0
            self.stats[f"escalation_scored_{spec['name']}"] = 0
            self.stats[f"escalation_moved_{spec['name']}"] = 0
            self.stats[f"escalation_refused_trigger_{spec['name']}"] = 0

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
                escalation_code = self._escalation_code(spec)
                self.stats["escalation_scored"] += 1
                self.stats[f"escalation_scored_{spec['name']}"] += 1
                best_index, scores = self._score(
                    features, representatives, escalation_code
                )
                pinned_index, _ = self._score(
                    features, representatives, self.teacher_code
                )
                # Counted, not used: the pin's answer is what v5 would have
                # played here, so the difference is the deployed change rate
                # of the whole version and it must be a measured number.
                if pinned_index != best_index:
                    self.stats["escalation_moved"] += 1
                    self.stats[f"escalation_moved_{spec['name']}"] += 1
                    if self._in_class(features[pinned_index], spec):
                        self.stats["escalation_refused_trigger"] += 1
                        self.stats[
                            f"escalation_refused_trigger_{spec['name']}"
                        ] += 1
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
                    escalation_code = self._escalation_code(spec)
                    self.stats["escalation_scored"] += 1
                    self.stats[f"escalation_scored_{spec['name']}"] += 1
                    moved, escalated_scores = self._score(
                        features, representatives, escalation_code
                    )
                    if moved != best_index:
                        self.stats["escalation_moved"] += 1
                        self.stats["escalation_refused_trigger"] += 1
                        self.stats[f"escalation_moved_{spec['name']}"] += 1
                        self.stats[
                            f"escalation_refused_trigger_{spec['name']}"
                        ] += 1
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

    def _escalation_code(self, spec: dict[str, Any]) -> Any:
        """Teacher code for one class, with the legacy probe override.

        v6 exposed ``ranker.escalation_code`` to the counterfactual probe.  A
        probe that explicitly replaces that value should still be able to
        sweep one teacher across a selected class; production keeps the class
        table's code because the two shipped classes have different teachers.
        """
        configured = spec.get("teacher_code", self.escalation_code)
        if getattr(self, "escalation_code_overridden", False):
            return self.escalation_code
        return configured

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
            if any(
                self._in_class(features[position], spec)
                for position in representatives
            ):
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

    def save_dynamic_state(self) -> dict[str, Any]:
        """Snapshot only mutable per-game state for counterfactual rollouts.

        The 45 MB tree ensemble is intentionally not copied.  Search branches
        may advance the same-turn history as if their candidate had been
        played, then restore this compact snapshot before the real action is
        committed.
        """
        return copy.deepcopy({
            "teacher_forced": self.teacher_forced,
            "_pending": self._pending,
            "_turn_key": self._turn_key,
            "_seen_candidate": self._seen_candidate,
            "_seen_class": self._seen_class,
            "_previous_offered": self._previous_offered,
            "_position": self._position,
            "last_scores": self.last_scores,
            "stats": self.stats,
        })

    def restore_dynamic_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot returned by :meth:`save_dynamic_state`."""
        for name, value in state.items():
            setattr(self, name, copy.deepcopy(value))

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)
