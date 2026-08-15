from __future__ import annotations

import sys
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

from h2_search import DETERMINIZATIONS, H2SearchPlanner  # noqa: E402


def leaf(*, lead: int = 0, deck: int = 10) -> dict:
    own = 4
    opponent = own + lead
    return {
        "current": {
            "yourIndex": 1,
            "result": -1,
            "players": [
                {"prize": [{}] * own, "deckCount": deck},
                {"prize": [{}] * opponent, "deckCount": 10},
            ],
        }
    }


def root(turn: int = 7) -> dict:
    return {
        "search_begin_input": "stub",
        "remainingOverageTime": 600.0,
        "current": {
            "turn": turn,
            "turnActionCount": 0,
            "yourIndex": 0,
            "stadium": [],
            "players": [
                {
                    "active": [], "bench": [], "discard": [], "hand": [],
                    "handCount": 0, "prize": [None] * 6, "deckCount": 54,
                },
                {
                    "active": [], "bench": [], "discard": [], "hand": None,
                    "handCount": 0, "prize": [None] * 6, "deckCount": 54,
                },
            ],
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 7}, {"type": 14}],
        },
    }


class FakeSearch:
    def __init__(self) -> None:
        self.begins = 0
        self.ends = 0

    def begin(self, _observation, _hidden):
        search_id = self.begins
        self.begins += 1
        return {"searchId": search_id}

    def end(self):
        self.ends += 1


class FakeRanker:
    def __init__(self) -> None:
        self.state = {"_pending": [{}, {}]}

    def save_dynamic_state(self):
        return dict(self.state)

    def restore_dynamic_state(self, state):
        self.state = dict(state)


class ScriptedH2(H2SearchPlanner):
    def __init__(self, values):
        self.scripted = values
        super().__init__(search_api=FakeSearch(), value_model=object())

    def _rollout(
        self, root_id, candidate, perspective, ranker, ranker_state, **_kwargs,
    ):
        value = self.scripted[root_id][candidate]
        return leaf(lead=0), value


def test_three_determinizations_must_unanimously_dominate_v22() -> None:
    planner = ScriptedH2([{0: 0.50, 1: 0.70}] * DETERMINIZATIONS)
    chosen = planner.adjust(
        root(), 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1, is_mirror=True
    )
    assert chosen == 1
    assert planner.stats["overrides"] == 1
    assert planner.search.begins == DETERMINIZATIONS
    assert planner.search.ends == DETERMINIZATIONS


def test_hidden_state_disagreement_returns_v22() -> None:
    planner = ScriptedH2([
        {0: 0.50, 1: 0.70},
        {0: 0.75, 1: 0.60},
        {0: 0.50, 1: 0.70},
    ])
    assert planner.adjust(
        root(), 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1, is_mirror=True
    ) == 0
    assert planner.stats["skip_determinization_disagreement"] == 1


def test_non_mirror_never_calls_search() -> None:
    planner = ScriptedH2([{0: 0.50, 1: 0.70}] * DETERMINIZATIONS)
    assert planner.adjust(
        root(), 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1, is_mirror=False
    ) == 0
    assert planner.search.begins == 0
    assert planner.stats["skip_non_mirror"] == 1


def test_only_one_search_is_allowed_per_turn() -> None:
    planner = ScriptedH2([{0: 0.50, 1: 0.70}] * DETERMINIZATIONS)
    observation = root()
    assert planner.adjust(
        observation, 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1,
        is_mirror=True,
    ) == 1
    assert planner.adjust(
        observation, 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1,
        is_mirror=True,
    ) == 0
    assert planner.search.begins == DETERMINIZATIONS
    assert planner.stats["skip_already_searched_turn"] == 1


def test_incomplete_branch_returns_v22() -> None:
    class Broken(ScriptedH2):
        def _rollout(self, *args, **kwargs):
            return None

    planner = Broken([{0: 0.50, 1: 0.70}] * DETERMINIZATIONS)
    assert planner.adjust(
        root(), 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1, is_mirror=True
    ) == 0
    assert planner.stats["skip_incomplete"] == 1


def test_rollout_resolves_second_attack_promotion_before_h2_leaf() -> None:
    def state(actor: int, context: int, turn: int) -> dict:
        return {
            "searchId": turn * 10 + context,
            "observation": {
                "current": {
                    "turn": turn,
                    "yourIndex": actor,
                    "result": -1,
                    "players": [
                        {"prize": [{}] * 4, "deckCount": 10},
                        {"prize": [{}] * 4, "deckCount": 10},
                    ],
                },
                "select": {
                    "context": context,
                    "minCount": 1,
                    "maxCount": 1,
                    "option": [{"type": 14}],
                },
            },
        }

    class SequenceSearch:
        def __init__(self):
            self.states = [
                state(0, 0, 7),   # rest of root turn
                state(1, 0, 8),   # opponent reply
                state(0, 0, 9),   # our H2 turn
                state(1, 3, 10),  # promotion caused by our attack
                state(1, 0, 10),  # complete opponent-turn board
            ]
            self.steps = 0

        def step(self, _search_id, _selection):
            result = self.states[self.steps]
            self.steps += 1
            return result

    class RollRanker:
        teacher_forced = False

        def restore_dynamic_state(self, _state):
            return None

        def commit(self, _index):
            return None

        def is_scorable(self, _select):
            return False

        def observe_external(self, _observation, _chosen):
            return None

    class Value:
        def probability(self, _observation, _perspective):
            return 0.61

    planner = H2SearchPlanner(search_api=SequenceSearch(), value_model=Value())
    result = planner._rollout(1, 0, 0, RollRanker(), {})
    assert result is not None
    leaf_observation, value = result
    assert planner.search.steps == 5
    assert leaf_observation["current"]["yourIndex"] == 1
    assert leaf_observation["select"]["context"] == 0
    assert value == 0.61
