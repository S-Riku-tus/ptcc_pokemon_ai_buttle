"""v7 value-head and counterfactual-search invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(AGENT_DIR))

import search_planner as S  # noqa: E402


def state(result: int = -1) -> dict:
    return {
        "turn": 5,
        "turnActionCount": 1,
        "yourIndex": 0,
        "firstPlayer": 0,
        "supporterPlayed": False,
        "stadiumPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": result,
        "stadium": [],
        "looking": None,
        "players": [
            {
                "active": [], "bench": [], "deckCount": 50,
                "discard": [], "prize": [None] * 6,
                "handCount": 4, "hand": [],
            },
            {
                "active": [], "bench": [], "deckCount": 50,
                "discard": [], "prize": [None] * 6,
                "handCount": 4, "hand": None,
            },
        ],
    }


def observation() -> dict:
    return {
        "current": state(),
        "logs": [],
        "search_begin_input": "stub",
        "select": {
            "type": 0, "context": 0, "minCount": 1, "maxCount": 1,
            "option": [{"type": 7}, {"type": 7}],
        },
    }


def test_value_model_contains_only_state_features() -> None:
    model = json.loads((AGENT_DIR / "value_model.json").read_text())
    names = model["feature_names"]
    assert model["kind"] == "grimmsnarl_state_value"
    assert model["scope"] == "public_next_turn"
    assert len(names) == 381
    assert not any("candidate" in name or name.startswith("option_")
                   for name in names)
    assert not any(name.startswith("hand_") for name in names)


def test_hidden_predictions_have_engine_required_lengths() -> None:
    hidden = S._hidden_state(observation(), 0)
    assert [len(values) for values in hidden] == [50, 6, 50, 6, 4, 0]
    legal = set(S.fallback_policy.MY_DECK)
    assert all(card_id in legal for values in hidden for card_id in values)


class FakeRanker:
    def __init__(self) -> None:
        self.teacher_forced = False
        self.pending = [
            {"action_type_id": 1, "candidate_card_id": 100},
            {"action_type_id": 2, "candidate_card_id": 200},
        ]

    def save_dynamic_state(self):
        return {"_pending": list(self.pending), "teacher_forced": False}

    def restore_dynamic_state(self, saved):
        self.teacher_forced = saved["teacher_forced"]

    def commit(self, chosen):
        return None

    def is_scorable(self, select):
        return False

    def observe_external(self, observation, chosen):
        return None


class FakeSearch:
    def __init__(self) -> None:
        self.ended = 0

    def begin(self, observation, hidden):
        return {"searchId": 0, "observation": observation}

    def step(self, search_id, selection):
        # Candidate 0 loses; candidate 1 wins for perspective seat 0.
        result = 1 if selection == [0] else 0
        return {
            "searchId": selection[0] + 1,
            "observation": {
                "current": state(result), "select": None, "logs": []
            },
        }

    def end(self):
        self.ended += 1


def test_large_counterfactual_value_gain_overrides_imitation() -> None:
    planner = object.__new__(S.SearchPlanner)
    planner.disabled = False
    planner.search = FakeSearch()
    planner.value = object()
    planner.reset()
    chosen = planner.adjust(
        observation(), proposed=0, scores={0: 3.0, 1: 2.5},
        ranker=FakeRanker(),
    )
    snapshot = planner.snapshot()
    assert chosen == 1
    assert snapshot["overrides"] == 1
    assert snapshot["override_records"][0]["gain"] == 1.0
    assert planner.search.ended == 1


def test_search_is_limited_to_once_per_turn() -> None:
    planner = object.__new__(S.SearchPlanner)
    planner.disabled = False
    planner.search = FakeSearch()
    planner.value = object()
    planner.reset()
    ranker = FakeRanker()
    planner.adjust(observation(), 0, {0: 3.0, 1: 2.5}, ranker)
    planner.adjust(observation(), 0, {0: 3.0, 1: 2.5}, ranker)
    assert planner.snapshot()["searched"] == 1
    assert planner.snapshot()["skip_already_searched_turn"] == 1
