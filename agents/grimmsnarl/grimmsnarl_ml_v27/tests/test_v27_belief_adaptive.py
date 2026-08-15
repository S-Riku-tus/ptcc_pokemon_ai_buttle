from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest


AGENT = Path(__file__).resolve().parents[1]
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

import fallback_policy  # noqa: E402
from h2_search import H2SearchPlanner, _hidden_state  # noqa: E402


def card(card_id: int, **extra) -> dict:
    return {"id": card_id, **extra}


def belief_root() -> dict:
    deck = list(fallback_policy.MY_DECK)
    own_hand = [card(card_id) for card_id in deck[:7]]
    # Eight public opponent cards: a full evolution stack, two attached
    # energies, Munkidori, Poffin, and the owned stadium.
    opponent_active = card(
        648,
        hp=320,
        maxHp=320,
        preEvolution=[card(646), card(647)],
        energyCards=[card(7)],
    )
    opponent_bench = card(112, hp=110, maxHp=110, energyCards=[card(7)])
    return {
        "search_begin_input": "stub",
        "remainingOverageTime": 600.0,
        "current": {
            "turn": 9,
            "turnActionCount": 2,
            "yourIndex": 0,
            "stadium": [card(1259, playerIndex=1)],
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "hand": own_hand,
                    "handCount": len(own_hand),
                    "prize": [None, None, card(deck[7])],
                    "deckCount": 50,
                },
                {
                    "active": [opponent_active],
                    "bench": [opponent_bench],
                    "discard": [card(1086)],
                    "hand": None,
                    "handCount": 5,
                    "prize": [None, None, None],
                    "deckCount": 44,
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


def test_hidden_world_conserves_both_exact_deck_multisets() -> None:
    observation = belief_root()
    hidden = _hidden_state(observation, 0, 3)
    own_deck, own_prize, opp_deck, opp_prize, opp_hand, opp_active = hidden

    assert list(map(len, hidden)) == [50, 3, 44, 3, 5, 0]
    expected = Counter(fallback_policy.MY_DECK)
    own_known_hand = [entry["id"] for entry in observation["current"]["players"][0]["hand"]]
    assert Counter(own_known_hand + own_deck + own_prize) == expected

    opponent_public = [648, 646, 647, 7, 112, 7, 1086, 1259]
    assert Counter(
        opponent_public + opp_deck + opp_prize + opp_hand + opp_active
    ) == expected
    assert max(Counter(opponent_public + opp_deck + opp_prize + opp_hand).values()) <= 10


def test_hidden_world_rejects_an_impossible_public_mirror() -> None:
    observation = belief_root()
    opponent = observation["current"]["players"][1]
    opponent["discard"] = [card(646) for _ in range(5)]
    with pytest.raises(ValueError, match="exceeds mirror deck count"):
        _hidden_state(observation, 0, 0)


def leaf(value_lead: int = 0) -> dict:
    return {
        "current": {
            "yourIndex": 1,
            "result": -1,
            "players": [
                {"prize": [{}] * 3, "deckCount": 20},
                {"prize": [{}] * (3 + value_lead), "deckCount": 20},
            ],
        }
    }


class FakeSearch:
    def __init__(self) -> None:
        self.begins = 0
        self.ends = 0

    def begin(self, _observation, _hidden):
        result = {"searchId": self.begins}
        self.begins += 1
        return result

    def end(self) -> None:
        self.ends += 1


class FakeRanker:
    def __init__(self) -> None:
        self.state = {"_pending": [{}, {}]}

    def save_dynamic_state(self):
        return dict(self.state)

    def restore_dynamic_state(self, state):
        self.state = dict(state)


class AdaptiveScript(H2SearchPlanner):
    def __init__(self, h3_gains: list[float]) -> None:
        self.h3_gains = h3_gains
        super().__init__(search_api=FakeSearch(), value_model=object())

    def _rollout(
        self, root_id, candidate, perspective, ranker, ranker_state, *,
        future_own_turns=1, **_kwargs,
    ):
        if future_own_turns == 1:
            # H2 is close but still prefers v22, which triggers selective H3.
            return leaf(), 0.60 if candidate == 0 else 0.58
        sample = root_id - 3
        base = 0.50
        return leaf(), base if candidate == 0 else base + self.h3_gains[sample]


def test_close_tactical_h2_extends_to_five_world_h3() -> None:
    planner = AdaptiveScript([0.08] * 5)
    chosen = planner.adjust(
        belief_root(), 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1,
        is_mirror=True,
    )
    assert chosen == 1
    assert planner.search.begins == 8  # 3 H2 worlds + 5 H3 worlds
    assert planner.search.ends == 8
    assert planner.stats["overrides_h3"] == 1
    assert planner.snapshot()["records"][0]["samples"] == 5


def test_borderline_h3_adaptively_expands_from_five_to_seven_worlds() -> None:
    planner = AdaptiveScript([0.02] * 5 + [0.10, 0.10])
    chosen = planner.adjust(
        belief_root(), 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1,
        is_mirror=True,
    )
    assert chosen == 1
    assert planner.search.begins == 10  # 3 H2 + 7 H3
    assert planner.stats["h3_worlds"] == 7
    assert planner.snapshot()["records"][0]["samples"] == 7


def test_one_worse_h3_world_preserves_v22_without_more_sampling() -> None:
    planner = AdaptiveScript([0.08, -0.01, 0.08])
    chosen = planner.adjust(
        belief_root(), 0, {0: 1.0, 1: 0.9}, FakeRanker(), 1,
        is_mirror=True,
    )
    assert chosen == 0
    assert planner.search.begins == 6  # 3 H2 + only the initial 3 H3
    assert planner.stats["skip_h3_not_dominant"] == 1
