"""Invariants of the vfinal search layer.

The layer is allowed to overrule a ranker that imitates a 1220-rated pilot, so
what has to be pinned is not that it is clever but that it is *narrow*: it only
speaks when a line takes a prize the ranker's action cannot, it never guesses
at hidden information it does not have, and any failure inside it is silent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT = Path(__file__).resolve().parents[1]
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

import turn_search  # noqa: E402

DECK = [
    int(line) for line in (AGENT / "deck.csv").read_text(encoding="utf-8").split()
    if line.strip()
]
GRIMMSNARL_EX = 648
DARK_ENERGY = 7


def board(hand_ids, deck_count, prize_count, discard_ids=(), opponent_hidden=True):
    """A minimal but count-consistent observation for the determinizer."""
    return {
        "search_begin_input": "stub",
        "select": {"context": 0, "minCount": 1, "maxCount": 1, "option": []},
        "current": {
            "turn": 5,
            "turnActionCount": 1,
            "yourIndex": 0,
            "result": -1,
            "stadium": [],
            "players": [
                {
                    "active": [{
                        "id": GRIMMSNARL_EX, "serial": 1, "hp": 320,
                        "maxHp": 320, "energies": [7, 7],
                        "energyCards": [
                            {"id": DARK_ENERGY, "serial": 90},
                            {"id": DARK_ENERGY, "serial": 91},
                        ],
                        "tools": [], "preEvolution": [],
                    }],
                    "bench": [],
                    "discard": [{"id": card, "serial": 200 + i}
                                for i, card in enumerate(discard_ids)],
                    "hand": [{"id": card, "serial": 300 + i}
                             for i, card in enumerate(hand_ids)],
                    "handCount": len(hand_ids),
                    "deckCount": deck_count,
                    "prize": [None] * prize_count,
                },
                {
                    "active": [None] if opponent_hidden else [{
                        "id": 743, "serial": 2, "hp": 150, "maxHp": 150,
                        "energies": [], "energyCards": [], "tools": [],
                        "preEvolution": [],
                    }],
                    "bench": [], "discard": [], "hand": None,
                    "handCount": 4, "deckCount": 40, "prize": [None] * 6,
                },
            ],
        },
    }


# ----- determinization ------------------------------------------------------


def test_a_face_down_opponent_active_is_not_guessed() -> None:
    """We do not know which Basic it is, so the layer must decline, not invent."""
    with pytest.raises(turn_search.SearchUnavailable):
        turn_search.determinize(board(["7"], 40, 6), DECK, _rng())


def test_hidden_counts_match_the_engine_arguments() -> None:
    # Public: the Active Grimmsnarl ex, the two Dark Energy attached to it,
    # and four Dark Energy in hand - seven of the sixty.
    observation = board([DARK_ENERGY] * 4, 47, 6, opponent_hidden=False)
    own_deck, own_prize, opp_deck, opp_prize, opp_hand, opp_active = (
        turn_search.determinize(observation, DECK, _rng())
    )
    assert len(own_deck) == 47
    assert len(own_prize) == 6
    assert len(opp_deck) == 40
    assert len(opp_prize) == 6
    assert len(opp_hand) == 4
    assert opp_active == []
    # Nothing may be invented: our sampled hidden zones plus the public cards
    # have to reconstitute the 60-card list exactly.
    public = turn_search.own_public_ids(observation["current"], 0)
    assert sorted(own_deck + own_prize + public) == sorted(DECK)


def test_a_public_card_outside_our_list_stops_the_search() -> None:
    """A count mismatch means our model of the board is wrong; do not sample."""
    observation = board([9999], 44, 6, opponent_hidden=False)
    with pytest.raises(turn_search.SearchUnavailable):
        turn_search.determinize(observation, DECK, _rng())


def test_wrong_deck_count_stops_the_search() -> None:
    observation = board([DARK_ENERGY] * 4, 12, 6, opponent_hidden=False)
    with pytest.raises(turn_search.SearchUnavailable):
        turn_search.determinize(observation, DECK, _rng())


def _rng():
    import random

    return random.Random(0)


# ----- option identity ------------------------------------------------------


def test_signatures_separate_options_that_differ_only_in_target() -> None:
    """A plan is replayed by signature, so two targets must not collide."""
    first = {"type": 7, "index": 0, "inPlayArea": 5, "inPlayIndex": 0}
    second = {"type": 7, "index": 0, "inPlayArea": 5, "inPlayIndex": 1}
    assert turn_search.option_signature(first) != turn_search.option_signature(second)


def test_signatures_ignore_nothing_that_identifies_an_attack() -> None:
    shadow = {"type": 13, "attackId": 937}
    corkscrew = {"type": 13, "attackId": 936}
    assert turn_search.option_signature(shadow) != turn_search.option_signature(corkscrew)


def test_a_hand_option_is_named_by_its_card_not_its_slot() -> None:
    """The engine never sets Option.cardId, so position must be resolved."""
    observation = board([1182, 1080, 1227], 44, 6, opponent_hidden=False)
    options = [
        {"type": 3, "area": 2, "index": 0, "playerIndex": 0},
        {"type": 3, "area": 2, "index": 1, "playerIndex": 0},
    ]
    first, second = (
        turn_search.option_signature(o, observation) for o in options
    )
    assert first != second
    assert ("card", 1182) in first and ("card", 1080) in second

    # Same hand, different order: the signature has to follow the card.
    reordered = board([1080, 1182, 1227], 44, 6, opponent_hidden=False)
    resolved = turn_search.resolve_selection(
        {**reordered, "select": {**reordered["select"], "option": options}},
        (first,),
    )
    assert resolved == [1]


def test_an_in_play_option_is_named_by_serial() -> None:
    observation = board([1182], 46, 6, opponent_hidden=False)
    option = {"type": 7, "area": 2, "index": 0, "playerIndex": 0,
              "inPlayArea": 4, "inPlayIndex": 0}
    signature = turn_search.option_signature(option, observation)
    assert ("serial", 1) in signature


# ----- Shadow Bullet range --------------------------------------------------


def test_dark_weakness_doubles_the_shadow_bullet_threshold() -> None:
    table = turn_search.load_weakness_table()
    if not table:
        pytest.skip("native card table unavailable")
    # Slowking is Darkness-weak, Teal Mask Ogerpon ex is not.
    assert table.get(163) == turn_search.DARKNESS
    assert table.get(96) != turn_search.DARKNESS
    assert turn_search.in_shadow_range({"id": 163, "hp": 300}) is True
    assert turn_search.in_shadow_range({"id": 96, "hp": 180}) is True
    assert turn_search.in_shadow_range({"id": 96, "hp": 210}) is False


# ----- authority ------------------------------------------------------------


class _StubLines(turn_search.TurnSearch):
    """Replace the engine with fixed lines so the gate can be tested alone."""

    def __init__(self, lines, **kwargs):
        self._lines_out = lines
        self.deck_list = DECK
        self.multi_pick = None
        self.max_nodes = 10
        self.beam_width = 4
        self.branch_cap = 4
        self.max_depth = 4
        self.determinizations = kwargs.get("determinizations", 2)
        self.per_decision_seconds = 5.0
        self.authority = kwargs.get("authority", 1)
        self.commit_plan = kwargs.get("commit_plan", True)
        self.plan = None
        self.state_guard = None
        self.api = None
        self.budget = turn_search.Budget()
        self.last = {}

    def _lines(self, observation, seat, rng, deadline):
        return self._lines_out.pop(0)


def _line(sig, prizes=0, threat=0, damage=0, result=-1, plan=None,
          uses_deck=False):
    return {
        "first": [0], "first_signature": (sig,), "prizes": prizes,
        "threat": threat, "damage": damage, "result": result, "complete": True,
        "plan": plan if plan is not None else [(sig,)],
        "uses_deck": uses_deck,
    }


def _observation_with(options):
    return {
        "search_begin_input": "stub",
        "select": {"context": 0, "minCount": 1, "maxCount": 1, "option": options},
        "current": {"turn": 7, "turnActionCount": 3, "yourIndex": 0,
                    "players": [{}, {}], "result": -1},
    }


def test_no_override_when_the_ranker_line_already_takes_the_prizes() -> None:
    options = [{"type": 13, "attackId": 937}, {"type": 13, "attackId": 936}]
    ranker, other = (turn_search.option_signature(o) for o in options)
    worlds = [
        [_line(ranker, prizes=2), _line(other, prizes=1)],
        [_line(ranker, prizes=2), _line(other, prizes=1)],
    ]
    search = _StubLines(worlds)
    assert search.suggest(_observation_with(options), 0) is None
    assert search.budget.no_improvement == 1


def test_override_when_another_opening_takes_a_prize_in_every_world() -> None:
    options = [{"type": 14}, {"type": 13, "attackId": 936}]
    ranker, other = (turn_search.option_signature(o) for o in options)
    worlds = [
        [_line(ranker, prizes=0), _line(other, prizes=1)],
        [_line(ranker, prizes=0), _line(other, prizes=1)],
    ]
    search = _StubLines(worlds)
    assert search.suggest(_observation_with(options), 0) == 1
    assert search.budget.overrides == 1


def test_one_dissenting_world_vetoes_the_override() -> None:
    """A line that only exists on one deck order is a draw, not a plan."""
    options = [{"type": 14}, {"type": 13, "attackId": 936}, {"type": 13, "attackId": 935}]
    ranker, first, second = (turn_search.option_signature(o) for o in options)
    worlds = [
        [_line(ranker, prizes=0), _line(first, prizes=1), _line(second, prizes=0)],
        [_line(ranker, prizes=0), _line(first, prizes=0), _line(second, prizes=1)],
    ]
    search = _StubLines(worlds)
    assert search.suggest(_observation_with(options), 0) is None
    assert search.budget.world_disagreement == 1


def test_threat_alone_cannot_override_at_authority_one() -> None:
    options = [{"type": 14}, {"type": 13, "attackId": 936}]
    ranker, other = (turn_search.option_signature(o) for o in options)
    worlds = [
        [_line(ranker, prizes=1, threat=0), _line(other, prizes=1, threat=1)],
        [_line(ranker, prizes=1, threat=0), _line(other, prizes=1, threat=1)],
    ]
    assert _StubLines(list(worlds)).suggest(_observation_with(options), 0) is None
    raised = _StubLines(list(worlds), authority=2)
    assert raised.suggest(_observation_with(options), 0) == 1


def test_multi_pick_and_non_main_contexts_are_left_to_the_ranker() -> None:
    search = _StubLines([])
    multi = _observation_with([{"type": 3}, {"type": 3}])
    multi["select"]["maxCount"] = 3
    assert search.suggest(multi, 0) is None
    other = _observation_with([{"type": 3}])
    other["select"]["context"] = 7
    assert search.suggest(other, 0) is None
    assert search.budget.considered == 0


def test_an_exhausted_overage_bank_disables_the_layer() -> None:
    options = [{"type": 14}, {"type": 13, "attackId": 936}]
    search = _StubLines([])
    search.budget.note({"remainingOverageTime": 60.0})
    assert search.suggest(_observation_with(options), 0) is None
    assert search.budget.skipped_budget == 1


def test_early_turns_are_never_searched() -> None:
    options = [{"type": 14}, {"type": 13, "attackId": 936}]
    observation = _observation_with(options)
    observation["current"]["turn"] = 2
    search = _StubLines([])
    assert search.suggest(observation, 0) is None
    assert search.budget.considered == 0


# ----- packaging ------------------------------------------------------------


def test_metadata_names_the_search_layer() -> None:
    meta = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    assert "turn_search" in meta["policy"]
    assert meta["deck_changed"] is False
    assert meta["deck_hash"] == "9714ab5c3996f6cc"


def test_the_agent_still_answers_with_the_layer_removed(monkeypatch) -> None:
    """A search failure must degrade to v22, never to a crash."""
    monkeypatch.setenv("GRIMMSNARL_TURN_SEARCH_DISABLE", "1")
    assert turn_search.build("deck.csv") is None


def test_an_override_commits_the_rest_of_its_line() -> None:
    """The opening is only played together with the continuation it earned."""
    options = [{"type": 14}, {"type": 13, "attackId": 936}]
    ranker, other = (turn_search.option_signature(o) for o in options)
    attack = turn_search.option_signature({"type": 13, "attackId": 937})
    line = _line(other, prizes=1, plan=[(other,), (attack,)])
    worlds = [
        [_line(ranker, prizes=0), line],
        [_line(ranker, prizes=0), _line(other, prizes=1)],
    ]
    search = _StubLines(worlds)
    assert search.suggest(_observation_with(options), 0) == 1
    assert search.plan is not None and search.plan["steps"] == [(attack,)]

    # The continuation is then replayed by signature, not by index: the attack
    # sits at a different position on the real board.
    follow = _observation_with([{"type": 13, "attackId": 935},
                                {"type": 13, "attackId": 937}])
    follow["current"]["turn"] = 7
    assert search.planned(follow) == [1]
    assert search.planned(follow) is None  # plan exhausted


def test_a_plan_is_abandoned_when_the_board_does_not_match() -> None:
    search = _StubLines([])
    attack = turn_search.option_signature({"type": 13, "attackId": 937})
    search.plan = {"turn": 7, "steps": [(attack,)], "cursor": 0}
    absent = _observation_with([{"type": 14}])
    assert search.planned(absent) is None
    assert search.plan is None
    assert search.budget.plans_abandoned == 1


def test_a_plan_does_not_survive_into_the_next_turn() -> None:
    search = _StubLines([])
    attack = turn_search.option_signature({"type": 13, "attackId": 937})
    search.plan = {"turn": 7, "steps": [(attack,)], "cursor": 0}
    later = _observation_with([{"type": 13, "attackId": 937}])
    later["current"]["turn"] = 9
    assert search.planned(later) is None
    assert search.plan is None


def test_a_line_that_reaches_into_the_deck_cannot_override() -> None:
    """It was planned on one deck order and may not resolve on the real one."""
    options = [{"type": 14}, {"type": 13, "attackId": 936}]
    ranker, other = (turn_search.option_signature(o) for o in options)
    worlds = [
        [_line(ranker, prizes=0), _line(other, prizes=2, uses_deck=True)],
        [_line(ranker, prizes=0), _line(other, prizes=2, uses_deck=True)],
    ]
    search = _StubLines(worlds)
    assert search.suggest(_observation_with(options), 0) is None
    # The deck-reaching line is excluded from the challengers, so the ranker's
    # own deck-free line is the best remaining opening and nothing improves.
    assert search.budget.no_improvement == 1

    # And when every line reaches into the deck there is nothing to commit to.
    everything_searches = [
        [_line(ranker, prizes=0, uses_deck=True),
         _line(other, prizes=2, uses_deck=True)],
    ]
    empty = _StubLines(everything_searches)
    assert empty.suggest(_observation_with(options), 0) is None
    assert empty.budget.no_committable == 1
