"""v11 arithmetic-search safety and counterfactual invariants."""

from __future__ import annotations

import copy
import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(AGENT_DIR))

import arithmetic_search as S  # noqa: E402


def card(
    card_id: int,
    serial: int,
    *,
    hp: int,
    max_hp: int | None = None,
    energy: int = 0,
) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": hp if max_hp is None else max_hp,
        "energyCards": [{"id": S.mf.DARK_ENERGY_ID}] * energy,
        "energies": [],
        "preEvolution": [],
        "tools": [],
    }


def player(active: list[dict], bench: list[dict], *, own: bool) -> dict:
    hand = [
        {"id": S.mf.RARE_CANDY_ID, "serial": 40},
        {"id": S.mf.GRIMMSNARL_EX_ID, "serial": 41},
        {"id": S.mf.DARK_ENERGY_ID, "serial": 42},
        {"id": S.mf.LILLIE_ID, "serial": 43},
    ] if own else None
    return {
        "active": active,
        "bench": bench,
        "deckCount": 46,
        "discard": [],
        "prize": [None] * 6,
        "handCount": 4,
        "hand": hand,
        "benchMax": 5,
    }


def observation() -> dict:
    me = player(
        [card(S.mf.GRIMMSNARL_EX_ID, 1, hp=320, energy=2)],
        [card(S.mf.IMPIDIMP_ID, 2, hp=70)],
        own=True,
    )
    opponent = player(
        [card(S.mf.IMPIDIMP_ID, 101, hp=100)],
        [],
        own=False,
    )
    return {
        "current": {
            "turn": 2,
            "turnActionCount": 1,
            "yourIndex": 0,
            "firstPlayer": 1,
            "result": -1,
            "players": [me, opponent],
            "stadium": [],
        },
        "logs": [],
        "search_begin_input": "stub",
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 14},
                {"type": 13, "attackId": S.mf.SHADOW_BULLET_ID},
            ],
        },
    }


def post_state(root: dict, candidate: int, *, lose_bench: bool = False) -> dict:
    output = copy.deepcopy(root)
    current = output["current"]
    current["turn"] = 3
    current["yourIndex"] = 1
    current["players"][0]["hand"] = None
    current["players"][1]["hand"] = [
        {"id": S.mf.DARK_ENERGY_ID, "serial": 90},
    ] * 4
    if candidate == 1:
        current["players"][1]["active"][0]["hp"] = 70
    if lose_bench and candidate == 1:
        current["players"][0]["bench"] = []
    output["select"] = {"context": 0, "option": [{"type": 14}]}
    return output


class FakeRanker:
    def __init__(self) -> None:
        self.teacher_forced = False
        self.pending = [
            {"action_type_id": 14, "candidate_card_id": -1},
            {"action_type_id": 13, "candidate_card_id": -1},
        ]

    def save_dynamic_state(self):
        return {
            "teacher_forced": self.teacher_forced,
            "_pending": copy.deepcopy(self.pending),
        }

    def restore_dynamic_state(self, saved):
        self.teacher_forced = saved["teacher_forced"]

    def commit(self, chosen):
        return None

    def is_scorable(self, select):
        return False

    def observe_external(self, observation, chosen):
        return None


class FakeSearch:
    def __init__(self, *, second_sample_regresses: bool = False) -> None:
        self.begin_count = 0
        self.ended = 0
        self.second_sample_regresses = second_sample_regresses

    def begin(self, root, hidden):
        self.begin_count += 1
        return {"searchId": self.begin_count * 10, "observation": root}

    def step(self, search_id, selection):
        candidate = selection[0]
        lose_bench = self.second_sample_regresses and search_id >= 20
        return {
            "searchId": search_id + candidate + 1,
            "observation": post_state(
                observation(), candidate, lose_bench=lose_bench,
            ),
        }

    def end(self):
        self.ended += 1


def planner_with(search: FakeSearch) -> S.ArithmeticSearch:
    planner = object.__new__(S.ArithmeticSearch)
    planner.disabled = False
    planner.search = search
    planner.reset()
    return planner


def test_attack_line_wins_in_both_determinizations_and_overrides() -> None:
    search = FakeSearch()
    planner = planner_with(search)
    chosen = planner.adjust(
        observation(), 0, {0: 3.0, 1: 2.0}, FakeRanker(), None,
    )
    snapshot = planner.snapshot()
    assert chosen == 1
    assert snapshot["overrides"] == 1
    assert snapshot["determinizations"] == 2
    assert snapshot["branches"] == 4
    assert search.ended == 2


def test_one_determinization_regression_returns_v8() -> None:
    planner = planner_with(FakeSearch(second_sample_regresses=True))
    chosen = planner.adjust(
        observation(), 0, {0: 3.0, 1: 2.0}, FakeRanker(), None,
    )
    assert chosen == 0
    assert planner.snapshot()["overrides"] == 0
    assert planner.snapshot()["skip_nonrobust"] == 1


def test_search_runs_at_most_once_per_turn() -> None:
    planner = planner_with(FakeSearch())
    ranker = FakeRanker()
    first = planner.adjust(observation(), 0, {0: 3.0, 1: 2.0}, ranker)
    second = planner.adjust(observation(), 0, {0: 3.0, 1: 2.0}, ranker)
    assert first == 1
    assert second == 0
    assert planner.snapshot()["searched"] == 1
    assert planner.snapshot()["skip_already_searched_turn"] == 1


def test_alakazam_going_second_gets_one_targeted_extra_search() -> None:
    planner = planner_with(FakeSearch())
    ranker = FakeRanker()
    root = observation()
    root["current"]["players"][1]["active"][0]["id"] = (
        S.fallback_policy.C.ABRA
    )
    assert planner.adjust(root, 0, {0: 3.0, 1: 2.0}, ranker) == 1
    assert planner.adjust(root, 0, {0: 3.0, 1: 2.0}, ranker) == 1
    assert planner.adjust(root, 0, {0: 3.0, 1: 2.0}, ranker) == 0
    snapshot = planner.snapshot()
    assert snapshot["searched"] == 2
    assert snapshot["alakazam_second_extra_searches"] == 1
    assert snapshot["skip_already_searched_turn"] == 1


def test_alakazam_trigger_stands_down_when_going_first() -> None:
    planner = planner_with(FakeSearch())
    ranker = FakeRanker()
    root = observation()
    root["current"]["firstPlayer"] = 0
    root["current"]["players"][1]["active"][0]["id"] = (
        S.fallback_policy.C.ALAKAZAM
    )
    assert planner.adjust(root, 0, {0: 3.0, 1: 2.0}, ranker) == 1
    assert planner.adjust(root, 0, {0: 3.0, 1: 2.0}, ranker) == 0
    assert planner.snapshot()["alakazam_second_extra_searches"] == 0


def test_planner_note_accepts_multi_pick_without_counting_an_activation() -> None:
    planner = S.ml_planner.Planner()
    root = observation()
    planner.note(root, root["select"], [0, 1])
    planner.note(root, root["select"], [])
    assert planner.snapshot()["errors"] == 0


def test_arithmetic_planner_override_is_never_undone() -> None:
    planner = planner_with(FakeSearch())
    # Index 1 can only be proposed when the proven arithmetic planner moved
    # away from the ranker's index-0 argmax.  Search must stand down.
    chosen = planner.adjust(
        observation(), 1, {0: 3.0, 1: 2.0}, FakeRanker(), None,
    )
    assert chosen == 1
    assert planner.snapshot()["searched"] == 0
    assert planner.snapshot()["skip_planner_guard"] == 1


def test_missing_search_state_returns_v8_without_starting_engine() -> None:
    planner = planner_with(FakeSearch())
    root = observation()
    root.pop("search_begin_input")
    chosen = planner.adjust(
        root, 0, {0: 3.0, 1: 2.0}, FakeRanker(), None,
    )
    assert chosen == 0
    assert planner.snapshot()["searched"] == 0
    assert planner.snapshot()["skip_no_search_state"] == 1


def test_turn_counter_rewind_reenables_search_for_next_game() -> None:
    planner = planner_with(FakeSearch())
    ranker = FakeRanker()
    assert planner.adjust(
        observation(), 0, {0: 3.0, 1: 2.0}, ranker,
    ) == 1
    next_game = observation()
    next_game["current"]["turn"] = 1
    next_game["current"]["turnActionCount"] = 0
    assert planner.adjust(
        next_game, 0, {0: 3.0, 1: 2.0}, ranker,
    ) == 1
    assert planner.snapshot()["searched"] == 2


def test_critical_regression_is_never_bought_with_soft_utility() -> None:
    incumbent = S.LeafEvaluation(
        result=0, prizes_taken=0, prizes_conceded=0,
        attacked=1, damage_dealt=30,
        active_ready=1, ready_grimms=1, bodies=2, setup_progress=6,
        useful_energy=3, wasted_energy=0, active_survival_margin=40,
        hand_plan=5, hand_count=4, deck_count=30, own_damage_added=0,
    )
    candidate = S.dataclasses.replace(
        incumbent,
        attacked=0,
        hand_plan=100,
        hand_count=20,
    )
    assert S._grade_upgrade(candidate, incumbent) < 0


def test_extra_prize_does_not_hide_a_larger_concession() -> None:
    incumbent = S.LeafEvaluation(
        result=0, prizes_taken=0, prizes_conceded=0,
        attacked=1, damage_dealt=30, active_ready=1, ready_grimms=1,
        bodies=3, setup_progress=8, useful_energy=4, wasted_energy=0,
        active_survival_margin=40, hand_plan=8, hand_count=4,
        deck_count=30, own_damage_added=0,
    )
    sacrifice = S.dataclasses.replace(
        incumbent,
        prizes_taken=1,
        prizes_conceded=2,
        bodies=2,
    )
    assert S._grade_upgrade(sacrifice, incumbent) < 0


def test_semantic_candidate_builder_keeps_attack_line() -> None:
    root = observation()
    root["select"]["option"] = [
        {"type": 10, "area": S.mf.AREA_ACTIVE, "index": 0},
        {"type": 10, "area": S.mf.AREA_BENCH, "index": 0},
        {"type": 12},
        {"type": 13, "attackId": S.mf.SHADOW_BULLET_ID},
        {"type": 14},
    ]
    scores = {0: 5.0, 1: 4.9, 2: 3.0, 3: -6.0, 4: -20.0}
    candidates = S.ArithmeticSearch._candidate_indices(root, 0, scores)
    assert 3 in candidates
    assert len(candidates) <= S.TOP_K
