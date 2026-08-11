"""The conditional teacher escalation: what it fires on, and what it cannot do.

The escalation is the only thing v6 changes, and the two ways it could go wrong
are both silent. It could fire where it was never meant to - outside MAIN, or on
a decision that does not offer the Froslass evolve at all - and it could mix
scores from two pilots inside one argmax, which would compare two different
functions and call the larger number the better move. Both are pinned here.

``tree_score`` is replaced with a recorder, so every assertion is about which
teacher code reached the trees rather than about what the 45 MB model happens to
think of a synthetic board.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parents[2]
sys.path.insert(0, str(AGENT_DIR))

import ml_runtime  # noqa: E402
from ml_runtime import Ranker  # noqa: E402

PIN_CODE = 16
ELITE_CODE = ml_runtime.ESCALATION_TEACHER_CODE
FROSLASS_CLASS = ml_runtime.ESCALATION_CLASSES[0]
TRIGGER = FROSLASS_CLASS["column"]
NAMES = ["candidate_card_id", TRIGGER, "teacher_team_id"]


def build(mode: str, scores: dict[tuple[int, int], float]) -> Ranker:
    """A Ranker with no model: ``scores[(teacher_code, card_id)]`` decides.

    Everything ``choose`` reads is set explicitly, so a change to the real
    constructor cannot quietly make these tests pass for the wrong reason.
    """
    ranker = object.__new__(Ranker)
    ranker.model = {"trees": []}
    ranker.names = list(NAMES)
    ranker.contexts = frozenset({0, 5})
    ranker.teacher_code = PIN_CODE
    ranker.teacher_index = NAMES.index("teacher_team_id")
    ranker.escalation_mode = mode
    ranker.escalation_classes = (
        () if mode == "off" else (dict(FROSLASS_CLASS),)
    )
    ranker.escalation_code = None if mode == "off" else ELITE_CODE
    ranker.reset()
    ranker.calls: list[tuple[int, int]] = []

    def recorder(row: list[float], model: dict) -> float:
        key = (int(row[ranker.teacher_index]), int(row[0]))
        ranker.calls.append(key)
        return scores[key]

    ranker._tree_score = recorder
    return ranker


@pytest.fixture(autouse=True)
def _stub_tree_score(monkeypatch: pytest.MonkeyPatch):
    """Route ``tree_score`` to whichever Ranker is being exercised."""
    holder: dict[str, Ranker] = {}

    def dispatch(row, model):
        return holder["ranker"]._tree_score(row, model)

    monkeypatch.setattr(ml_runtime, "tree_score", dispatch)
    yield holder


def observation(cards: list[int], trigger: list[int], context: int = 0) -> dict:
    """A select whose options are distinguishable by ``candidate_card_id``."""
    return {
        "current": {"turn": 3},
        "select": {
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"index": card} for card in cards],
        },
        "_cards": cards,
        "_trigger": trigger,
    }


def wire(ranker: Ranker, holder: dict, observation_: dict) -> None:
    cards = observation_["_cards"]
    trigger = observation_["_trigger"]
    rows = [
        {"candidate_card_id": card, TRIGGER: int(card in trigger)}
        for card in cards
    ]
    ranker._rows = lambda _: (rows, list(range(len(rows))))
    ranker._turn_state = lambda *_: None
    holder["ranker"] = ranker


# ----- the class mode, which is what ships -----------------------------------


def test_class_mode_scores_the_whole_set_as_the_escalation_pilot(
    _stub_tree_score,
) -> None:
    # The pin would take the Froslass evolve (card 104); the escalation pilot
    # prefers the energy attachment (card 7).
    scores = {
        (PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0,
        (ELITE_CODE, 104): 1.0, (ELITE_CODE, 7): 9.0,
    }
    ranker = build("class", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 1
    assert ranker.stats["escalation_offered"] == 1
    assert ranker.stats["escalation_scored"] == 1
    assert ranker.stats["escalation_moved"] == 1
    assert ranker.stats["escalation_refused_trigger"] == 1


def test_class_mode_keeps_one_teacher_code_per_argmax(_stub_tree_score) -> None:
    """``last_scores`` must be one pilot's opinion, not a mixture."""
    scores = {
        (PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0,
        (ELITE_CODE, 104): 1.0, (ELITE_CODE, 7): 9.0,
    }
    ranker = build("class", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)
    ranker.choose(obs)

    # The planner tie-breaks on last_scores, so those have to be the scores the
    # returned index actually won under.
    assert ranker.last_scores == {0: 1.0, 1: 9.0}


def test_class_mode_can_still_take_the_evolve(_stub_tree_score) -> None:
    """It is a teacher swap, not a veto: the elite pilot evolves 80% of turns."""
    scores = {
        (PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0,
        (ELITE_CODE, 104): 9.0, (ELITE_CODE, 7): 1.0,
    }
    ranker = build("class", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 0
    assert ranker.stats["escalation_scored"] == 1
    assert ranker.stats["escalation_moved"] == 0


def test_class_mode_counts_a_move_that_is_not_a_refusal(
    _stub_tree_score,
) -> None:
    """The class belongs to the escalation pilot, so unrelated picks can move.

    That is the cost of the design and it has to be visible: a move where the
    pin was not taking the evolve counts as moved but not as a refusal.
    """
    scores = {
        (PIN_CODE, 104): 1.0, (PIN_CODE, 7): 9.0, (PIN_CODE, 1182): 5.0,
        (ELITE_CODE, 104): 1.0, (ELITE_CODE, 7): 5.0, (ELITE_CODE, 1182): 9.0,
    }
    ranker = build("class", scores)
    obs = observation([104, 7, 1182], trigger=[104])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 2
    assert ranker.stats["escalation_moved"] == 1
    assert ranker.stats["escalation_refused_trigger"] == 0


# ----- what it must not touch ------------------------------------------------


def test_no_trigger_offered_is_scored_by_the_pin_alone(
    _stub_tree_score,
) -> None:
    scores = {(PIN_CODE, 7): 9.0, (PIN_CODE, 1182): 1.0}
    ranker = build("class", scores)
    obs = observation([7, 1182], trigger=[])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 0
    assert ranker.stats["escalation_offered"] == 0
    assert ranker.stats["escalation_scored"] == 0
    assert {code for code, _ in ranker.calls} == {PIN_CODE}


def test_a_non_main_context_never_escalates(_stub_tree_score) -> None:
    """The trigger column exists on every row; only MAIN owns this class."""
    scores = {(PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0}
    ranker = build("class", scores)
    obs = observation([104, 7], trigger=[104], context=5)
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 0
    assert ranker.stats["escalation_offered"] == 0
    assert {code for code, _ in ranker.calls} == {PIN_CODE}


def test_off_mode_is_v5(_stub_tree_score) -> None:
    scores = {(PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0}
    ranker = build("off", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 0
    assert ranker.stats["escalation_offered"] == 0
    assert {code for code, _ in ranker.calls} == {PIN_CODE}


# ----- the conservative control ----------------------------------------------


def test_confirm_mode_does_not_ask_when_the_pin_is_not_evolving(
    _stub_tree_score,
) -> None:
    scores = {(PIN_CODE, 104): 1.0, (PIN_CODE, 7): 9.0}
    ranker = build("confirm", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 1
    assert ranker.stats["escalation_offered"] == 1
    assert ranker.stats["escalation_scored"] == 0
    assert {code for code, _ in ranker.calls} == {PIN_CODE}


def test_confirm_mode_vetoes_only_the_evolve(_stub_tree_score) -> None:
    scores = {
        (PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0,
        (ELITE_CODE, 104): 1.0, (ELITE_CODE, 7): 9.0,
    }
    ranker = build("confirm", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 1
    assert ranker.stats["escalation_scored"] == 1
    assert ranker.stats["escalation_moved"] == 1
    assert ranker.last_scores == {0: 1.0, 1: 9.0}


def test_confirm_mode_keeps_the_evolve_when_the_pilot_agrees(
    _stub_tree_score,
) -> None:
    scores = {
        (PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0,
        (ELITE_CODE, 104): 9.0, (ELITE_CODE, 7): 1.0,
    }
    ranker = build("confirm", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)

    assert ranker.choose(obs) == 0
    assert ranker.stats["escalation_moved"] == 0
    # The pin's own scores stay in place when nothing moved.
    assert ranker.last_scores == {0: 9.0, 1: 1.0}


# ----- the wiring that would make the whole thing a silent no-op -------------


def test_mode_is_overridable_only_to_a_known_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRIMMSNARL_ESCALATION", "confirm")
    assert ml_runtime.escalation_mode() == "confirm"
    monkeypatch.setenv("GRIMMSNARL_ESCALATION", "nonsense")
    assert ml_runtime.escalation_mode() == ml_runtime.ESCALATION_MODE


def test_shipped_mode_is_the_one_the_results_report_measured() -> None:
    assert ml_runtime.ESCALATION_MODE == "off"
    assert ml_runtime.ESCALATION_TEACHER_TEAM == 16371703


def test_the_deployed_model_is_unconditioned_and_escalation_is_off() -> None:
    """v19 learned consensus directly, so there is no runtime pilot pin."""
    tail = (AGENT_DIR / "ranker_model.json").read_bytes()[-4096:]
    match = re.search(rb'"teacher_team_code":\s*(\d+)', tail)
    assert match is None
    assert ml_runtime.ESCALATION_MODE == "off"


def test_escalation_code_is_the_corpus_dense_code_for_that_team() -> None:
    """LightGBM bins categoricals over the raw range, so the pilot reaches the
    trees as a dense 0..N-1 code. If the corpus is rebuilt with a different team
    set every code shifts and the escalation would silently score a *different*
    pilot - this is the test that fails instead.
    """
    corpus = (
        ROOT / "data" / "ml" / "grimmsnarl" / "processed"
        / "corpus_v5_data_refresh_candidate.npz"
    )
    if not corpus.exists():
        pytest.skip("training corpus not present in this checkout")
    numpy = pytest.importorskip("numpy")
    teams = sorted(
        {int(x) for x in numpy.load(corpus, allow_pickle=False)["team_ids"]}
    )
    codes = {team: index for index, team in enumerate(teams)}
    assert codes[ml_runtime.ESCALATION_TEACHER_TEAM] == ELITE_CODE
    assert codes[16494330] == PIN_CODE


def test_metadata_records_the_escalation() -> None:
    """A behaviour change that is not in the metadata cannot be reproduced."""
    meta = json.loads((AGENT_DIR / "metadata.json").read_text(encoding="utf-8"))
    escalation = meta["policy_escalation"]
    assert escalation["mode"] == ml_runtime.ESCALATION_MODE
    assert escalation["teacher_team_id"] == ml_runtime.ESCALATION_TEACHER_TEAM
    assert escalation["teacher_code"] == ELITE_CODE
    assert escalation["trigger_feature"] == TRIGGER


# ----- the class table -------------------------------------------------------


def test_only_the_froslass_class_ships() -> None:
    """One class, so one ladder run measures one change."""
    assert [spec["name"] for spec in ml_runtime.ESCALATION_CLASSES] == [
        "froslass_evolve"
    ]
    assert FROSLASS_CLASS["context"] == 0
    assert FROSLASS_CLASS["column"] == "evolve_froslass"
    assert FROSLASS_CLASS["value"] == 1


def test_the_petrel_class_is_defined_but_held_back() -> None:
    """It is measured in RESULTS.md and pre-registered for the next version."""
    names = [spec["name"] for spec in ml_runtime.AVAILABLE_ESCALATION_CLASSES]
    assert names == ["froslass_evolve", "petrel_stamp"]
    petrel = ml_runtime.AVAILABLE_ESCALATION_CLASSES[1]
    assert petrel["context"] == 7 and petrel["value"] == 1080
    assert petrel not in ml_runtime.ESCALATION_CLASSES


def test_env_selects_measured_classes_and_never_an_unknown_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRIMMSNARL_ESCALATION_CLASSES", "petrel_stamp")
    assert [s["name"] for s in ml_runtime.escalation_classes()] == [
        "petrel_stamp"
    ]
    monkeypatch.setenv("GRIMMSNARL_ESCALATION_CLASSES", "nonsense")
    assert ml_runtime.escalation_classes() == ml_runtime.ESCALATION_CLASSES
    monkeypatch.delenv("GRIMMSNARL_ESCALATION_CLASSES")
    assert ml_runtime.escalation_classes() == ml_runtime.ESCALATION_CLASSES


def test_the_class_that_fired_is_named_in_the_counters(
    _stub_tree_score,
) -> None:
    scores = {
        (PIN_CODE, 104): 9.0, (PIN_CODE, 7): 1.0,
        (ELITE_CODE, 104): 1.0, (ELITE_CODE, 7): 9.0,
    }
    ranker = build("class", scores)
    obs = observation([104, 7], trigger=[104])
    wire(ranker, _stub_tree_score, obs)
    ranker.choose(obs)

    assert ranker.stats["escalation_offered_froslass_evolve"] == 1
