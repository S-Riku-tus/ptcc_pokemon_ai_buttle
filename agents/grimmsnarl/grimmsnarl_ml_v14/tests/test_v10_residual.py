"""The residual: what it fires on, and everything it must never touch.

A residual is only as good as the set of boards it *cannot* reach, so most of
this file is negative. The four invariants v10 is held to - never lose a legal
attachment, never lose the attachment that turns an attack on, never reduce a
Grimmsnarl ex or Froslass evolve, never overwrite a certain knockout or the
maximum prizes - are all satisfied by the context gate rather than by a
threshold, and the tests that pin that are the ones that would fail first if a
later version widened it.

The panel is stubbed to a recorder wherever the question is "which pilot
reached the trees", so no assertion depends on what the 45 MB model happens to
think of a synthetic board.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parents[2]
sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
import ml_residual  # noqa: E402
from ml_residual import PANEL, PANEL_NEED, Residual  # noqa: E402

STAMP = mf.UNFAIR_STAMP_ID
PETREL = mf.PETREL_ID
LILLIE = mf.LILLIE_ID
NIGHT_STRETCHER = mf.NIGHT_STRETCHER_ID


class FakeRanker:
    """Returns a per-pilot argmax over a fixed option list."""

    def __init__(self, picks: dict[int, int], options: int = 3):
        self.picks = picks
        self.options = options
        self.scored: list[int] = []

    def _rows(self, observation):
        return [{"slot": i} for i in range(self.options)], list(
            range(self.options)
        )

    def _turn_state(self, observation, features):
        return None

    def _score(self, features, representatives, code):
        self.scored.append(code)
        return self.picks[code], {}


def observation(
    cards: list[int],
    *,
    hand: list[int] | None = None,
    opponent_prizes: int = 5,
    turn: int = 6,
    context: int = mf.CTX_TO_HAND,
    effect_id: int = PETREL,
) -> dict:
    """A Petrel search whose options resolve to ``cards``."""
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "players": [
                {"hand": [{"id": card} for card in (hand or [])]},
                {"prize": [{}] * opponent_prizes},
            ],
        },
        "select": {
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "effect": {"id": effect_id},
            "option": [
                {"index": index, "area": mf.AREA_DECK, "playerIndex": 0}
                for index in range(len(cards))
            ],
        },
        "_cards": cards,
    }


@pytest.fixture(autouse=True)
def _resolve_by_card(monkeypatch: pytest.MonkeyPatch):
    """``resolve_option`` reads the ids the test declared, not the engine's."""

    def fake(current, select, option):
        cards = _resolve_by_card.cards
        index = int(option.get("index", -1))
        card = {"id": cards[index]} if 0 <= index < len(cards) else None
        return card, True, mf.AREA_DECK

    monkeypatch.setattr(mf, "resolve_option", fake)
    yield


def wire(residual: Residual, obs: dict) -> None:
    _resolve_by_card.cards = obs["_cards"]
    residual.note(obs)


def build(picks_by_team: dict[int, int], options: int = 3) -> FakeRanker:
    return FakeRanker(
        {
            member["code"]: picks_by_team[member["team"]]
            for member in PANEL
        },
        options=options,
    )


def fresh(seen_prizes: int = 6, turn: int = 4) -> Residual:
    """A residual that has already watched an earlier turn.

    Without this the "did they take a prize last turn" question falls back to a
    full six, and every test would be measuring the opening turn.
    """
    residual = Residual()
    residual.note({
        "current": {
            "turn": turn, "yourIndex": 0,
            "players": [{}, {"prize": [{}] * seen_prizes}],
        }
    })
    return residual


# ----- the one thing it does -------------------------------------------------


def test_dead_stamp_is_replaced_when_the_panel_agrees() -> None:
    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 1
    assert residual.stats["overrides"] == 1
    assert residual.stats["dead_stamp_chosen"] == 1
    assert sorted(ranker.scored) == sorted(m["code"] for m in PANEL)


def test_a_live_stamp_is_kept() -> None:
    """They took a prize last turn, so the Stamp is playable now."""
    residual = fresh(seen_prizes=6)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["stamp_live_kept"] == 1
    assert residual.stats["overrides"] == 0
    assert ranker.scored == []  # the panel is not even asked


def test_a_split_panel_keeps_v8() -> None:
    picks = {PANEL[0]["team"]: 1, PANEL[1]["team"]: 2,
             PANEL[2]["team"]: 1, PANEL[3]["team"]: 0}
    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)

    assert residual.adjust(obs, obs["select"], 0, build(picks)) == 0
    assert residual.stats["panel_short"] == 1
    assert residual.stats["overrides"] == 0


def test_a_panel_that_agrees_with_v8_changes_nothing() -> None:
    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)
    ranker = build({member["team"]: 0 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["panel_agreed_with_v8"] == 1


def test_a_second_stamp_copy_is_not_a_replacement() -> None:
    """Two interchangeable copies are one decision, not a disagreement."""
    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, STAMP, LILLIE], opponent_prizes=5)
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["panel_alternative_was_stamp"] == 1


def test_a_stamp_already_in_hand_is_excluded() -> None:
    residual = fresh(seen_prizes=5)
    obs = observation(
        [STAMP, LILLIE, NIGHT_STRETCHER], hand=[STAMP], opponent_prizes=5
    )
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["stamp_already_in_hand"] == 1
    assert ranker.scored == []


def test_v8_not_taking_the_stamp_is_left_alone() -> None:
    """It is a veto on one pick, never a preference between the others."""
    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)
    ranker = build({member["team"]: 2 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 1, ranker) == 1
    assert residual.stats["overrides"] == 0
    assert ranker.scored == []


# ----- everything it must not touch ------------------------------------------


@pytest.mark.parametrize(
    "context",
    [
        mf.MAIN_CONTEXT,        # attachments, evolves, attacks, Boss
        mf.CTX_SWITCH,          # the gust target
        mf.CTX_TO_ACTIVE,       # promoting after a knockout
        mf.CTX_DAMAGE,          # the Bench-30 target: prizes this turn
        mf.CTX_REMOVE_DAMAGE_COUNTER,
        mf.CTX_ATTACH_FROM,     # Punk Up allocation
        mf.CTX_REMOVE_COUNTER_COUNT,
        mf.CTX_ACTIVATE,
    ],
)
def test_no_other_context_is_ever_reached(context: int) -> None:
    """The invariants are a context gate, not a threshold.

    Contexts 0, 3, 4, 15, 16, 21, 40 and 43 are where every attachment, evolve,
    attack, knockout and prize decision lives. If a later version widens the
    residual past context 7 this is the test that fails.
    """
    residual = fresh(seen_prizes=5)
    obs = observation(
        [STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5, context=context
    )
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["petrel_searches"] == 0
    assert ranker.scored == []


def test_a_search_that_is_not_petrel_is_left_alone() -> None:
    """Context 7 is also Night Stretcher, Poffin and the Poke Pad look."""
    residual = fresh(seen_prizes=5)
    obs = observation(
        [STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5,
        effect_id=NIGHT_STRETCHER,
    )
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["petrel_searches"] == 0


def test_a_petrel_search_without_a_stamp_is_left_alone() -> None:
    residual = fresh(seen_prizes=5)
    obs = observation([LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL}, options=2)

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["stamp_offered"] == 0


# ----- failure is always v8 --------------------------------------------------


def test_a_ranker_that_raises_falls_back_to_v8() -> None:
    class Broken(FakeRanker):
        def _rows(self, observation):
            raise RuntimeError("features")

    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)

    assert residual.adjust(obs, obs["select"], 0, Broken({})) == 0
    assert residual.stats["errors"] == 1
    assert residual.stats["overrides"] == 0


def test_a_panel_member_that_raises_does_not_silence_the_rest() -> None:
    """Three of four is the rule, so one failed advisor is survivable."""

    class Flaky(FakeRanker):
        def _score(self, features, representatives, code):
            if code == PANEL[0]["code"]:
                raise RuntimeError("scoring")
            return super()._score(features, representatives, code)

    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)
    ranker = Flaky({member["code"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 1
    assert residual.stats["errors"] == 1
    assert residual.stats["overrides"] == 1


def test_two_surviving_advisors_cannot_reach_the_threshold() -> None:
    class VeryFlaky(FakeRanker):
        def _score(self, features, representatives, code):
            if code in (PANEL[0]["code"], PANEL[1]["code"]):
                raise RuntimeError("scoring")
            return super()._score(features, representatives, code)

    residual = fresh(seen_prizes=5)
    obs = observation([STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=5)
    wire(residual, obs)
    ranker = VeryFlaky({member["code"]: 1 for member in PANEL})

    assert residual.adjust(obs, obs["select"], 0, ranker) == 0
    assert residual.stats["panel_short"] == 1


def test_a_new_game_in_a_reused_process_clears_the_prize_history() -> None:
    """Nothing calls ``reset`` between Kaggle episodes.

    Without this the first Petrel search of the next game compares against the
    previous game's prize count, which is the one board where getting live and
    dead the wrong way round throws away a playable Unfair Stamp.
    """
    residual = fresh(seen_prizes=2, turn=12)   # late in a previous game
    obs = observation(
        [STAMP, LILLIE, NIGHT_STRETCHER], opponent_prizes=6, turn=2
    )
    wire(residual, obs)
    ranker = build({member["team"]: 1 for member in PANEL})

    assert residual.stats["new_game_detected"] == 1
    # Six prizes against a fresh history is not a prize taken, so: dead.
    assert residual.adjust(obs, obs["select"], 0, ranker) == 1
    assert residual.stats["stamp_live_kept"] == 0


def test_a_malformed_observation_is_v8() -> None:
    residual = Residual()
    _resolve_by_card.cards = []
    assert residual.adjust({}, {"context": mf.CTX_TO_HAND}, 3, None) == 3


# ----- the wiring that would make it a silent no-op --------------------------


def test_the_panel_codes_are_the_corpus_dense_codes() -> None:
    """LightGBM bins categoricals over the raw range, so a pilot reaches the
    trees as a dense 0..N-1 code. A rebuilt corpus moves every code and the
    panel would silently score four other pilots; this fails instead.
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
    for member in PANEL:
        assert codes[member["team"]] == member["code"]


def test_no_panel_member_is_the_pin_v8_already_uses() -> None:
    """A panel containing v8's own pilot would be voting for itself."""
    assert 16494330 not in {member["team"] for member in PANEL}


def test_the_threshold_is_a_strict_supermajority() -> None:
    assert PANEL_NEED == 3
    assert len(PANEL) == 4
    assert PANEL_NEED > len(PANEL) / 2


def test_metadata_records_the_residual() -> None:
    meta = json.loads(
        (AGENT_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    residual = meta["residual"]
    assert residual["context"] == ml_residual.CTX_TO_HAND
    assert residual["trigger_card_id"] == STAMP
    assert residual["trigger_effect_card_id"] == PETREL
    assert residual["panel_need"] == PANEL_NEED
    assert residual["panel_team_ids"] == [m["team"] for m in PANEL]
    assert residual["panel_teacher_codes"] == [m["code"] for m in PANEL]
    assert meta["deck_hash"] == "9714ab5c3996f6cc"
    assert meta["ranker"]["model_changed"] is False


def test_the_files_v10_claims_are_unchanged_really_are() -> None:
    """The whole safety argument is "v8 plus one module"; this proves it."""
    import hashlib

    v8 = AGENT_DIR.parent / "grimmsnarl_ml_v8"
    if not v8.exists():
        pytest.skip("v8 not present in this checkout")
    meta = json.loads(
        (AGENT_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    for name in meta["files_identical_to_v8"]:
        ours = hashlib.sha256((AGENT_DIR / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((v8 / name).read_bytes()).hexdigest()
        assert ours == theirs, name
