"""v33 runtime regressions: model provenance, blending and intra-turn state."""

from __future__ import annotations

import json
from pathlib import Path

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from ml_runtime import (  # noqa: E402
    TURN_FEATURE_NAMES,
    HybridRanker,
    _TurnHistory,
)


HERE = Path(__file__).resolve().parent


def _keys(*specs):
    """(semantic, class) pairs shaped like ``_turn_option_keys`` output."""
    return [
        ((option_type, card_id, -1, -1, -1), (action_type, card_id))
        for option_type, card_id, action_type in specs
    ]


def test_v34_model_is_the_recent_corpus_yushin_ranker():
    model = json.loads((HERE / "ranker_model.json").read_text("utf-8"))

    assert model["runtime_scope"] == "v34_yushin_recent_corpus_ranker"
    # v33's turn-order graded relevance survived a like-for-like retest on the
    # enlarged corpus, so v34 keeps the label and the recency weight and
    # changes only which episodes are kept and how many trees are deployed.
    assert model["label_definition"] == "turn_order_graded_relevance"
    assert model["teacher_team"] == "Yushin Ito"
    assert model["teacher_submission_id"] == 54773249
    # v33 trained on 1,284 usable episodes; v34 refetched the same submission
    # for 980 more and then dropped the oldest eighth of the enlarged corpus.
    assert model["teacher_trajectories"] == 1981
    assert model["episode_fraction_kept"] == 0.875
    assert model["training_recency_weight"]["floor"] == 0.25
    assert model["training_recency_weight"]["power"] == 2.0
    assert len(model["feature_names"]) > 600
    assert "v29_ranker_score" in model["feature_names"]


def test_v33_deploys_a_single_gated_ensemble_member():
    runtime = HybridRanker()
    snapshot = runtime.snapshot()

    assert snapshot["model_loaded"]
    assert snapshot["ensemble_size"] == len(runtime.ensemble)
    assert snapshot["ensemble_weights"][0] == 1.0
    # A further member is only shipped when it clears the validation gate, so
    # the intra-turn tracker stays inert unless a deployed model needs it.
    assert snapshot["uses_turn_features"] == runtime.uses_turn_features
    for name in ("ranker_model_1.json", "ranker_model_2.json"):
        if not (HERE / name).exists():
            assert f"{name}" not in snapshot["errors"]


def test_turn_history_counts_offers_and_pass_overs():
    history = _TurnHistory()
    current = {"turn": 4, "yourIndex": 0}
    keys = _keys((8, 1225, 10), (7, 1079, 10), (9, 742, 6))

    first = history.columns(current, keys)
    assert [row["turn_candidate_offer_count"] for row in first] == [0, 0, 0]
    assert [row["turn_new_candidate"] for row in first] == [1, 1, 1]
    assert all(row["turn_decision_index"] == 0 for row in first)

    history.record(current, keys, 0)
    second = history.columns(current, keys)
    assert [row["turn_candidate_offer_count"] for row in second] == [1, 1, 1]
    # Only the two candidates that were declined count as passed over.
    assert [row["turn_candidate_passed_over"] for row in second] == [0, 1, 1]
    assert all(row["turn_candidate_offered_previous"] == 1 for row in second)
    assert all(row["turn_decision_index"] == 1 for row in second)


def test_turn_history_counts_interchangeable_copies_once():
    history = _TurnHistory()
    current = {"turn": 2, "yourIndex": 1}
    duplicated = _keys((8, 1225, 10), (8, 1225, 10), (7, 1079, 10))

    history.record(current, duplicated, 2)
    columns = history.columns(current, duplicated)

    # Both copies are the same semantic candidate, so a single decision may
    # only advance its counters once.
    assert columns[0]["turn_candidate_offer_count"] == 1
    assert columns[0]["turn_candidate_passed_over"] == 1
    assert columns[2]["turn_candidate_passed_over"] == 0


def test_turn_history_resets_on_a_new_turn():
    history = _TurnHistory()
    keys = _keys((8, 1225, 10))
    history.record({"turn": 3, "yourIndex": 0}, keys, 0)

    later = history.columns({"turn": 5, "yourIndex": 0}, keys)

    assert later[0]["turn_candidate_offer_count"] == 0
    assert later[0]["turn_decision_index"] == 0
    assert later[0]["turn_new_candidate"] == 1


def test_note_decision_ignores_decisions_outside_the_learned_scope():
    runtime = HybridRanker()
    runtime.uses_turn_features = True
    runtime.turn_history.reset()
    nested = {
        "current": {"turn": 1, "yourIndex": 0},
        "select": {
            "type": 0, "context": 3, "minCount": 1, "maxCount": 1,
            "option": [{"type": 8}, {"type": 8}],
        },
    }

    runtime.note_decision(nested, [0])

    assert runtime.turn_history.position == 0
    assert not runtime.turn_history.candidates


def test_turn_feature_names_match_the_trained_columns():
    model = json.loads((HERE / "ranker_model.json").read_text("utf-8"))
    names = set(model["feature_names"])

    if model.get("uses_turn_features"):
        assert set(TURN_FEATURE_NAMES) <= names
    else:
        assert not set(TURN_FEATURE_NAMES) & names
