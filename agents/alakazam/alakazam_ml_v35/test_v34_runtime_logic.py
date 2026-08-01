"""v34 regressions: tree budget, corpus recency, and single-member deployment.

v34 keeps the whole v31 safety shell and the v33 label design. What changed is
the corpus (submission 54773249 refetched for 980 more games, oldest eighth
dropped) and how the deployed tree count is chosen (validation Top-1 rather
than LightGBM's NDCG early stopping). These tests pin both, plus the export
invariant that the earlier build violated silently: metadata written next to
the walked trees must not overwrite them.
"""

from __future__ import annotations

import json
from pathlib import Path

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from ml_runtime import HybridRanker  # noqa: E402


HERE = Path(__file__).resolve().parent


def _model():
    return json.loads((HERE / "ranker_model.json").read_text("utf-8"))


def test_v34_ships_the_validation_selected_tree_budget():
    model = _model()

    assert model["tree_count"] == 2050
    assert model["tree_count_selected_by"] == "validation_top1"
    # The booster itself has to carry that many trees. A previous build wrote
    # the integer over compact_booster's tree list and shipped a 17 KB model
    # that loaded without error, so assert the walked trees directly.
    assert isinstance(model["trees"], list)
    assert len(model["trees"]) == model["tree_count"]


def test_v34_keeps_only_the_recent_share_of_the_refetched_corpus():
    model = _model()

    assert model["episode_fraction_kept"] == 0.875
    assert model["teacher_trajectories"] == 1981
    assert sum(model["teacher_cohorts"].values()) == 2269
    # The refetch cohort has to be present, otherwise this is a v33 corpus.
    assert model["teacher_cohorts"]["yushin_20260801"] == 980


def test_v34_deploys_exactly_one_member_without_turn_features():
    runtime = HybridRanker()
    snapshot = runtime.snapshot()

    assert snapshot["model_loaded"]
    assert snapshot["ensemble_size"] == 1
    assert snapshot["ensemble_weights"] == [1.0]
    # v33 generalised the runtime to a validation-gated blend and then gated
    # the second member out. v34 selects a single model, so the intra-turn
    # tracker must stay inert; leaving a stale member file behind would
    # silently re-enable it.
    assert snapshot["uses_turn_features"] is False
    for index in (1, 2, 3):
        assert not (HERE / f"ranker_model_{index}.json").exists()


def test_v34_preserves_the_v31_safety_shell():
    runtime = HybridRanker()
    snapshot = runtime.snapshot()

    assert snapshot["v29_model_loaded"]
    assert snapshot["legacy_model_loaded"]
    assert snapshot["memory_loaded"]
    assert snapshot["runtime_scope"] == (
        "v35_narrowed_shell_plus_recent_corpus_ranker"
    )
    assert not snapshot["errors"]
