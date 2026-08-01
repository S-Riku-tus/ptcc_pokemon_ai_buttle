"""Ablate the shape of the graded relevance gain, not just its presence.

v33 introduced turn-order graded labels -- chosen 7, played next 3, played
later in the same turn 1, never played 0 -- and v34 re-checked them against
binary labels and kept them. Neither version looked at the gain those label
values map to. LightGBM's default ``label_gain`` is ``2**label - 1``, so the
deployed configuration scores a chosen action at 127, a played-next action at
7, a played-later action at 1 and an unplayed action at 0.

LambdaRank weights a candidate pair by the NDCG it would gain from swapping
them, which is proportional to the difference of those gains. Under the
deployed shape, separating the chosen action from one the teacher plays a
moment later is worth 120, while separating an action the teacher plays later
in the turn from one they never play at all is worth 1. Almost the entire
gradient goes into predicting intra-turn *order*, and almost none into
predicting the *set* of actions the teacher takes.

That matters because the two error types are not equally expensive in play.
Playing a turn's actions in a different order usually converges to the same
board; playing an action the teacher never plays does not. ``turn_set``
already measures the split, and on the v34 holdout it sits at 95.15% against
83.01% strict Top-1, so four fifths of the residual is ordering.

This ablation keeps everything else fixed -- corpus, recency weights, seed,
657 features, 2,200 trees, no stopping rule -- and varies only the gain vector,
scoring every variant on both metrics. Flat positive gains are deliberately
included as a control and are expected to fail: ``end`` belongs to every turn's
action set, so a model indifferent to order ends its turn immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    BASE_CATEGORICAL,
    CONFIGS,
    graded_labels,
    rows_for,
    turn_blocks,
    turn_pick_sets,
    TURN_FEATURES,
)
from scripts.train_alakazam_v34_teacher import recency_multiplier  # noqa: E402
from scripts.experiment_alakazam_v35_residual import (  # noqa: E402
    load_cache,
    residual_report,
)

# Each entry maps the v33 grade (7 chosen, 3 next, 1 later, 0 unplayed) onto a
# label index, and gives the gain vector those indices address.
VARIANTS: dict[str, dict[str, Any]] = {
    "v34_default": {
        "map": {7: 7, 3: 3, 1: 1, 0: 0},
        "gain": [0, 1, 3, 7, 15, 31, 63, 127],
        "note": "deployed: gains 127/7/1/0",
    },
    "compressed": {
        "map": {7: 3, 3: 2, 1: 1, 0: 0},
        "gain": [0, 1, 3, 7],
        "note": "gains 7/3/1/0",
    },
    "linear": {
        "map": {7: 3, 3: 2, 1: 1, 0: 0},
        "gain": [0, 1, 2, 3],
        "note": "gains 3/2/1/0",
    },
    "set_weighted": {
        "map": {7: 3, 3: 2, 1: 1, 0: 0},
        "gain": [0, 4, 5, 7],
        "note": "large step out of the turn set, small steps inside it",
    },
    "set_flat_control": {
        "map": {7: 1, 3: 1, 1: 1, 0: 0},
        "gain": [0, 1],
        "note": "control: order-blind, expected to end turns early",
    },
    "binary": {
        "map": {7: 1, 3: 0, 1: 0, 0: 0},
        "gain": [0, 1],
        "note": "chosen action only",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--model", default="large_leaf")
    parser.add_argument("--n-estimators", type=int, default=2200)
    parser.add_argument("--episode-fraction", type=float, default=0.875)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--tree-step", type=int, default=200)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    cache = load_cache(args.cache)
    features, labels, groups = (
        cache["features"], cache["labels"], cache["groups"]
    )
    names, episode_ids = cache["names"], cache["episode_ids"]
    base_names = [n for n in names if n not in TURN_FEATURES]
    base_features = np.ascontiguousarray(
        features[:, [names.index(n) for n in base_names]]
    )
    categorical = [
        i for i, n in enumerate(base_names)
        if n in BASE_CATEGORICAL or n.endswith("_id")
    ]
    blocks = turn_blocks(features, groups, episode_ids, names)
    graded, _ = graded_labels(features, labels, groups, blocks, names)
    pick_sets, sem_cols = turn_pick_sets(
        features, labels, groups, blocks, names
    )

    decisions = {
        split: np.flatnonzero(cache["splits"] == split)
        for split in ("train", "validation", "test")
    }
    rows = {k: rows_for(groups, v) for k, v in decisions.items()}
    group_sizes = {k: groups[v].astype(int) for k, v in decisions.items()}

    train_pool = episode_ids[decisions["train"]]
    ordered = np.unique(train_pool)
    keep = ordered[-max(1, int(round(len(ordered) * args.episode_fraction))):]
    fit_decisions = decisions["train"][np.isin(train_pool, keep)]
    fit_rows = rows_for(groups, fit_decisions)
    fit_groups = groups[fit_decisions].astype(int)
    fit_weight = cache["weights"][fit_rows] * np.repeat(
        recency_multiplier(
            episode_ids[fit_decisions], args.recency_floor,
            args.recency_power,
        ),
        fit_groups,
    )
    config = CONFIGS[args.model]

    results: dict[str, Any] = {}
    for name in args.variants:
        variant = VARIANTS[name]
        remapped = np.zeros_like(graded)
        for source, target in variant["map"].items():
            remapped[graded == source] = target
        params = dict(
            objective=config["objective"], metric="ndcg",
            num_leaves=config["num_leaves"], n_estimators=args.n_estimators,
            learning_rate=config["learning_rate"],
            min_child_samples=config["min_child_samples"], max_depth=-1,
            subsample=0.9, subsample_freq=1,
            colsample_bytree=config["colsample_bytree"],
            reg_alpha=0.2, reg_lambda=1.0, random_state=args.seed,
            n_jobs=20, verbosity=-1, label_gain=variant["gain"],
        )
        started = time.perf_counter()
        model = lgb.LGBMRanker(**params)
        model.fit(
            X=base_features[fit_rows], y=remapped[fit_rows],
            group=fit_groups, sample_weight=fit_weight,
            feature_name=base_names, categorical_feature=categorical,
        )
        elapsed = time.perf_counter() - started

        curve = []
        for trees in range(
            args.tree_step, args.n_estimators + 1, args.tree_step
        ):
            scores = model.predict(
                base_features[rows["validation"]], num_iteration=trees
            ).astype(np.float32)
            report = residual_report(
                scores, labels[rows["validation"]],
                group_sizes["validation"], decisions["validation"], features,
                rows["validation"], sem_cols, pick_sets, cache["action_types"],
            )
            curve.append({
                "trees": trees,
                "top1": report["overall"]["top1"],
                "turn_set": report["overall"]["turn_set"],
            })
        best = max(curve, key=lambda p: p["top1"])
        held_out = {}
        for split in ("validation", "test"):
            scores = model.predict(
                base_features[rows[split]], num_iteration=best["trees"]
            ).astype(np.float32)
            held_out[split] = residual_report(
                scores, labels[rows[split]], group_sizes[split],
                decisions[split], features, rows[split], sem_cols,
                pick_sets, cache["action_types"],
            )
        results[name] = {
            "note": variant["note"],
            "gain": variant["gain"],
            "label_map": {str(k): v for k, v in variant["map"].items()},
            "fit_seconds": elapsed,
            "selected_trees": best["trees"],
            "validation_curve": curve,
            "validation": held_out["validation"],
            "test": held_out["test"],
        }
        print(json.dumps({name: {
            "trees": best["trees"],
            "val_top1": round(held_out["validation"]["overall"]["top1"], 4),
            "val_turn_set": round(
                held_out["validation"]["overall"]["turn_set"], 4
            ),
            "test_top1": round(held_out["test"]["overall"]["top1"], 4),
            "test_turn_set": round(held_out["test"]["overall"]["turn_set"], 4),
            "test_divergence": round(
                held_out["test"]["overall"]["divergence_rate"], 4
            ),
            "fit_s": round(elapsed, 1),
        }}), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
