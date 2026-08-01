"""Separate a real corpus-truncation effect from early-stopping noise.

The v34 truncation sweep produced a non-monotone curve, and the ranking of its
runs tracked ``best_iteration`` almost exactly: every run that early-stopped
below 700 trees scored badly and every run that reached 880+ scored well. That
is the signature of an unstable stopping rule, not of a data effect.

The v33 pipeline stops on LightGBM's ``ndcg`` over the graded labels, but the
deployed and reported metric is strict Top-1 agreement. Those are different
objectives, so the tree count that maximises one is not the one that maximises
the other, and the gap between them is resampled on every fit.

This script removes the stopping rule. Each corpus is fitted once to a fixed
large tree budget, then scored at many prefixes of that same booster, so the
whole accuracy-versus-trees curve is observed instead of one noisy point on it.
With the iteration count controlled, any remaining separation between corpora
is attributable to the corpus.

Selection stays on validation; test is reported alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    BASE_CATEGORICAL,
    CONFIGS,
    LABEL_GAIN,
    evaluate,
    graded_labels,
    rows_for,
    turn_blocks,
    turn_pick_sets,
    TURN_FEATURES,
)


def fit_no_stopping(name, matrix, cols, x_rows, y, group, weight, seed,
                    n_estimators, graded):
    """Same configuration as the v33 trainer, minus the stopping callback."""
    config = CONFIGS[name]
    params: dict[str, Any] = dict(
        objective=config["objective"], metric="ndcg",
        num_leaves=config["num_leaves"], n_estimators=n_estimators,
        learning_rate=config["learning_rate"],
        min_child_samples=config["min_child_samples"], max_depth=-1,
        subsample=0.9, subsample_freq=1,
        colsample_bytree=config["colsample_bytree"],
        reg_alpha=0.2, reg_lambda=1.0, random_state=seed,
        n_jobs=20, verbosity=-1,
    )
    if graded:
        params["label_gain"] = LABEL_GAIN
    model = lgb.LGBMRanker(**params)
    kwargs: dict[str, Any] = {
        "X": matrix[x_rows], "y": y, "group": group, "sample_weight": weight,
        "feature_name": cols,
    }
    if config["categorical_ids"]:
        kwargs["categorical_feature"] = [
            i for i, n in enumerate(cols)
            if n in BASE_CATEGORICAL or n.endswith("_id")
        ]
    model.fit(**kwargs)
    return model


def recency_multiplier(episodes, floor, power):
    ordered = np.unique(episodes)
    position = {
        int(e): i / max(len(ordered) - 1, 1) for i, e in enumerate(ordered)
    }
    return np.asarray(
        [floor + (1.0 - floor) * position[int(e)] ** power for e in episodes],
        dtype=np.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1091])
    parser.add_argument("--n-estimators", type=int, default=2200)
    parser.add_argument("--model", default="large_leaf")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.75, 0.875, 1.0],
    )
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--label", default="graded")
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        weights = cached["weights"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        action_types = cached["teacher_action_types"]
        names = cached["feature_names"].astype(str).tolist()

    base_names = [n for n in names if n not in TURN_FEATURES]
    base_columns = [names.index(n) for n in base_names]
    base_features = np.ascontiguousarray(features[:, base_columns])

    blocks = turn_blocks(features, groups, episode_ids, names)
    graded, _ = graded_labels(features, labels, groups, blocks, names)
    pick_sets, sem_cols = turn_pick_sets(
        features, labels, groups, blocks, names
    )

    decisions = {
        split: np.flatnonzero(splits == split)
        for split in ("train", "validation", "test")
    }
    rows = {k: rows_for(groups, v) for k, v in decisions.items()}
    group_sizes = {k: groups[v].astype(int) for k, v in decisions.items()}
    train_episodes = np.unique(episode_ids[decisions["train"]])

    grid = list(range(args.step, args.n_estimators + 1, args.step))
    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()),
        "model": args.model,
        "label": args.label,
        "recency": {"floor": args.recency_floor, "power": args.recency_power},
        "n_estimators": args.n_estimators,
        "iteration_grid": grid,
        "curves": {},
    }

    for fraction in args.fractions:
        count = max(1, int(round(len(train_episodes) * fraction)))
        subset = train_episodes[-count:]
        keep = np.isin(episode_ids[decisions["train"]], subset)
        subset_decisions = decisions["train"][keep]
        subset_rows = rows_for(groups, subset_decisions)
        subset_groups = groups[subset_decisions].astype(int)
        multiplier = recency_multiplier(
            episode_ids[subset_decisions], args.recency_floor,
            args.recency_power,
        )
        weight = weights[subset_rows] * np.repeat(multiplier, subset_groups)
        use_graded = args.label == "graded"
        y = (graded if use_graded else labels)[subset_rows]

        for seed in args.seeds:
            model = fit_no_stopping(
                args.model, base_features, base_names, subset_rows, y,
                subset_groups, weight, seed, args.n_estimators, use_graded,
            )
            curve = []
            for trees in grid:
                point = {"trees": trees}
                for split in ("validation", "test"):
                    scores = model.predict(
                        base_features[rows[split]], num_iteration=trees
                    ).astype(np.float32)
                    metrics = evaluate(
                        scores, labels[rows[split]], group_sizes[split],
                        decisions[split], features, rows[split], sem_cols,
                        pick_sets, action_types,
                    )
                    point[f"{split}_top1"] = metrics["top1"]
                    if split == "test":
                        point["test_top3"] = metrics["top3"]
                        point["test_turn_set"] = metrics["turn_set"]
                curve.append(point)
            best = max(curve, key=lambda p: p["validation_top1"])
            tag = f"{fraction:g}_s{seed}"
            report["curves"][tag] = {
                "fraction": fraction,
                "seed": seed,
                "train_episodes": int(count),
                "train_decisions": int(len(subset_decisions)),
                "curve": curve,
                "best_by_validation": best,
                "validation_top1_at_2200": curve[-1]["validation_top1"],
                "test_top1_at_2200": curve[-1]["test_top1"],
            }
            print(json.dumps({tag: {
                "episodes": int(count),
                "best_trees": best["trees"],
                "val_top1": round(best["validation_top1"], 4),
                "test_top1": round(best["test_top1"], 4),
                "val_at_2200": round(curve[-1]["validation_top1"], 4),
                "test_at_2200": round(curve[-1]["test_top1"], 4),
            }}), flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
