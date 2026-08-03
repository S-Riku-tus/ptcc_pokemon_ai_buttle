"""Refit one validation-selected Alakazam configuration before final test.

The configuration and tree count must already have been selected on the
chronological validation block.  This script then uses train+validation for a
single final fit and evaluates the still untouched chronological test block.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_alakazam_v35_residual import load_cache  # noqa: E402
from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    evaluate,
    graded_labels,
    rows_for,
    turn_blocks,
    turn_pick_sets,
    TURN_FEATURES,
)
from scripts.train_alakazam_v34_teacher import (  # noqa: E402
    fit_fixed,
    recency_multiplier,
)


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return centre - radius, centre + radius


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="large_leaf")
    parser.add_argument("--trees", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--label", choices=("graded", "binary"), default="graded")
    parser.add_argument("--episode-fraction", type=float, default=1.0)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--recent-min-episode", type=int, default=0)
    parser.add_argument("--recent-boost", type=float, default=1.0)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=0)
    parser.add_argument("--include-turn-features", action="store_true")
    args = parser.parse_args()

    cache = load_cache(args.cache)
    features = cache["features"]
    labels = cache["labels"]
    groups = cache["groups"]
    names = cache["names"]
    if args.include_turn_features:
        matrix = features
        columns = names
    else:
        columns = [name for name in names if name not in TURN_FEATURES]
        matrix = np.ascontiguousarray(
            features[:, [names.index(name) for name in columns]]
        )

    blocks = turn_blocks(
        features, groups, cache["episode_ids"], names
    )
    graded, graded_counts = graded_labels(
        features, labels, groups, blocks, names
    )
    pick_sets, semantic_columns = turn_pick_sets(
        features, labels, groups, blocks, names
    )
    targets = graded if args.label == "graded" else labels

    train = np.flatnonzero(cache["splits"] == "train")
    validation = np.flatnonzero(cache["splits"] == "validation")
    test = np.flatnonzero(cache["splits"] == "test")
    refit_pool = np.concatenate([train, validation])
    ordered_episodes = np.unique(cache["episode_ids"][refit_pool])
    keep_count = max(1, int(round(
        len(ordered_episodes) * args.episode_fraction
    )))
    kept_episodes = ordered_episodes[-keep_count:]
    fit_decisions = refit_pool[np.isin(
        cache["episode_ids"][refit_pool], kept_episodes
    )]
    fit_rows = rows_for(groups, fit_decisions)
    fit_groups = groups[fit_decisions].astype(np.int64)
    fit_weight = cache["weights"][fit_rows] * np.repeat(
        recency_multiplier(
            cache["episode_ids"][fit_decisions],
            args.recency_floor,
            args.recency_power,
        ),
        fit_groups,
    )
    if args.recent_min_episode:
        fit_weight *= np.repeat(
            np.where(
                cache["episode_ids"][fit_decisions]
                >= args.recent_min_episode,
                args.recent_boost,
                1.0,
            ).astype(np.float32),
            fit_groups,
        )

    started = time.perf_counter()
    model = fit_fixed(
        args.model,
        matrix,
        columns,
        fit_rows,
        targets[fit_rows],
        fit_groups,
        fit_weight,
        args.seed,
        args.trees,
        args.label == "graded",
        args.lambdarank_truncation_level,
    )
    fit_seconds = time.perf_counter() - started

    test_rows = rows_for(groups, test)
    test_groups = groups[test].astype(np.int64)
    scores = model.predict(matrix[test_rows]).astype(np.float32)
    metrics = evaluate(
        scores,
        labels[test_rows],
        test_groups,
        test,
        features,
        test_rows,
        semantic_columns,
        pick_sets,
        cache["action_types"],
    )
    correct = int(round(metrics["top1"] * metrics["decisions"]))
    lower, upper = wilson_interval(correct, metrics["decisions"])
    report = {
        "method": "validation-selected final refit on train+validation",
        "cache": str(args.cache.resolve()),
        "selection_provenance": (
            "configuration and tree count selected before this run on the "
            "chronological validation split"
        ),
        "configuration": {
            "model": args.model,
            "trees": args.trees,
            "seed": args.seed,
            "label": args.label,
            "episode_fraction": args.episode_fraction,
            "recency_floor": args.recency_floor,
            "recency_power": args.recency_power,
            "recent_min_episode": args.recent_min_episode,
            "recent_boost": args.recent_boost,
            "lambdarank_truncation_level": (
                args.lambdarank_truncation_level
            ),
            "include_turn_features": args.include_turn_features,
        },
        "fit_episodes": int(len(kept_episodes)),
        "fit_decisions": int(len(fit_decisions)),
        "fit_seconds": fit_seconds,
        "graded_label_counts": graded_counts,
        "test": {
            **metrics,
            "top1_correct": correct,
            "top1_wilson_95": [lower, upper],
        },
        "target_top1": 0.90,
        "target_met": bool(metrics["top1"] > 0.90),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "fit_episodes": report["fit_episodes"],
        "fit_decisions": report["fit_decisions"],
        "fit_seconds": fit_seconds,
        "test": report["test"],
        "target_met": report["target_met"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
