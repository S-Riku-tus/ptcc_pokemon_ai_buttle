"""Ablate what the enlarged same-teacher corpus buys, on one honest holdout.

v33 measured the same-teacher data scaling curve at +8.47 Top-1 points per
decade and extrapolated that ~16,400 games would be needed for 90%. It had
1,288. Refetching submission 54773249 recovered 980 further games, so v34 has
2,268 and the extrapolation can be checked instead of trusted.

Every new game is chronologically *newer* than the v33 frozen holdout, so that
holdout cannot be reused: training on it would mean training on games that
postdate the test block. This script therefore runs on a re-cut, temporally
honest split (oldest games train, newest games validate and test) and measures:

  data_scaling      Top-1 as a function of how many of the most recent training
                    episodes are used, all else fixed. Confirms or refutes the
                    v33 scaling law on a holdout neither version has seen.
  recency           The v33 recency weight (floor 0.25, power 2.0) was tuned
                    when the corpus spanned a shorter period. The teacher
                    submission is fixed, so the policy is constant and only the
                    opponent distribution drifts; whether down-weighting old
                    games still pays is an open question at this corpus size.
  label             The v33 graded relevance (chosen 7 / next 3 / later 1 /
                    unplayed 0) against binary, to confirm the v33 finding
                    survives the larger corpus.

Selection is on validation. The test block is reported alongside but is not
used to choose the v34 configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    evaluate,
    fit,
    graded_labels,
    rows_for,
    turn_blocks,
    turn_pick_sets,
    TURN_FEATURES,
)


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
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--model", default="large_leaf")
    parser.add_argument(
        "--episode-fractions", type=float, nargs="*",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Share of the most recent training episodes to fit on.",
    )
    parser.add_argument(
        "--recency", nargs="*", default=["0.25:2.0", "0.5:2.0", "1.0:1.0"],
        help="floor:power pairs, swept at --sweep-fraction.",
    )
    parser.add_argument(
        "--labels", nargs="*", default=["graded", "binary"],
        help="Label schemes, swept at --sweep-fraction.",
    )
    parser.add_argument(
        "--sweep-fraction", type=float, default=1.0,
        help=(
            "Training-episode share the recency and label sweeps run at. "
            "Truncation is selected first, so later sweeps must be measured "
            "on the corpus that will actually ship, not on the full one."
        ),
    )
    parser.add_argument(
        "--run-v33-baseline", action="store_true",
        help="Also fit the v33 training set for a same-holdout reference.",
    )
    parser.add_argument(
        "--baseline-episode-max", type=int, default=88002730,
        help=(
            "Reproduce the v33 training set by keeping only training episodes "
            "below this ID. v33 trained on everything below its own frozen "
            "validation boundary, so this is the honest 'what v33 could have "
            "learned' baseline, scored on the same new holdout as v34."
        ),
    )
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
    print(
        f"train_episodes={len(train_episodes)} "
        f"train_decisions={len(decisions['train'])} "
        f"validation_decisions={len(decisions['validation'])} "
        f"test_decisions={len(decisions['test'])}",
        flush=True,
    )

    def run(tag, episode_subset, floor, power, label_scheme):
        keep = np.isin(episode_ids[decisions["train"]], episode_subset)
        subset_decisions = decisions["train"][keep]
        subset_rows = rows_for(groups, subset_decisions)
        subset_groups = groups[subset_decisions].astype(int)
        subset_episodes = episode_ids[subset_decisions]
        multiplier = recency_multiplier(subset_episodes, floor, power)
        weight = weights[subset_rows] * np.repeat(multiplier, subset_groups)
        use_graded = label_scheme == "graded"
        y = (graded if use_graded else labels)[subset_rows]
        model = fit(
            args.model, base_features, base_names, subset_rows, y,
            subset_groups, weight,
            (rows["validation"], labels[rows["validation"]],
             group_sizes["validation"]),
            args.seed, args.n_estimators, use_graded,
        )
        best = int(model.best_iteration_ or args.n_estimators)
        metrics = {}
        for split in ("validation", "test"):
            scores = model.predict(
                base_features[rows[split]], num_iteration=best
            ).astype(np.float32)
            metrics[split] = evaluate(
                scores, labels[rows[split]], group_sizes[split],
                decisions[split], features, rows[split], sem_cols,
                pick_sets, action_types,
            )
        entry = {
            "train_episodes": int(len(episode_subset)),
            "train_decisions": int(len(subset_decisions)),
            "recency_floor": floor,
            "recency_power": power,
            "label": label_scheme,
            "best_iteration": best,
            "validation_top1": metrics["validation"]["top1"],
            "test_top1": metrics["test"]["top1"],
            "test_top2": metrics["test"]["top2"],
            "test_top3": metrics["test"]["top3"],
            "test_turn_set": metrics["test"]["turn_set"],
            "test_by_teacher_action": metrics["test"]["by_teacher_action"],
        }
        print(json.dumps({tag: {
            "episodes": entry["train_episodes"],
            "iter": best,
            "val_top1": round(entry["validation_top1"], 4),
            "test_top1": round(entry["test_top1"], 4),
            "test_turn_set": round(entry["test_turn_set"], 4),
        }}), flush=True)
        return entry

    report: dict[str, dict] = {
        "cache": str(args.cache.resolve()),
        "model": args.model,
        "seed": args.seed,
        "train_episodes_available": int(len(train_episodes)),
        "split_decisions": {k: int(len(v)) for k, v in decisions.items()},
        "data_scaling": {},
        "recency": {},
        "label": {},
        "v33_baseline": {},
    }

    def most_recent(fraction):
        count = max(1, int(round(len(train_episodes) * fraction)))
        return train_episodes[-count:]

    if args.run_v33_baseline:
        # What v33 could have learned: its own training episodes, this holdout.
        report["v33_baseline"][str(args.baseline_episode_max)] = run(
            "v33_baseline",
            train_episodes[train_episodes < args.baseline_episode_max],
            0.25, 2.0, "graded",
        )

    for fraction in args.episode_fractions:
        report["data_scaling"][f"{fraction:g}"] = run(
            f"scale_{fraction:g}", most_recent(fraction), 0.25, 2.0, "graded"
        )

    swept = most_recent(args.sweep_fraction)
    report["sweep_fraction"] = args.sweep_fraction
    for spec in args.recency:
        floor, power = (float(x) for x in spec.split(":"))
        report["recency"][spec] = run(
            f"recency_{spec}", swept, floor, power, "graded"
        )

    for scheme in args.labels:
        report["label"][scheme] = run(
            f"label_{scheme}", swept, 0.25, 2.0, scheme
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
