"""v33 data-scaling curve for the primary-teacher corpus.

The v33 diagnosis is that the ranker is variance limited: it reaches 96.9%
Top-1 on its own training decisions and 78.1% on the frozen chronological
holdout, and no training state ever recurs. This script measures how holdout
Top-1 responds to the number of primary-teacher episodes so that the remaining
distance to the 90% target can be expressed as a data requirement instead of a
guess. The holdout is identical for every point on the curve.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

BASE_CATEGORICAL = {
    "action_type", "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "self_active_id", "opp_active_id", "stadium_id",
    "fallback_action_type", "fallback_card_id",
}


def ranges(groups):
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def rows_for(groups, decisions):
    starts, ends = ranges(groups)
    return np.concatenate([
        np.arange(starts[d], ends[d], dtype=np.int64) for d in decisions
    ])


def topk(scores, labels, groups):
    starts, ends = ranges(groups)
    t1 = t2 = t3 = 0
    for a, b in zip(starts, ends):
        order = np.argsort(-scores[a:b], kind="stable")
        lab = labels[a:b]
        t1 += int(lab[order[0]] == 1)
        t2 += int(bool(np.any(lab[order[:2]] == 1)))
        t3 += int(bool(np.any(lab[order[:3]] == 1)))
    n = len(groups)
    return {
        "top1": t1 / n, "top2": t2 / n, "top3": t3 / n, "decisions": int(n),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=[0.125, 0.25, 0.5, 0.75, 1.0],
    )
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        weights = cached["weights"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        names = cached["feature_names"].astype(str).tolist()

    train = np.flatnonzero(splits == "train")
    valid = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    categorical = [
        i for i, n in enumerate(names)
        if n in BASE_CATEGORICAL or n.endswith("_id")
    ]
    va_rows, te_rows = rows_for(groups, valid), rows_for(groups, test)
    gva = groups[valid].astype(int)
    gte = groups[test].astype(int)

    train_episodes = np.unique(episode_ids[train])
    points = []
    for fraction in args.fractions:
        keep = train_episodes[
            max(0, int(len(train_episodes) * (1.0 - fraction))):
        ]
        keep_set = set(keep.tolist())
        subset = train[np.asarray([
            int(episode_ids[d]) in keep_set for d in train
        ])]
        rows = rows_for(groups, subset)
        gsub = groups[subset].astype(int)

        episodes = episode_ids[subset]
        ordered = np.unique(episodes)
        pos = {
            int(e): i / max(len(ordered) - 1, 1)
            for i, e in enumerate(ordered)
        }
        mult = np.asarray(
            [0.25 + 0.75 * pos[int(e)] ** 2.0 for e in episodes],
            dtype=np.float32,
        )
        weight = weights[rows] * np.repeat(mult, gsub)

        model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            num_leaves=args.num_leaves, n_estimators=args.n_estimators,
            learning_rate=0.03, min_child_samples=40, max_depth=-1,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.88,
            reg_alpha=0.2, reg_lambda=1.0, random_state=args.seed,
            n_jobs=20, verbosity=-1,
        )
        model.fit(
            features[rows], labels[rows], group=gsub, sample_weight=weight,
            feature_name=names, categorical_feature=categorical,
            eval_set=[(features[va_rows], labels[va_rows])], eval_group=[gva],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        entry = {
            "fraction": fraction,
            "episodes": int(len(keep)),
            "decisions": int(len(subset)),
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "train": topk(
                model.predict(features[rows]).astype(np.float32),
                labels[rows], gsub,
            ),
            "validation": topk(
                model.predict(features[va_rows]).astype(np.float32),
                labels[va_rows], gva,
            ),
            "test": topk(
                model.predict(features[te_rows]).astype(np.float32),
                labels[te_rows], gte,
            ),
        }
        points.append(entry)
        print(json.dumps({
            "episodes": entry["episodes"],
            "decisions": entry["decisions"],
            "train_top1": round(entry["train"]["top1"], 4),
            "test_top1": round(entry["test"]["top1"], 4),
        }), flush=True)

    # Fit test_top1 = a + b * log10(episodes) to express the 90% target as a
    # data requirement. This is an extrapolation, reported as such.
    x = np.log10([p["episodes"] for p in points])
    y = np.asarray([p["test"]["top1"] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    needed = (
        10 ** ((0.90 - intercept) / slope) if slope > 0 else float("inf")
    )
    projection = {
        "log10_slope_per_decade": float(slope),
        "intercept": float(intercept),
        "episodes_needed_for_top1_0.90": (
            float(needed) if math.isfinite(needed) else None
        ),
        "caveat": (
            "Log-linear extrapolation from five points; it estimates the "
            "order of magnitude of missing same-teacher games, not an exact "
            "requirement."
        ),
    }
    print(json.dumps(projection, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"cache": str(args.cache), "points": points,
             "projection": projection},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
