"""Measure how much of the v33 generalisation gap is seed variance.

v33 established that the teacher ranker is variance limited, not capacity
limited: training Top-1 is 96.9% against a 78.1% holdout, every training board
key is unique, and no candidate rows are indistinguishable. The textbook fix for
a variance-limited learner is to average decorrelated fits of the *same*
configuration. v33 never tested this: it trained a single seed (1091) and its
greedy blend only searched across different configurations.

This script trains the deployed v33 configuration (large_leaf + turn-order
graded labels) under several seeds and reports:

  per_seed          Each fit's own validation/test metrics, so the spread is
                    visible rather than assumed.
  cumulative        Top-1 of the within-candidate-set z-score average of the
                    first k seeds, k = 1..N. This is exactly how the shipped
                    runtime blends members, so the number transfers.
  seed_oracle       Fraction of test decisions where at least one seed is right.
                    Upper bound on what any seed combination can reach.
  error_taxonomy    Break-down of the residual errors of the k=1 and k=N models,
                    to show which error class averaging actually removes.

Nothing here is selected on the test block; it is reported alongside validation
so the two can be compared, and the v34 configuration decision is made on
validation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    ACTION_TYPES,
    evaluate,
    fit,
    graded_labels,
    normalise,
    ranges,
    rows_for,
    turn_blocks,
    turn_pick_sets,
    TURN_FEATURES,
)


def error_taxonomy(scores, labels, groups, decisions, features, row_index,
                   sem_cols, pick_sets, action_types, names):
    """Classify every wrong Top-1 pick on one split."""
    starts, ends = ranges(np.asarray(groups))
    i_turn = names.index("turn")
    counts: Counter[str] = Counter()
    by_action: dict[str, Counter] = {}
    by_turn_position: Counter[int] = Counter()
    for local, (a, b) in enumerate(zip(starts, ends)):
        block = scores[a:b]
        lab = labels[a:b]
        order = np.argsort(-block, kind="stable")
        counts["decisions"] += 1
        teacher = ACTION_TYPES[int(action_types[decisions[local]])]
        bucket = by_action.setdefault(teacher, Counter())
        bucket["count"] += 1
        if lab[order[0]] == 1:
            counts["correct"] += 1
            bucket["correct"] += 1
            continue
        counts["wrong"] += 1
        picked_row = row_index[a + int(order[0])]
        sem = tuple(features[picked_row, sem_cols].tolist())
        if sem in pick_sets[int(decisions[local])]:
            counts["wrong_intra_turn_reorder"] += 1
            bucket["wrong_reorder"] += 1
        else:
            counts["wrong_action_never_played"] += 1
            bucket["wrong_never"] += 1
        if bool(np.any(lab[order[:3]] == 1)):
            counts["wrong_but_teacher_in_top3"] += 1
        by_turn_position[int(features[picked_row, i_turn])] += 1
    return {
        "totals": dict(counts),
        "by_teacher_action": {
            action: {
                "count": int(stats["count"]),
                "top1": stats["correct"] / stats["count"],
                "wrong_reorder": int(stats["wrong_reorder"]),
                "wrong_never_played": int(stats["wrong_never"]),
            }
            for action, stats in sorted(
                by_action.items(), key=lambda kv: -kv[1]["count"]
            )
        },
        "wrong_by_game_turn": {
            str(turn): int(n)
            for turn, n in sorted(by_turn_position.items())[:25]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[1091, 7, 42, 2024, 31337, 65535],
    )
    parser.add_argument("--n-estimators", type=int, default=2500)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--model", default="large_leaf")
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

    episodes = episode_ids[decisions["train"]]
    ordered = np.unique(episodes)
    position = {
        int(e): i / max(len(ordered) - 1, 1) for i, e in enumerate(ordered)
    }
    floor, power = args.recency_floor, args.recency_power
    multiplier = np.asarray(
        [floor + (1.0 - floor) * position[int(e)] ** power for e in episodes],
        dtype=np.float32,
    )
    train_weight = weights[rows["train"]] * np.repeat(
        multiplier, group_sizes["train"]
    )

    per_seed = {}
    raw_scores: dict[int, dict[str, np.ndarray]] = {}
    for seed in args.seeds:
        model = fit(
            args.model, base_features, base_names, rows["train"],
            graded[rows["train"]], group_sizes["train"], train_weight,
            (rows["validation"], labels[rows["validation"]],
             group_sizes["validation"]),
            seed, args.n_estimators, True,
        )
        best = int(model.best_iteration_ or args.n_estimators)
        scores = {
            split: model.predict(
                base_features[rows[split]], num_iteration=best
            ).astype(np.float32)
            for split in ("validation", "test")
        }
        raw_scores[seed] = scores
        metrics = {
            split: evaluate(
                scores[split], labels[rows[split]], group_sizes[split],
                decisions[split], features, rows[split], sem_cols,
                pick_sets, action_types,
            )
            for split in ("validation", "test")
        }
        per_seed[str(seed)] = {
            "best_iteration": best,
            "validation_top1": metrics["validation"]["top1"],
            "test_top1": metrics["test"]["top1"],
            "test_top3": metrics["test"]["top3"],
            "test_turn_set": metrics["test"]["turn_set"],
        }
        print(json.dumps({str(seed): {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in per_seed[str(seed)].items()
        }}), flush=True)

    spread = {
        split: {
            "min": min(v[f"{split}_top1"] for v in per_seed.values()),
            "max": max(v[f"{split}_top1"] for v in per_seed.values()),
            "mean": float(np.mean(
                [v[f"{split}_top1"] for v in per_seed.values()]
            )),
            "std": float(np.std(
                [v[f"{split}_top1"] for v in per_seed.values()]
            )),
        }
        for split in ("validation", "test")
    }
    spread["validation"]["range_points"] = 100 * (
        spread["validation"]["max"] - spread["validation"]["min"]
    )
    spread["test"]["range_points"] = 100 * (
        spread["test"]["max"] - spread["test"]["min"]
    )

    normalised = {
        seed: {
            split: normalise(raw_scores[seed][split], group_sizes[split])
            for split in ("validation", "test")
        }
        for seed in args.seeds
    }

    cumulative = {}
    ensemble_scores: dict[str, np.ndarray] = {}
    for k in range(1, len(args.seeds) + 1):
        used = args.seeds[:k]
        totals = {
            split: sum(normalised[s][split] for s in used) / k
            for split in ("validation", "test")
        }
        metrics = {
            split: evaluate(
                totals[split].astype(np.float32), labels[rows[split]],
                group_sizes[split], decisions[split], features, rows[split],
                sem_cols, pick_sets, action_types,
            )
            for split in ("validation", "test")
        }
        cumulative[str(k)] = {
            "seeds": used,
            "validation_top1": metrics["validation"]["top1"],
            "test_top1": metrics["test"]["top1"],
            "test_top2": metrics["test"]["top2"],
            "test_top3": metrics["test"]["top3"],
            "test_turn_set": metrics["test"]["turn_set"],
        }
        print(json.dumps({f"k={k}": {
            "val_top1": round(metrics["validation"]["top1"], 4),
            "test_top1": round(metrics["test"]["top1"], 4),
            "test_turn_set": round(metrics["test"]["turn_set"], 4),
        }}), flush=True)
        if k == len(args.seeds):
            ensemble_scores = totals

    # Upper bound: a decision any single seed already gets right.
    starts, ends = ranges(np.asarray(group_sizes["test"]))
    test_labels = labels[rows["test"]]
    hit = np.zeros(len(group_sizes["test"]), dtype=bool)
    for seed in args.seeds:
        s = raw_scores[seed]["test"]
        for local, (a, b) in enumerate(zip(starts, ends)):
            if test_labels[a + int(np.argmax(s[a:b]))] == 1:
                hit[local] = True
    seed_oracle = float(hit.mean())

    taxonomy = {
        "single_seed_%d" % args.seeds[0]: error_taxonomy(
            raw_scores[args.seeds[0]]["test"], test_labels,
            group_sizes["test"], decisions["test"], features, rows["test"],
            sem_cols, pick_sets, action_types, names,
        ),
        "ensemble_k%d" % len(args.seeds): error_taxonomy(
            ensemble_scores["test"].astype(np.float32), test_labels,
            group_sizes["test"], decisions["test"], features, rows["test"],
            sem_cols, pick_sets, action_types, names,
        ),
    }

    report = {
        "cache": str(args.cache.resolve()),
        "model": args.model,
        "label": "turn_order_graded_relevance",
        "seeds": args.seeds,
        "per_seed": per_seed,
        "seed_spread": spread,
        "cumulative_ensemble": cumulative,
        "seed_oracle_test_top1": seed_oracle,
        "error_taxonomy": taxonomy,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
