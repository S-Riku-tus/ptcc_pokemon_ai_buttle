"""Fit the v34 configuration once and take the residual error apart.

v34 closed on 83.01% strict Top-1 and left three observations without a
measurement behind them: ``trainer`` is the largest absolute loss, ``boss`` has
the lowest agreement of any class, and 86% of the errors keep the teacher's
action inside the model's Top-3. None of those say how much of the residual is
*recoverable*. The v33/v34 ``turn_set`` metric already distinguishes the two
failure modes at the aggregate level -- picking an action the teacher also
plays later in the same turn costs nothing, picking one the teacher never plays
is a real divergence -- but it was never broken down by class, so there is no
evidence about which classes are worth spending capacity on.

This script fits the deployed configuration to a fixed tree budget, caches the
raw candidate scores for both holdout blocks, and reports the residual split
into ordering errors and divergences per action class. The cached scores are
what the stage-2 cascade experiment consumes, so the expensive fit happens
once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    ACTION_TYPE_MAP,
    ACTION_TYPES,
    graded_labels,
    ranges,
    rows_for,
    turn_blocks,
    turn_pick_sets,
    TURN_FEATURES,
)
from scripts.train_alakazam_v34_teacher import (  # noqa: E402
    fit_fixed,
    recency_multiplier,
)


def load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as cached:
        return {
            "features": cached["features"],
            "labels": cached["labels"],
            "weights": cached["weights"],
            "groups": cached["groups"],
            "splits": cached["splits"].astype(str),
            "episode_ids": cached["episode_ids"],
            "action_types": cached["teacher_action_types"],
            "fallback_correct": cached["fallback_correct"],
            "names": cached["feature_names"].astype(str).tolist(),
        }


TERMINAL_ACTIONS = frozenset(
    {ACTION_TYPE_MAP["attack"], ACTION_TYPE_MAP["end"]}
)


def residual_report(scores, labels, group_sizes, decisions, features,
                    row_index, sem_cols, pick_sets, action_types,
                    action_column=None):
    """Strict agreement plus the error taxonomy, per teacher action class.

    Three error types, in increasing cost. An *ordering* error picks an action
    the teacher also plays in this turn, so the turn still converges on the
    same board. A *premature* error is an ordering error on an action that
    ends the turn -- an attack or an explicit end -- which skips every
    remaining action the teacher took, so it is not recoverable. A
    *divergence* picks an action the teacher never plays in that turn at all.
    """
    starts, ends = ranges(np.asarray(group_sizes))
    totals: Counter[str] = Counter()
    by_class: dict[int, Counter] = {}
    teacher_rank_hist: Counter[int] = Counter()
    for local, (a, b) in enumerate(zip(starts, ends)):
        block = scores[a:b]
        lab = labels[a:b]
        order = np.argsort(-block, kind="stable")
        correct = bool(lab[order[0]] == 1)
        picked_row = row_index[a + int(order[0])]
        sem = tuple(features[picked_row, sem_cols].tolist())
        in_turn = not correct and sem in pick_sets[int(decisions[local])]
        premature = in_turn and action_column is not None and int(
            features[picked_row, action_column]
        ) in TERMINAL_ACTIONS
        ordering = in_turn and not premature
        rank = int(np.flatnonzero(lab[order] == 1)[0])
        teacher_rank_hist[min(rank, 9)] += 1

        action = int(action_types[decisions[local]])
        bucket = by_class.setdefault(action, Counter())
        for target in (totals, bucket):
            target["count"] += 1
            target["correct"] += int(correct)
            target["ordering"] += int(ordering)
            target["premature"] += int(premature)
            target["divergence"] += int(not correct and not in_turn)
            target["in_top3"] += int(rank < 3)
            target["candidates"] += int(b - a)

    def summarise(stats):
        n = max(1, stats["count"])
        return {
            "count": int(stats["count"]),
            "top1": stats["correct"] / n,
            "turn_set": (
                stats["correct"] + stats["ordering"] + stats["premature"]
            ) / n,
            "ordering_error_rate": stats["ordering"] / n,
            "premature_rate": stats["premature"] / n,
            "divergence_rate": stats["divergence"] / n,
            "unrecoverable_rate": (
                stats["premature"] + stats["divergence"]
            ) / n,
            "top3": stats["in_top3"] / n,
            "mean_candidates": stats["candidates"] / n,
        }

    return {
        "overall": summarise(totals),
        "by_teacher_action": {
            ACTION_TYPES[action]: summarise(stats)
            for action, stats in sorted(
                by_class.items(),
                key=lambda kv: -kv[1]["divergence"],
            )
        },
        "teacher_rank_histogram": {
            str(rank): int(count)
            for rank, count in sorted(teacher_rank_hist.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--model", default="large_leaf")
    parser.add_argument("--n-estimators", type=int, default=2200)
    parser.add_argument("--episode-fraction", type=float, default=0.875)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--tree-step", type=int, default=50)
    parser.add_argument(
        "--reuse-scores", action="store_true",
        help="Re-score cached predictions instead of refitting the booster.",
    )
    args = parser.parse_args()

    cache = load_cache(args.cache)
    features = cache["features"]
    labels = cache["labels"]
    groups = cache["groups"]
    names = cache["names"]
    episode_ids = cache["episode_ids"]

    base_names = [n for n in names if n not in TURN_FEATURES]
    base_features = np.ascontiguousarray(
        features[:, [names.index(n) for n in base_names]]
    )
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
    print(
        f"fit: {len(keep)}/{len(ordered)} episodes, "
        f"{len(fit_decisions)} decisions, {len(fit_rows)} rows",
        flush=True,
    )

    if args.reuse_scores:
        with np.load(args.scores, allow_pickle=False) as stored:
            cached_scores = {
                "validation": stored["validation"], "test": stored["test"]
            }
            trees = int(stored["trees"])
        held_out = {
            split: residual_report(
                cached_scores[split], labels[rows[split]], group_sizes[split],
                decisions[split], features, rows[split], sem_cols,
                pick_sets, cache["action_types"], names.index("action_type"),
            )
            for split in ("validation", "test")
        }
        report = {
            "cache": str(args.cache.resolve()),
            "reused_scores": str(args.scores.resolve()),
            "selected_trees": trees,
            "validation": held_out["validation"],
            "test": held_out["test"],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(held_out["test"]["overall"], indent=2), flush=True)
        print(json.dumps(held_out["test"]["by_teacher_action"], indent=2),
              flush=True)
        return 0

    started = time.perf_counter()
    model = fit_fixed(
        args.model, base_features, base_names, fit_rows,
        graded[fit_rows], fit_groups, fit_weight, args.seed,
        args.n_estimators, True,
    )
    fit_seconds = time.perf_counter() - started
    print(f"fit took {fit_seconds:.1f}s", flush=True)

    curve = []
    for trees in range(args.tree_step, args.n_estimators + 1, args.tree_step):
        scores = model.predict(
            base_features[rows["validation"]], num_iteration=trees
        ).astype(np.float32)
        report = residual_report(
            scores, labels[rows["validation"]], group_sizes["validation"],
            decisions["validation"], features, rows["validation"], sem_cols,
            pick_sets, cache["action_types"], names.index("action_type"),
        )
        curve.append({
            "trees": trees,
            "validation_top1": report["overall"]["top1"],
            "validation_turn_set": report["overall"]["turn_set"],
        })
    best = max(curve, key=lambda p: p["validation_top1"])
    print(json.dumps(best), flush=True)

    held_out = {}
    cached_scores = {}
    for split in ("validation", "test"):
        scores = model.predict(
            base_features[rows[split]], num_iteration=best["trees"]
        ).astype(np.float32)
        cached_scores[split] = scores
        held_out[split] = residual_report(
            scores, labels[rows[split]], group_sizes[split],
            decisions[split], features, rows[split], sem_cols,
            pick_sets, cache["action_types"], names.index("action_type"),
        )

    args.scores.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores,
        validation=cached_scores["validation"],
        test=cached_scores["test"],
        trees=np.asarray(best["trees"]),
    )

    report = {
        "cache": str(args.cache.resolve()),
        "model": args.model,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "episode_fraction": args.episode_fraction,
        "fit_seconds": fit_seconds,
        "fit_episodes": int(len(keep)),
        "fit_decisions": int(len(fit_decisions)),
        "selected_trees": int(best["trees"]),
        "tree_curve": curve,
        "validation": held_out["validation"],
        "test": held_out["test"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(held_out["test"]["overall"], indent=2), flush=True)
    print(json.dumps(held_out["test"]["by_teacher_action"], indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
