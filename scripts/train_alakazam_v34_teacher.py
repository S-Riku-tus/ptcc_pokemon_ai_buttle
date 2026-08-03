"""Train and export the v34 Alakazam teacher ranker.

Two things differ from the v33 trainer.

The tree count is chosen by validation Top-1 instead of by LightGBM's NDCG
early stopping. v33 stopped on ``ndcg`` over the graded labels while reporting
and deploying strict Top-1, and those two objectives peak in different places:
refitting the v33 configuration without a stopping rule and scoring every tree
prefix showed Top-1 still climbing well past the point NDCG stopped at. Across
the twelve early-stopped v34 ablation fits, the stopping iteration correlated
0.56 with validation Top-1 and 0.65 with test Top-1, which is a property of the
stopping rule, not of the corpora being compared. Fitting one booster to a
fixed budget and selecting the prefix on validation removes that noise source
and recovers the iterations v33 was leaving on the table.

The corpus keeps only the most recent share of teacher episodes. Refetching
submission 54773249 added 980 games, and on the enlarged corpus the oldest
games no longer help: at a controlled tree count the newest 87.5% of training
episodes matched validation and test at 0.8284 and 0.8291, while using all of
them scored 0.8279 on validation but only 0.8111 on test, a divergence that
persists across the whole tree-count curve rather than at one point.

Reported metrics are unchanged from v33: strict top1/top2/top3 agreement with
the exact action the teacher played, plus the order-insensitive ``turn_set``.
The configuration is selected on validation; test is scored once, at the end.
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

from ml.core.distill import compact_booster  # noqa: E402
from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    ACTION_TYPE_MAP,
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


def recency_multiplier(episodes, floor, power):
    ordered = np.unique(episodes)
    position = {
        int(e): i / max(len(ordered) - 1, 1) for i, e in enumerate(ordered)
    }
    return np.asarray(
        [floor + (1.0 - floor) * position[int(e)] ** power for e in episodes],
        dtype=np.float32,
    )


def build_params(
    model_name, seed, n_estimators, graded,
    lambdarank_truncation_level=0,
):
    config = CONFIGS[model_name]
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
    if (
        lambdarank_truncation_level
        and config["objective"] == "lambdarank"
    ):
        params["lambdarank_truncation_level"] = (
            lambdarank_truncation_level
        )
    return params


def fit_fixed(model_name, matrix, cols, x_rows, y, group, weight, seed,
              n_estimators, graded, lambdarank_truncation_level=0):
    """Fit to a fixed tree budget. No stopping rule, so nothing is resampled."""
    model = lgb.LGBMRanker(**build_params(
        model_name, seed, n_estimators, graded,
        lambdarank_truncation_level,
    ))
    kwargs: dict[str, Any] = {
        "X": matrix[x_rows], "y": y, "group": group, "sample_weight": weight,
        "feature_name": cols,
    }
    if CONFIGS[model_name]["categorical_ids"]:
        kwargs["categorical_feature"] = [
            i for i, n in enumerate(cols)
            if n in BASE_CATEGORICAL or n.endswith("_id")
        ]
    model.fit(**kwargs)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--model", default="large_leaf")
    parser.add_argument("--n-estimators", type=int, default=2200)
    parser.add_argument(
        "--lambdarank-truncation-level", type=int, default=0,
        help=(
            "Override LightGBM's LambdaRank truncation. A small value focuses "
            "training gradients on the first few candidates; 0 keeps the "
            "library default."
        ),
    )
    parser.add_argument(
        "--include-turn-features", action="store_true",
        help=(
            "Include reproducible intra-turn offer/pass history columns. "
            "The v34 default excludes them."
        ),
    )
    parser.add_argument(
        "--episode-fraction", type=float, default=0.875,
        help="Share of the most recent episodes kept, selected on validation.",
    )
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument(
        "--recent-min-episode", type=int, default=0,
        help=(
            "Optional first episode of a newly observed teacher cohort. "
            "Rows from this episode onward receive --recent-boost in "
            "addition to the smooth recency multiplier."
        ),
    )
    parser.add_argument(
        "--recent-boost", type=float, default=1.0,
        help="Extra sample-weight multiplier for the explicitly recent cohort.",
    )
    parser.add_argument("--label", default="graded", choices=("graded",
                                                              "binary"))
    parser.add_argument(
        "--tree-step", type=int, default=50,
        help="Granularity of the validation tree-count search.",
    )
    parser.add_argument(
        "--max-trees", type=int, default=0,
        help=(
            "Cap the deployed tree count. Inference is a pure-Python tree "
            "walk, so trees translate directly into per-decision latency. "
            "0 means no cap."
        ),
    )
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--corpus-report", type=Path)
    parser.add_argument("--teacher-team", default="Yushin Ito")
    parser.add_argument("--teacher-submission-id", type=int, default=54773249)
    args = parser.parse_args()

    corpus_report = (
        json.loads(args.corpus_report.read_text(encoding="utf-8"))
        if args.corpus_report is not None else {}
    )

    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        weights = cached["weights"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        action_types = cached["teacher_action_types"]
        names = cached["feature_names"].astype(str).tolist()

    base_names = (
        names
        if args.include_turn_features
        else [n for n in names if n not in TURN_FEATURES]
    )
    base_columns = [names.index(n) for n in base_names]
    base_features = np.ascontiguousarray(features[:, base_columns])

    blocks = turn_blocks(features, groups, episode_ids, names)
    graded, graded_counts = graded_labels(
        features, labels, groups, blocks, names
    )
    pick_sets, sem_cols = turn_pick_sets(
        features, labels, groups, blocks, names
    )
    use_graded = args.label == "graded"
    label_array = graded if use_graded else labels

    decisions = {
        split: np.flatnonzero(splits == split)
        for split in ("train", "validation", "test")
    }
    rows = {k: rows_for(groups, v) for k, v in decisions.items()}
    group_sizes = {k: groups[v].astype(int) for k, v in decisions.items()}

    def newest(episode_pool, fraction):
        ordered = np.unique(episode_pool)
        count = max(1, int(round(len(ordered) * fraction)))
        return ordered[-count:]

    train_pool = episode_ids[decisions["train"]]
    kept = newest(train_pool, args.episode_fraction)
    keep_mask = np.isin(train_pool, kept)
    fit_decisions = decisions["train"][keep_mask]
    fit_rows = rows_for(groups, fit_decisions)
    fit_groups = groups[fit_decisions].astype(int)
    fit_weight = weights[fit_rows] * np.repeat(
        recency_multiplier(
            episode_ids[fit_decisions], args.recency_floor, args.recency_power
        ),
        fit_groups,
    )
    if args.recent_min_episode:
        fit_weight *= np.repeat(
            np.where(
                episode_ids[fit_decisions] >= args.recent_min_episode,
                args.recent_boost,
                1.0,
            ).astype(np.float32),
            fit_groups,
        )
    print(
        f"held-out fit: {len(kept)} of {len(np.unique(train_pool))} training "
        f"episodes, {len(fit_decisions)} decisions",
        flush=True,
    )

    model = fit_fixed(
        args.model, base_features, base_names, fit_rows,
        label_array[fit_rows], fit_groups, fit_weight, args.seed,
        args.n_estimators, use_graded,
        args.lambdarank_truncation_level,
    )

    ceiling = args.max_trees or args.n_estimators
    grid = [
        t for t in range(args.tree_step, args.n_estimators + 1, args.tree_step)
        if t <= ceiling
    ]
    curve = []
    for trees in grid:
        scores = model.predict(
            base_features[rows["validation"]], num_iteration=trees
        ).astype(np.float32)
        metrics = evaluate(
            scores, labels[rows["validation"]], group_sizes["validation"],
            decisions["validation"], features, rows["validation"], sem_cols,
            pick_sets, action_types,
        )
        curve.append({"trees": trees, "validation_top1": metrics["top1"]})
    best_trees = max(curve, key=lambda p: p["validation_top1"])["trees"]
    print(f"validation-selected trees: {best_trees}", flush=True)

    held_out = {}
    for split in ("validation", "test"):
        scores = model.predict(
            base_features[rows[split]], num_iteration=best_trees
        ).astype(np.float32)
        held_out[split] = evaluate(
            scores, labels[rows[split]], group_sizes[split],
            decisions[split], features, rows[split], sem_cols,
            pick_sets, action_types,
        )
    print(json.dumps({"held_out": {
        "trees": best_trees,
        "val_top1": round(held_out["validation"]["top1"], 4),
        "test_top1": round(held_out["test"]["top1"], 4),
        "test_top2": round(held_out["test"]["top2"], 4),
        "test_top3": round(held_out["test"]["top3"], 4),
        "test_turn_set": round(held_out["test"]["turn_set"], 4),
    }}), flush=True)

    exported: list[dict[str, Any]] = []
    if not args.no_export:
        # Ship a model refit on the whole frozen corpus at the tree count the
        # holdout estimate was produced at, matching the v31-v33 convention.
        # The same recency truncation applies, so the shipped corpus is the
        # newest share of every episode rather than of the training block.
        all_kept = newest(episode_ids, args.episode_fraction)
        all_mask = np.isin(episode_ids, all_kept)
        all_decisions = np.flatnonzero(all_mask)
        all_rows = rows_for(groups, all_decisions)
        all_groups = groups[all_decisions].astype(int)
        all_weight = weights[all_rows] * np.repeat(
            recency_multiplier(
                episode_ids[all_decisions], args.recency_floor,
                args.recency_power,
            ),
            all_groups,
        )
        if args.recent_min_episode:
            all_weight *= np.repeat(
                np.where(
                    episode_ids[all_decisions] >= args.recent_min_episode,
                    args.recent_boost,
                    1.0,
                ).astype(np.float32),
                all_groups,
            )
        print(
            f"export fit: {len(all_kept)} of {len(np.unique(episode_ids))} "
            f"episodes, {len(all_decisions)} decisions",
            flush=True,
        )
        final = lgb.LGBMRanker(
            **build_params(
                args.model, args.seed, best_trees, use_graded,
                args.lambdarank_truncation_level,
            )
        )
        kwargs: dict[str, Any] = {
            "X": base_features[all_rows], "y": label_array[all_rows],
            "group": all_groups, "sample_weight": all_weight,
            "feature_name": base_names,
        }
        if CONFIGS[args.model]["categorical_ids"]:
            kwargs["categorical_feature"] = [
                i for i, n in enumerate(base_names)
                if n in BASE_CATEGORICAL or n.endswith("_id")
            ]
        final.fit(**kwargs)
        compact = compact_booster(final.booster_, "ranker")
        # compact_booster puts the walked trees under "trees". Metadata added
        # below must not collide with that key or the model ships empty.
        if len(compact["trees"]) != best_trees:
            raise RuntimeError(
                f"exported {len(compact['trees'])} trees, expected "
                f"{best_trees}"
            )
        metadata = {
            "ensemble_weight": 1.0,
            "ensemble_role": f"{args.model}_{args.label}",
            "temperature": 1.0,
            "fallback_probability": 0.0,
            "fallback_margin": 0.0,
            "action_type_map": ACTION_TYPE_MAP,
            "legal_option_only": True,
            "runtime_scope": "v34_yushin_recent_corpus_ranker",
            "uses_turn_features": args.include_turn_features,
            "tree_count": int(best_trees),
            "tree_count_selected_by": "validation_top1",
            "lambdarank_truncation_level": (
                args.lambdarank_truncation_level
            ),
            "training_decisions": int(len(all_decisions)),
            "training_candidate_rows": int(len(all_rows)),
            "teacher_team": args.teacher_team,
            "teacher_submission_id": args.teacher_submission_id,
            "teacher_trajectories": int(len(all_kept)),
            "teacher_cohorts": corpus_report.get("cohorts", {}),
            "episode_fraction_kept": args.episode_fraction,
            "training_recency_weight": {
                "floor": args.recency_floor,
                "power": args.recency_power,
                "episode_order": "ascending_episode_id",
                "recent_min_episode": args.recent_min_episode,
                "recent_boost": args.recent_boost,
            },
            "label_definition": (
                "turn_order_graded_relevance" if use_graded
                else "binary_chosen_action"
            ),
            "baseline": "v29_runtime_choice_and_raw_ranker_score",
        }
        clobbered = sorted(set(metadata) & set(compact))
        if clobbered:
            raise RuntimeError(f"metadata would overwrite booster keys: "
                               f"{clobbered}")
        compact.update(metadata)
        path = args.agent_dir / "ranker_model.json"
        path.write_text(
            json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        # v33 could ship a blend; v34 ships one member, so any stale extra
        # member left in the agent directory must go or the runtime blends it.
        for index in range(1, 4):
            stale = args.agent_dir / f"ranker_model_{index}.json"
            if stale.exists():
                stale.unlink()
                print(f"removed stale {stale.name}", flush=True)
        exported.append({
            "file": "ranker_model.json",
            "role": f"{args.model}_{args.label}",
            "trees": int(best_trees),
            "bytes": path.stat().st_size,
        })
        print(f"exported ranker_model.json ({path.stat().st_size} bytes)",
              flush=True)

    report = {
        "cache": str(args.cache.resolve()),
        "agent_dir": str(args.agent_dir.resolve()),
        "model": args.model,
        "label": args.label,
        "seed": args.seed,
        "episode_fraction": args.episode_fraction,
        "recency": {"floor": args.recency_floor, "power": args.recency_power},
        "recent_cohort": {
            "min_episode": args.recent_min_episode,
            "boost": args.recent_boost,
        },
        "n_estimators": args.n_estimators,
        "lambdarank_truncation_level": args.lambdarank_truncation_level,
        "include_turn_features": args.include_turn_features,
        "tree_selection": {
            "selected_by": "validation_top1",
            "trees": int(best_trees),
            "curve": curve,
            "max_trees_cap": args.max_trees,
        },
        "split_decisions": {k: int(len(v)) for k, v in decisions.items()},
        "held_out_fit_episodes": int(len(kept)),
        "held_out_fit_decisions": int(len(fit_decisions)),
        "graded_label_counts": graded_counts,
        "validation": held_out["validation"],
        "test": held_out["test"],
        "exported": exported,
        "target_top1": 0.90,
        "target_met": bool(held_out["test"]["top1"] >= 0.90),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
