"""Two-stage cascade: re-rank only the candidates stage one cannot separate.

The v34 residual has a specific shape. The teacher's action is inside the
ranker's Top-3 on 98.5% of held-out decisions and inside Top-1 on 83.0%, so
candidate *recall* is nearly solved and the loss is concentrated in ordering
the few candidates at the top. The v34 seed study sharpened that: six seeds of
the identical configuration disagree enough that at least one is right on
84.2% of decisions, while averaging them scores 79.4%. Errors are near-ties
resolved differently by each fit, not confident mistakes, and averaging cannot
break a tie it is itself made of.

A second ranker trained only on those near-ties can. Stage two sees the K
highest-scoring candidates per decision and nothing else, so every gradient it
receives comes from a pair stage one already found hard, and it gets stage
one's score, rank and gaps as extra columns.

Stage-two training needs stage-one scores on training decisions that stage one
did not fit, or it learns to copy a 96.9%-accurate teacher signal that will not
exist at inference. Scores therefore come from chronological K-fold refits.
The folds are contiguous in episode order rather than random so that a fold is
never scored by a model fitted on games that ran seconds later.

Both the pure override and a blend against the stage-one score are scored, and
K, tree count and blend weight are all selected on validation.
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
    LABEL_GAIN,
    ranges,
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

STAGE1_COLUMNS = (
    "stage1_score_z", "stage1_rank", "stage1_gap_to_top",
    "stage1_gap_to_next", "stage1_prob", "stage1_candidate_count",
)


def fit_stage1(base_features, base_names, categorical, rows, y, groups,
               weight, seed, trees, config):
    model = lgb.LGBMRanker(
        objective=config["objective"], metric="ndcg",
        num_leaves=config["num_leaves"], n_estimators=trees,
        learning_rate=config["learning_rate"],
        min_child_samples=config["min_child_samples"], max_depth=-1,
        subsample=0.9, subsample_freq=1,
        colsample_bytree=config["colsample_bytree"],
        reg_alpha=0.2, reg_lambda=1.0, random_state=seed,
        n_jobs=20, verbosity=-1, label_gain=LABEL_GAIN,
    )
    model.fit(
        X=base_features[rows], y=y, group=groups, sample_weight=weight,
        feature_name=base_names, categorical_feature=categorical,
    )
    return model


def stage1_columns(scores, group_sizes):
    """Per-candidate view of the stage-one ranking, in decision-local units."""
    starts, ends = ranges(np.asarray(group_sizes))
    out = np.zeros((len(scores), len(STAGE1_COLUMNS)), dtype=np.float32)
    for a, b in zip(starts, ends):
        block = scores[a:b].astype(np.float64)
        order = np.argsort(-block, kind="stable")
        mean, scale = block.mean(), max(block.std(), 1e-5)
        z = (block - mean) / scale
        top = block[order[0]]
        second = block[order[1]] if b - a > 1 else top
        shifted = np.exp(np.clip(block - top, -50.0, 50.0))
        prob = shifted / max(shifted.sum(), 1e-12)
        rank = np.empty(b - a, dtype=np.float32)
        rank[order] = np.arange(b - a, dtype=np.float32)
        out[a:b, 0] = z
        out[a:b, 1] = rank
        out[a:b, 2] = block - top
        out[a:b, 3] = np.where(rank == 0, block - second, block - top)
        out[a:b, 4] = prob
        out[a:b, 5] = b - a
    return out


def select_topk(scores, group_sizes, k):
    """Row offsets of the K best candidates per decision, best first."""
    starts, ends = ranges(np.asarray(group_sizes))
    picked: list[np.ndarray] = []
    sizes: list[int] = []
    for a, b in zip(starts, ends):
        order = np.argsort(-scores[a:b], kind="stable")[: min(k, b - a)]
        picked.append(a + order)
        sizes.append(len(order))
    return np.concatenate(picked), np.asarray(sizes, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("stage1_scores", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--model", default="large_leaf")
    parser.add_argument("--stage1-trees", type=int, default=2050)
    parser.add_argument("--stage2-trees", type=int, default=1200)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--episode-fraction", type=float, default=0.875)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--tree-step", type=int, default=100)
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
    config = CONFIGS[args.model]

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

    # ---- stage one, out of fold over the kept training episodes -----------
    if args.oof.exists():
        with np.load(args.oof, allow_pickle=False) as cached:
            oof_scores = cached["scores"]
            if len(oof_scores) != len(rows_for(groups, fit_decisions)):
                raise RuntimeError("cached OOF scores do not match the corpus")
        print(f"reusing OOF scores from {args.oof}", flush=True)
    else:
        fold_of = np.array_split(keep, args.folds)
        oof_rows = rows_for(groups, fit_decisions)
        oof_scores = np.zeros(len(oof_rows), dtype=np.float32)
        offset = 0
        row_position = {int(r): i for i, r in enumerate(oof_rows)}
        for index, held in enumerate(fold_of):
            in_fold = np.isin(episode_ids[fit_decisions], held)
            train_decisions = fit_decisions[~in_fold]
            score_decisions = fit_decisions[in_fold]
            train_rows = rows_for(groups, train_decisions)
            train_groups = groups[train_decisions].astype(int)
            weight = cache["weights"][train_rows] * np.repeat(
                recency_multiplier(
                    episode_ids[train_decisions], args.recency_floor,
                    args.recency_power,
                ),
                train_groups,
            )
            started = time.perf_counter()
            model = fit_stage1(
                base_features, base_names, categorical, train_rows,
                graded[train_rows], train_groups, weight, args.seed,
                args.stage1_trees, config,
            )
            score_rows = rows_for(groups, score_decisions)
            predicted = model.predict(
                base_features[score_rows], num_iteration=args.stage1_trees
            ).astype(np.float32)
            for row, value in zip(score_rows, predicted):
                oof_scores[row_position[int(row)]] = value
            print(
                f"fold {index + 1}/{args.folds}: {len(train_decisions)} fit / "
                f"{len(score_decisions)} scored in "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )
            offset += len(score_rows)
        args.oof.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.oof, scores=oof_scores)

    with np.load(args.stage1_scores, allow_pickle=False) as cached:
        holdout_scores = {
            "validation": cached["validation"], "test": cached["test"]
        }

    fit_rows = rows_for(groups, fit_decisions)
    fit_groups = groups[fit_decisions].astype(int)
    oof_report = residual_report(
        oof_scores, labels[fit_rows], fit_groups, fit_decisions, features,
        fit_rows, sem_cols, pick_sets, cache["action_types"],
    )
    print(json.dumps({"oof_stage1": {
        "top1": round(oof_report["overall"]["top1"], 4),
        "top3": round(oof_report["overall"]["top3"], 4),
        "turn_set": round(oof_report["overall"]["turn_set"], 4),
    }}), flush=True)

    stage1_extra = {
        "train": stage1_columns(oof_scores, fit_groups),
        "validation": stage1_columns(
            holdout_scores["validation"], group_sizes["validation"]
        ),
        "test": stage1_columns(
            holdout_scores["test"], group_sizes["test"]
        ),
    }
    block_rows = {
        "train": fit_rows,
        "validation": rows["validation"],
        "test": rows["test"],
    }
    block_groups = {
        "train": fit_groups,
        "validation": group_sizes["validation"],
        "test": group_sizes["test"],
    }
    block_scores = {
        "train": oof_scores,
        "validation": holdout_scores["validation"],
        "test": holdout_scores["test"],
    }
    block_decisions = {
        "train": fit_decisions,
        "validation": decisions["validation"],
        "test": decisions["test"],
    }
    stage2_names = base_names + list(STAGE1_COLUMNS)

    results: dict[str, Any] = {"oof_stage1": oof_report["overall"]}
    for k in args.k:
        picks = {
            split: select_topk(block_scores[split], block_groups[split], k)
            for split in ("train", "validation", "test")
        }

        def matrix(split):
            local, _ = picks[split]
            absolute = block_rows[split][local]
            return np.hstack([
                base_features[absolute], stage1_extra[split][local]
            ])

        train_local, train_sizes = picks["train"]
        train_labels = labels[block_rows["train"]][train_local]
        train_graded = graded[block_rows["train"]][train_local]
        # A decision whose teacher action stage one dropped is unrecoverable
        # here; keeping it would only teach stage two to rank noise.
        starts, ends = ranges(train_sizes)
        keep_decision = np.array([
            bool(np.any(train_labels[a:b] == 1)) for a, b in zip(starts, ends)
        ])
        keep_rows = np.concatenate([
            np.arange(a, b) for a, b, ok in zip(starts, ends, keep_decision)
            if ok
        ])
        kept_sizes = train_sizes[keep_decision]
        weight = cache["weights"][block_rows["train"]][train_local][keep_rows]
        weight = weight * np.repeat(
            recency_multiplier(
                episode_ids[block_decisions["train"]][keep_decision],
                args.recency_floor, args.recency_power,
            ),
            kept_sizes,
        )
        train_matrix = matrix("train")[keep_rows]

        started = time.perf_counter()
        stage2 = lgb.LGBMRanker(
            objective=config["objective"], metric="ndcg",
            num_leaves=config["num_leaves"], n_estimators=args.stage2_trees,
            learning_rate=config["learning_rate"],
            min_child_samples=config["min_child_samples"], max_depth=-1,
            subsample=0.9, subsample_freq=1,
            colsample_bytree=config["colsample_bytree"],
            reg_alpha=0.2, reg_lambda=1.0, random_state=args.seed,
            n_jobs=20, verbosity=-1, label_gain=LABEL_GAIN,
        )
        stage2.fit(
            X=train_matrix, y=train_graded[keep_rows], group=kept_sizes,
            sample_weight=weight, feature_name=stage2_names,
            categorical_feature=[
                i for i, n in enumerate(stage2_names)
                if n in BASE_CATEGORICAL or n.endswith("_id")
            ],
        )
        elapsed = time.perf_counter() - started
        print(f"k={k} stage2 fit {elapsed:.0f}s "
              f"({len(kept_sizes)} decisions)", flush=True)

        def combined(split, trees, blend):
            """Stage-one scores with the top-K block replaced or blended."""
            local, sizes = picks[split]
            raw = stage2.predict(
                matrix(split), num_iteration=trees
            ).astype(np.float64)
            out = block_scores[split].astype(np.float64).copy()
            # Push every non-selected candidate below the re-ranked block so
            # the argmax can only come from the K stage one shortlisted. The
            # offset has to clear the z-scored block by a wide margin, and
            # subtracting rather than flattening keeps the tail's own order
            # intact for the Top-3 and teacher-rank statistics.
            starts_all, ends_all = ranges(np.asarray(block_groups[split]))
            starts_k, ends_k = ranges(sizes)
            mask = np.zeros(len(out), dtype=bool)
            mask[local] = True
            out[~mask] -= 1e6
            for (a, b), (ka, kb) in zip(
                zip(starts_all, ends_all), zip(starts_k, ends_k)
            ):
                block = raw[ka:kb]
                mean, scale = block.mean(), max(block.std(), 1e-5)
                z2 = (block - mean) / scale
                base = out[local[ka:kb]]
                mean1, scale1 = base.mean(), max(base.std(), 1e-5)
                z1 = (base - mean1) / scale1
                out[local[ka:kb]] = blend * z1 + z2
                del a, b
            return out.astype(np.float32)

        grid = list(range(
            args.tree_step, args.stage2_trees + 1, args.tree_step
        ))
        best = None
        for trees in grid:
            for blend in (0.0, 0.25, 0.5, 1.0):
                scores = combined("validation", trees, blend)
                report = residual_report(
                    scores, labels[rows["validation"]],
                    group_sizes["validation"], decisions["validation"],
                    features, rows["validation"], sem_cols, pick_sets,
                    cache["action_types"],
                )
                entry = (
                    report["overall"]["top1"], trees, blend,
                    report["overall"]["turn_set"],
                )
                if best is None or entry[0] > best[0]:
                    best = entry
        top1, trees, blend, turn_set = best
        print(json.dumps({f"k{k}_validation": {
            "trees": trees, "blend": blend, "top1": round(top1, 4),
            "turn_set": round(turn_set, 4),
        }}), flush=True)

        held_out = {}
        for split in ("validation", "test"):
            scores = combined(split, trees, blend)
            held_out[split] = residual_report(
                scores, labels[rows[split]], group_sizes[split],
                decisions[split], features, rows[split], sem_cols, pick_sets,
                cache["action_types"],
            )
        results[f"k{k}"] = {
            "stage2_trees": trees,
            "blend_weight": blend,
            "fit_seconds": elapsed,
            "train_decisions": int(len(kept_sizes)),
            "validation": held_out["validation"],
            "test": held_out["test"],
        }
        print(json.dumps({f"k{k}_test": {
            "top1": round(held_out["test"]["overall"]["top1"], 4),
            "turn_set": round(held_out["test"]["overall"]["turn_set"], 4),
            "divergence": round(
                held_out["test"]["overall"]["divergence_rate"], 4
            ),
        }}), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
