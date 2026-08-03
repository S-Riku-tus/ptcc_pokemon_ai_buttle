"""Validation-only OOF contextual Top-K reranker for Lopunny v3.

The base LambdaRank model already places the teacher action in its Top-3 very
often, but frequently gets the order inside that shortlist wrong.  This
experiment learns that second-stage ordering without in-sample base scores:
episode folds produce out-of-fold Top-5 candidates for training, while the
validation shortlist comes from a base model fit on every training episode.
The final test split is never indexed or predicted.
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

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402


SEED = 55137818
CRITICAL = (
    "option_type", "action_type", "candidate_card_id", "candidate_attack_id",
    "candidate_area", "candidate_inplay_area", "candidate_raw_inplay_index",
    "candidate_target_id", "candidate_target_hp", "candidate_target_energy",
    "candidate_target_special_energy", "candidate_target_appear_this_turn",
    "candidate_hand_cost", "candidate_total_draw_count",
    "candidate_net_hand_delta", "post_action_hand_count",
    "candidate_option_position", "same_action_option_count",
    "same_card_option_count", "lopunny_attack_damage_estimate",
    "lopunny_attack_lethal_estimate",
)


def _rankable(arrays: dict[str, np.ndarray], decisions: np.ndarray) -> np.ndarray:
    return decisions[
        (arrays["chosen_counts"][decisions] > 0)
        & (arrays["chosen_counts"][decisions] < arrays["groups"][decisions])
        & (arrays["forced"][decisions] == 0)
    ]


def _fit_base(
    arrays: dict[str, np.ndarray],
    names: list[str],
    decisions: np.ndarray,
    varying: np.ndarray,
    trees: int,
    seed: int,
) -> lgb.LGBMRanker:
    rankable = _rankable(arrays, decisions)
    rows = v1._rows_for(arrays["groups"], rankable)
    group_sizes = arrays["groups"][rankable].astype(int)
    selected_names = [names[index] for index in varying]
    model = lgb.LGBMRanker(**v1._ranker_params(seed, trees, False))
    model.fit(
        arrays["features"][rows][:, varying], arrays["labels"][rows],
        group=group_sizes,
        sample_weight=np.repeat(
            v1._episode_recency(arrays["episode_ids"][rankable], 0.35, 2.0),
            group_sizes,
        ),
        feature_name=selected_names,
        categorical_feature=v1._categorical_columns(selected_names),
    )
    return model


def _predict_global(
    model: lgb.LGBMRanker,
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    varying: np.ndarray,
    trees: int,
    output: np.ndarray,
) -> None:
    starts, ends = v1._group_ranges(arrays["groups"])
    rows = v1._rows_for(arrays["groups"], decisions)
    values = model.predict(
        arrays["features"][rows][:, varying], num_iteration=trees
    ).astype(np.float32)
    offset = 0
    for decision in decisions:
        start, end = int(starts[decision]), int(ends[decision])
        size = end - start
        output[start:end] = values[offset:offset + size]
        offset += size


def _oof_scores(
    arrays: dict[str, np.ndarray],
    names: list[str],
    train: np.ndarray,
    varying: np.ndarray,
    trees: int,
    folds: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    episode_ids = np.unique(arrays["episode_ids"][train])
    fold_episodes = np.array_split(episode_ids, folds)
    output = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    audit: list[dict[str, int]] = []
    for fold, held_episodes in enumerate(fold_episodes):
        held = train[np.isin(arrays["episode_ids"][train], held_episodes)]
        fit = train[~np.isin(arrays["episode_ids"][train], held_episodes)]
        model = _fit_base(
            arrays, names, fit, varying, trees, SEED + 101 * (fold + 1)
        )
        _predict_global(model, arrays, held, varying, trees, output)
        audit.append({
            "fold": fold,
            "fit_episodes": int(len(np.unique(arrays["episode_ids"][fit]))),
            "held_episodes": int(len(held_episodes)),
            "held_decisions": int(len(held)),
        })
        print(json.dumps(audit[-1]), flush=True)
    return output, audit


def _schema(names: list[str], top_k: int) -> tuple[int, np.ndarray, list[str]]:
    candidate_start = names.index("option_type")
    critical = np.asarray(
        [names.index(name) for name in CRITICAL if name in names], dtype=np.int64
    )
    output = [f"original__{name}" for name in names]
    output += [f"relative__{name}" for name in names[candidate_start:]]
    output += [
        "context__base_z", "context__base_margin_best", "context__base_rank",
        "context__base_mean", "context__base_std", "context__topk_size",
    ]
    for slot in range(top_k):
        output += [f"slot_{slot}__{names[index]}" for index in critical]
    return candidate_start, critical, output


def _context_rows(
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    global_scores: np.ndarray,
    names: list[str],
    top_k: int,
    *,
    require_teacher_in_topk: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int], list[str]]:
    starts, ends = v1._group_ranges(arrays["groups"])
    candidate_start, critical, output_names = _schema(names, top_k)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    plans: list[dict[str, Any]] = []
    stats = {"eligible": 0, "teacher_in_topk": 0, "retained": 0}
    for decision_value in decisions:
        decision = int(decision_value)
        if (
            int(arrays["select_contexts"][decision]) != 0
            or bool(arrays["forced"][decision])
            or int(arrays["chosen_counts"][decision]) != 1
        ):
            continue
        start, end = int(starts[decision]), int(ends[decision])
        block_scores = global_scores[start:end]
        if not np.all(np.isfinite(block_scores)):
            continue
        stats["eligible"] += 1
        order = np.argsort(-block_scores, kind="stable")[:min(top_k, end - start)]
        teacher = int(np.flatnonzero(arrays["labels"][start:end] == 1)[0])
        teacher_key = tuple(arrays["semantics"][start + teacher].tolist())
        top_labels = np.asarray([
            int(tuple(arrays["semantics"][start + local].tolist()) == teacher_key)
            for local in order
        ], dtype=np.int8)
        in_topk = bool(np.any(top_labels))
        stats["teacher_in_topk"] += int(in_topk)
        if require_teacher_in_topk and not in_topk:
            continue

        features = arrays["features"][start:end]
        candidate_matrix = features[:, candidate_start:]
        mean = candidate_matrix.mean(axis=0)
        scale = np.maximum(candidate_matrix.std(axis=0), 1e-5)
        score_mean = float(block_scores.mean())
        score_std = max(float(block_scores.std()), 1e-5)
        z = (block_scores - score_mean) / score_std
        best = float(block_scores[order[0]])
        slot_values = np.full((top_k, len(critical)), -1.0, dtype=np.float32)
        slot_values[:len(order)] = features[order][:, critical]
        flattened_slots = slot_values.reshape(-1)
        row_begin = len(rows)
        for rank, local in enumerate(order):
            original = features[local]
            relative = (candidate_matrix[local] - mean) / scale
            context = np.asarray([
                z[local], block_scores[local] - best, rank,
                score_mean, score_std, len(order),
            ], dtype=np.float32)
            rows.append(np.concatenate((
                original, relative, context, flattened_slots
            )).astype(np.float32))
            labels.append(int(top_labels[rank]))
        groups.append(len(order))
        plans.append({
            "decision": decision,
            "order": order,
            "row_begin": row_begin,
            "row_end": len(rows),
            "base_z": z[order].astype(np.float32),
        })
        stats["retained"] += 1
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(groups, dtype=np.int32),
        plans,
        stats,
        output_names,
    )


def _categorical(names: list[str]) -> list[int]:
    categorical_tokens = (
        "action_type", "option_type", "select_type", "select_context",
        "_id", "_area", "_player_relative", "_special_condition",
    )
    return [
        index for index, name in enumerate(names)
        if name.startswith("original__") or name.startswith("slot_")
        if any(token in name for token in categorical_tokens)
    ]


def _reranked_scores(
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    base_global: np.ndarray,
    predictions: np.ndarray,
    plans: list[dict[str, Any]],
    top_k: int,
    alpha: float,
) -> np.ndarray:
    starts, ends = v1._group_ranges(arrays["groups"])
    base_rows = v1._rows_for(arrays["groups"], decisions)
    output = base_global[base_rows].copy()
    local_offset = {
        int(decision): int(offset)
        for decision, offset in zip(
            decisions, np.r_[0, np.cumsum(arrays["groups"][decisions])[:-1]]
        )
    }
    for plan in plans:
        decision = int(plan["decision"])
        order = plan["order"][:top_k]
        begin, end = int(plan["row_begin"]), int(plan["row_end"])
        values = predictions[begin:end][:top_k]
        base_z = plan["base_z"][:top_k]
        offset = local_offset[decision]
        size = int(ends[decision] - starts[decision])
        output[offset:offset + size] = -1e6
        output[offset + order] = values + alpha * base_z
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-trees", type=int, default=900)
    parser.add_argument("--reranker-trees", type=int, default=1200)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    starts, _ = v1._group_ranges(arrays["groups"])
    arrays["decision_turns"] = np.rint(
        arrays["features"][starts, names.index("turn")]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)
    varying = v1._varying_columns(
        arrays["features"],
        v1._rows_for(arrays["groups"], _rankable(arrays, train)),
    )

    started = time.perf_counter()
    oof, fold_audit = _oof_scores(
        arrays, names, train, varying, args.base_trees, args.folds
    )
    base = _fit_base(
        arrays, names, train, varying, args.base_trees, SEED
    )
    validation_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    _predict_global(
        base, arrays, validation, varying, args.base_trees, validation_global
    )

    train_x, train_y, train_groups, _, train_stats, reranker_names = _context_rows(
        arrays, train, oof, names, 5, require_teacher_in_topk=True
    )
    validation_x, _, _, validation_plans, validation_stats, _ = _context_rows(
        arrays, validation, validation_global, names, 5,
        require_teacher_in_topk=False,
    )
    reranker = lgb.LGBMRanker(
        objective="lambdarank", metric="None",
        n_estimators=args.reranker_trees, learning_rate=0.025,
        num_leaves=127, min_child_samples=28, max_depth=-1,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.82,
        reg_alpha=0.25, reg_lambda=1.8, random_state=SEED + 700,
        n_jobs=20, verbosity=-1, label_gain=[0, 1],
    )
    reranker.fit(
        train_x, train_y, group=train_groups,
        feature_name=reranker_names,
        categorical_feature=_categorical(reranker_names),
    )

    count_names = arrays["count_feature_names"].astype(str).tolist()
    variable = train[arrays["minimums"][train] < arrays["maximums"][train]]
    count_model = lgb.LGBMRegressor(**v1._count_params(SEED, 250))
    count_model.fit(
        arrays["count_features"][variable], arrays["chosen_counts"][variable],
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][variable], 0.35, 2.0
        ),
        feature_name=count_names,
        categorical_feature=v1._categorical_columns(count_names),
    )
    counts = v1._predict_counts(
        count_model, arrays["count_features"], validation,
        arrays["minimums"], arrays["maximums"], num_iteration=250,
    )
    base_rows = v1._rows_for(arrays["groups"], validation)
    base_scores = validation_global[base_rows]
    base_metrics = v1.evaluate(base_scores, validation, arrays, counts)

    experiments: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for trees in range(100, args.reranker_trees + 1, 100):
        predictions = reranker.predict(
            validation_x, num_iteration=trees
        ).astype(np.float32)
        for top_k in (3, 5):
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                scores = _reranked_scores(
                    arrays, validation, validation_global, predictions,
                    validation_plans, top_k, alpha,
                )
                metrics = v1.evaluate(scores, validation, arrays, counts)
                row = {
                    "trees": trees, "top_k": top_k, "alpha": alpha,
                    "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
                    "single_top1": metrics["single_choice_semantic_top1"],
                    "main_top1": metrics["main_single_choice_semantic_top1"],
                }
                experiments.append(row)
                if best is None or (
                    row["nonforced_semantic_exact"], row["main_top1"],
                    -row["trees"], -row["top_k"]
                ) > (
                    best["nonforced_semantic_exact"], best["main_top1"],
                    -best["trees"], -best["top_k"]
                ):
                    best = row

    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()),
        "test_read": False,
        "base_trees": args.base_trees,
        "folds": fold_audit,
        "train_top5": train_stats,
        "validation_top5": validation_stats,
        "reranker_rows": int(len(train_y)),
        "reranker_features": int(train_x.shape[1]),
        "fit_seconds": time.perf_counter() - started,
        "base_validation": base_metrics,
        "selected": best,
        "target": {
            "metric": "validation_nonforced_semantic_exact",
            "value": 0.85,
            "met": bool(best and best["nonforced_semantic_exact"] >= 0.85),
        },
        "experiments": experiments,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "base": base_metrics["nonforced_semantic_exact"],
        "selected": best,
        "target": report["target"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
