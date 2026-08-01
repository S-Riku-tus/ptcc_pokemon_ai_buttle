"""Train and export the all-context Majkel1337 Lopunny imitation policy.

Two models are learned from the frozen chronological corpus:

* a LightGBM LambdaRank candidate model for *which* legal options to choose;
* a LightGBM robust regressor for *how many* options to choose when min/max
  allow a variable count.

Neither model uses early stopping.  A fixed tree budget is fit once and each
tree prefix is scored by the deployment metric on validation.  Test is scored
once after both prefix lengths have been selected.  Agreement is semantic:
choosing either indistinguishable copy of the same card is counted as the same
action, while different board targets retain their identity and condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import compact_booster  # noqa: E402


BASE_CATEGORICAL = {
    "action_type",
    "option_type",
    "select_type",
    "select_context",
    "candidate_raw_player_relative",
    "candidate_area",
    "candidate_inplay_area",
}


def _group_ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    return starts, ends


def _rows_for(groups: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    starts, ends = _group_ranges(groups)
    if len(decisions) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([
        np.arange(starts[index], ends[index], dtype=np.int64)
        for index in decisions
    ])


def _episode_recency(episodes: np.ndarray, floor: float, power: float) -> np.ndarray:
    ordered = np.unique(episodes)
    position = {
        int(episode): index / max(1, len(ordered) - 1)
        for index, episode in enumerate(ordered)
    }
    return np.asarray([
        floor + (1.0 - floor) * position[int(episode)] ** power
        for episode in episodes
    ], dtype=np.float32)


def _categorical_columns(names: list[str]) -> list[int]:
    return [
        index for index, name in enumerate(names)
        if name in BASE_CATEGORICAL
        or name.endswith("_id")
        or "_id_" in name
    ]


def _varying_columns(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Drop constants using chunks so the 0.9 GB candidate matrix is not copied."""
    minimum = np.full(matrix.shape[1], np.inf, dtype=np.float64)
    maximum = np.full(matrix.shape[1], -np.inf, dtype=np.float64)
    for begin in range(0, len(rows), 20_000):
        chunk = matrix[rows[begin:begin + 20_000]]
        minimum = np.minimum(minimum, np.nanmin(chunk, axis=0))
        maximum = np.maximum(maximum, np.nanmax(chunk, axis=0))
    return np.flatnonzero(minimum != maximum)


def _predict_counts(
    model: lgb.LGBMRegressor | None,
    count_features: np.ndarray,
    decisions: np.ndarray,
    minimums: np.ndarray,
    maximums: np.ndarray,
    *,
    num_iteration: int | None = None,
    baseline: str | None = None,
) -> dict[int, int]:
    result: dict[int, int] = {}
    variable = decisions[minimums[decisions] < maximums[decisions]]
    if not len(variable):
        return result
    if baseline == "minimum":
        raw = minimums[variable].astype(float)
    elif baseline == "maximum":
        raw = maximums[variable].astype(float)
    elif model is None:
        raise ValueError("count model is required")
    else:
        raw = model.predict(
            count_features[variable], num_iteration=num_iteration
        )
    for decision, value in zip(variable, raw):
        result[int(decision)] = int(np.clip(
            np.rint(value), minimums[decision], maximums[decision]
        ))
    return result


def _semantic_counter(values: np.ndarray) -> Counter[tuple[int, ...]]:
    return Counter(tuple(int(value) for value in row) for row in values)


def _turn_blocks(arrays: dict[str, Any]) -> list[list[int]]:
    blocks: list[list[int]] = []
    current: list[int] = []
    identity: tuple[int, int] | None = None
    for decision, (episode, turn) in enumerate(zip(
        arrays["episode_ids"], arrays["decision_turns"]
    )):
        value = (int(episode), int(turn))
        if identity is not None and value != identity:
            blocks.append(current)
            current = []
        current.append(decision)
        identity = value
    if current:
        blocks.append(current)
    return blocks


def _turn_pick_sets(arrays: dict[str, Any]) -> list[set[tuple[int, ...]]]:
    starts, ends = _group_ranges(arrays["groups"])
    labels = arrays["labels"]
    semantics = arrays["semantics"]
    result: list[set[tuple[int, ...]]] = [set() for _ in arrays["groups"]]
    for block in _turn_blocks(arrays):
        selected: set[tuple[int, ...]] = set()
        for decision in block:
            start, end = int(starts[decision]), int(ends[decision])
            for row in np.flatnonzero(labels[start:end] == 1):
                selected.add(tuple(int(value) for value in semantics[start + row]))
        for decision in block:
            result[decision] = selected
    return result


def _graded_labels(arrays: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    """Chosen=3, next decision=2, later in the same turn=1, else=0."""
    starts, ends = _group_ranges(arrays["groups"])
    binary = arrays["labels"]
    semantics = arrays["semantics"]
    graded = np.zeros_like(binary, dtype=np.int8)
    counts: Counter[str] = Counter()
    for block in _turn_blocks(arrays):
        selected_by_decision: list[set[tuple[int, ...]]] = []
        for decision in block:
            start, end = int(starts[decision]), int(ends[decision])
            selected_by_decision.append({
                tuple(int(value) for value in semantics[start + row])
                for row in np.flatnonzero(binary[start:end] == 1)
            })
        for position, decision in enumerate(block):
            start, end = int(starts[decision]), int(ends[decision])
            next_set = (
                selected_by_decision[position + 1]
                if position + 1 < len(block) else set()
            )
            later_set: set[tuple[int, ...]] = set()
            for values in selected_by_decision[position + 2:]:
                later_set.update(values)
            for row in range(start, end):
                key = tuple(int(value) for value in semantics[row])
                if binary[row] == 1:
                    graded[row] = 3
                    counts["chosen"] += 1
                elif key in next_set:
                    graded[row] = 2
                    counts["played_next"] += 1
                elif key in later_set:
                    graded[row] = 1
                    counts["played_later"] += 1
                else:
                    counts["not_played"] += 1
    return graded, dict(counts)


def evaluate(
    scores: np.ndarray,
    decisions: np.ndarray,
    arrays: dict[str, np.ndarray],
    count_predictions: dict[int, int],
) -> dict[str, Any]:
    groups = arrays["groups"]
    starts, ends = _group_ranges(groups)
    labels = arrays["labels"]
    semantics = arrays["semantics"]
    chosen_counts = arrays["chosen_counts"]
    minimums = arrays["minimums"]
    forced = arrays["forced"]
    contexts = arrays["select_contexts"]
    select_types = arrays["select_types"]

    totals: Counter[str] = Counter()
    by_context: dict[str, Counter[str]] = defaultdict(Counter)
    score_offset = 0
    for decision in decisions:
        decision = int(decision)
        size = int(groups[decision])
        group_scores = scores[score_offset:score_offset + size]
        score_offset += size
        start, end = int(starts[decision]), int(ends[decision])
        group_labels = labels[start:end]
        teacher_local = np.flatnonzero(group_labels == 1)
        teacher_count = int(chosen_counts[decision])
        predicted_count = (
            count_predictions.get(decision, int(minimums[decision]))
            if minimums[decision] < arrays["maximums"][decision]
            else int(minimums[decision])
        )
        order = np.argsort(-group_scores, kind="stable")
        predicted_local = order[:predicted_count]
        teacher_semantic = _semantic_counter(semantics[start:end][teacher_local])
        predicted_semantic = _semantic_counter(semantics[start:end][predicted_local])
        semantic_exact = teacher_semantic == predicted_semantic
        raw_exact = set(map(int, teacher_local)) == set(map(int, predicted_local))
        count_correct = predicted_count == teacher_count
        is_forced = bool(forced[decision])
        nonforced = not is_forced

        totals["decisions"] += 1
        totals["semantic_exact"] += int(semantic_exact)
        totals["raw_exact"] += int(raw_exact)
        totals["count_correct"] += int(count_correct)
        totals["forced"] += int(is_forced)
        totals["nonforced"] += int(nonforced)
        totals["nonforced_semantic_exact"] += int(nonforced and semantic_exact)
        totals["nonforced_raw_exact"] += int(nonforced and raw_exact)
        totals["nonforced_count_correct"] += int(nonforced and count_correct)
        totals["variable_count"] += int(minimums[decision] < arrays["maximums"][decision])
        totals["variable_count_correct"] += int(
            minimums[decision] < arrays["maximums"][decision] and count_correct
        )

        if nonforced and teacher_count == 1:
            totals["single_choice"] += 1
            teacher_key = next(iter(teacher_semantic))
            ranked_keys = [
                tuple(int(value) for value in semantics[start + local])
                for local in order[:3]
            ]
            totals["single_top1"] += int(ranked_keys[0] == teacher_key)
            totals["single_top2"] += int(teacher_key in ranked_keys[:2])
            totals["single_top3"] += int(teacher_key in ranked_keys[:3])
            totals["single_turn_set"] += int(
                ranked_keys[0] in arrays["turn_pick_sets"][decision]
            )
            if int(contexts[decision]) == 0:
                totals["main_single"] += 1
                totals["main_single_top1"] += int(ranked_keys[0] == teacher_key)

        key = f"type_{int(select_types[decision])}_context_{int(contexts[decision])}"
        bucket = by_context[key]
        bucket["decisions"] += 1
        bucket["semantic_exact"] += int(semantic_exact)
        bucket["nonforced"] += int(nonforced)
        bucket["nonforced_semantic_exact"] += int(nonforced and semantic_exact)
        if nonforced and teacher_count == 1:
            bucket["single_choice"] += 1
            bucket["single_top1"] += int(
                tuple(int(value) for value in semantics[start + order[0]])
                == next(iter(teacher_semantic))
            )

    if score_offset != len(scores):
        raise RuntimeError(f"Consumed {score_offset} scores, got {len(scores)}")

    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(1, totals[denominator])

    result: dict[str, Any] = {
        "decisions": totals["decisions"],
        "semantic_exact": ratio("semantic_exact", "decisions"),
        "raw_exact": ratio("raw_exact", "decisions"),
        "count_accuracy": ratio("count_correct", "decisions"),
        "forced_decisions": totals["forced"],
        "nonforced_decisions": totals["nonforced"],
        "nonforced_semantic_exact": ratio(
            "nonforced_semantic_exact", "nonforced"
        ),
        "nonforced_raw_exact": ratio("nonforced_raw_exact", "nonforced"),
        "nonforced_count_accuracy": ratio(
            "nonforced_count_correct", "nonforced"
        ),
        "variable_count_decisions": totals["variable_count"],
        "variable_count_accuracy": ratio(
            "variable_count_correct", "variable_count"
        ),
        "single_choice_decisions": totals["single_choice"],
        "single_choice_semantic_top1": ratio("single_top1", "single_choice"),
        "single_choice_semantic_top2": ratio("single_top2", "single_choice"),
        "single_choice_semantic_top3": ratio("single_top3", "single_choice"),
        "single_choice_turn_set": ratio("single_turn_set", "single_choice"),
        "main_single_choice_decisions": totals["main_single"],
        "main_single_choice_semantic_top1": ratio(
            "main_single_top1", "main_single"
        ),
        "by_context": {},
    }
    for key, bucket in sorted(by_context.items()):
        result["by_context"][key] = {
            "decisions": bucket["decisions"],
            "semantic_exact": bucket["semantic_exact"] / max(1, bucket["decisions"]),
            "nonforced_decisions": bucket["nonforced"],
            "nonforced_semantic_exact": (
                bucket["nonforced_semantic_exact"] / max(1, bucket["nonforced"])
            ),
            "single_choice_decisions": bucket["single_choice"],
            "single_choice_semantic_top1": (
                bucket["single_top1"] / max(1, bucket["single_choice"])
            ),
        }
    return result


def _ranker_params(seed: int, trees: int, graded: bool) -> dict[str, Any]:
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "n_estimators": trees,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_child_samples": 24,
        "max_depth": -1,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.78,
        "reg_alpha": 0.20,
        "reg_lambda": 1.5,
        "random_state": seed,
        "n_jobs": 20,
        "verbosity": -1,
    }
    params["label_gain"] = [0, 1, 3, 7] if graded else [0, 1]
    return params


def _count_params(seed: int, trees: int) -> dict[str, Any]:
    return {
        "objective": "regression_l1",
        "metric": "None",
        "n_estimators": trees,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 18,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 1.0,
        "random_state": seed + 17,
        "n_jobs": 20,
        "verbosity": -1,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=55137818)
    parser.add_argument("--ranker-trees", type=int, default=1200)
    parser.add_argument("--count-trees", type=int, default=500)
    parser.add_argument("--ranker-step", type=int, default=50)
    parser.add_argument("--count-step", type=int, default=25)
    parser.add_argument("--recency-floor", type=float, default=0.40)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--label", choices=("binary", "graded"), default="binary")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {name: cached[name] for name in cached.files}
    features = arrays["features"]
    feature_names = arrays["feature_names"].astype(str).tolist()
    count_features = arrays["count_features"]
    count_feature_names = arrays["count_feature_names"].astype(str).tolist()
    groups = arrays["groups"]
    labels = arrays["labels"]
    splits = arrays["splits"].astype(str)
    episodes = arrays["episode_ids"]
    group_starts, _ = _group_ranges(groups)
    turn_column = feature_names.index("turn")
    arrays["decision_turns"] = np.rint(
        features[group_starts, turn_column]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = _turn_pick_sets(arrays)
    graded_labels, graded_counts = _graded_labels(arrays)
    fit_labels = graded_labels if args.label == "graded" else labels

    decisions = {
        split: np.flatnonzero(splits == split)
        for split in ("train", "validation", "test")
    }
    rankable = {
        split: values[
            (arrays["chosen_counts"][values] > 0)
            & (arrays["chosen_counts"][values] < groups[values])
            & (arrays["forced"][values] == 0)
        ]
        for split, values in decisions.items()
    }
    train_rows_all = _rows_for(groups, rankable["train"])
    varying = _varying_columns(features, train_rows_all)
    selected_names = [feature_names[index] for index in varying]
    selected_features = np.ascontiguousarray(features[:, varying])
    categorical = _categorical_columns(selected_names)
    print(
        f"features: {len(feature_names)} -> {len(selected_names)} varying; "
        f"categorical={len(categorical)}",
        flush=True,
    )

    train_rows = _rows_for(groups, rankable["train"])
    train_groups = groups[rankable["train"]].astype(int)
    decision_weight = _episode_recency(
        episodes[rankable["train"]], args.recency_floor, args.recency_power
    )
    row_weight = np.repeat(decision_weight, train_groups)
    ranker = lgb.LGBMRanker(**_ranker_params(
        args.seed, args.ranker_trees, args.label == "graded"
    ))
    ranker.fit(
        selected_features[train_rows],
        fit_labels[train_rows],
        group=train_groups,
        sample_weight=row_weight,
        feature_name=selected_names,
        categorical_feature=categorical,
    )
    print(f"ranker fit: {len(rankable['train'])} decisions", flush=True)

    variable_train = decisions["train"][
        arrays["minimums"][decisions["train"]]
        < arrays["maximums"][decisions["train"]]
    ]
    count_model = lgb.LGBMRegressor(**_count_params(args.seed, args.count_trees))
    count_model.fit(
        count_features[variable_train],
        arrays["chosen_counts"][variable_train],
        sample_weight=_episode_recency(
            episodes[variable_train], args.recency_floor, args.recency_power
        ),
        feature_name=count_feature_names,
        categorical_feature=_categorical_columns(count_feature_names),
    )
    print(f"count fit: {len(variable_train)} decisions", flush=True)

    count_curve = []
    for trees in range(args.count_step, args.count_trees + 1, args.count_step):
        predicted = _predict_counts(
            count_model, count_features, decisions["validation"],
            arrays["minimums"], arrays["maximums"], num_iteration=trees,
        )
        variable = [
            decision for decision in decisions["validation"]
            if arrays["minimums"][decision] < arrays["maximums"][decision]
        ]
        accuracy = np.mean([
            predicted[int(decision)] == arrays["chosen_counts"][decision]
            for decision in variable
        ]) if variable else 1.0
        count_curve.append({"trees": trees, "validation_count_accuracy": float(accuracy)})
    best_count_trees = max(
        count_curve,
        key=lambda item: (item["validation_count_accuracy"], -item["trees"]),
    )["trees"]
    validation_counts = _predict_counts(
        count_model, count_features, decisions["validation"],
        arrays["minimums"], arrays["maximums"],
        num_iteration=best_count_trees,
    )

    validation_rows = _rows_for(groups, decisions["validation"])
    ranker_curve = []
    validation_metrics_by_tree: dict[int, dict[str, Any]] = {}
    for trees in range(args.ranker_step, args.ranker_trees + 1, args.ranker_step):
        scores = ranker.predict(
            selected_features[validation_rows], num_iteration=trees
        ).astype(np.float32)
        metrics = evaluate(
            scores, decisions["validation"], arrays, validation_counts
        )
        validation_metrics_by_tree[trees] = metrics
        ranker_curve.append({
            "trees": trees,
            "validation_nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
            "validation_single_top1": metrics["single_choice_semantic_top1"],
            "validation_main_single_top1": metrics["main_single_choice_semantic_top1"],
        })
    best_ranker_trees = max(
        ranker_curve,
        key=lambda item: (
            item["validation_nonforced_semantic_exact"],
            item["validation_single_top1"],
            -item["trees"],
        ),
    )["trees"]
    validation_metrics = validation_metrics_by_tree[best_ranker_trees]
    print(
        json.dumps({
            "selected_ranker_trees": best_ranker_trees,
            "selected_count_trees": best_count_trees,
            "validation_nonforced_exact": validation_metrics["nonforced_semantic_exact"],
            "validation_single_top1": validation_metrics["single_choice_semantic_top1"],
        }),
        flush=True,
    )

    test_rows = _rows_for(groups, decisions["test"])
    test_scores = ranker.predict(
        selected_features[test_rows], num_iteration=best_ranker_trees
    ).astype(np.float32)
    test_counts = _predict_counts(
        count_model, count_features, decisions["test"],
        arrays["minimums"], arrays["maximums"],
        num_iteration=best_count_trees,
    )
    test_metrics = evaluate(test_scores, decisions["test"], arrays, test_counts)

    # Honest no-learning baselines use stable option order and the visible
    # selection bounds.  They expose how much the learned policy adds.
    baseline_scores = np.zeros(len(test_rows), dtype=np.float32)
    baselines = {}
    for name in ("minimum", "maximum"):
        baseline_counts = _predict_counts(
            None, count_features, decisions["test"],
            arrays["minimums"], arrays["maximums"], baseline=name,
        )
        baselines[f"first_option_{name}_count"] = evaluate(
            baseline_scores, decisions["test"], arrays, baseline_counts
        )

    exported: list[dict[str, Any]] = []
    if not args.no_export:
        args.agent_dir.mkdir(parents=True, exist_ok=True)
        all_decisions = np.arange(len(groups), dtype=np.int64)
        all_rankable = all_decisions[
            (arrays["chosen_counts"] > 0)
            & (arrays["chosen_counts"] < groups)
            & (arrays["forced"] == 0)
        ]
        all_rows = _rows_for(groups, all_rankable)
        all_groups = groups[all_rankable].astype(int)
        final_ranker = lgb.LGBMRanker(
            **_ranker_params(
                args.seed, best_ranker_trees, args.label == "graded"
            )
        )
        final_ranker.fit(
            selected_features[all_rows],
            fit_labels[all_rows],
            group=all_groups,
            sample_weight=np.repeat(
                _episode_recency(
                    episodes[all_rankable], args.recency_floor, args.recency_power
                ),
                all_groups,
            ),
            feature_name=selected_names,
            categorical_feature=categorical,
        )
        compact_ranker = compact_booster(final_ranker.booster_, "ranker")
        compact_ranker.update({
            "tree_count": best_ranker_trees,
            "tree_count_selected_by": "validation_nonforced_semantic_exact",
            "runtime_scope": "all_select_contexts",
            "teacher_team": "Majkel1337",
            "teacher_submission_id": 55137818,
            "teacher_trajectories": int(len(np.unique(episodes))),
            "training_decisions": int(len(all_rankable)),
            "semantic_duplicate_tolerant": True,
            "label_definition": (
                "turn_order_graded" if args.label == "graded" else "binary_chosen"
            ),
            "recency_floor": args.recency_floor,
            "recency_power": args.recency_power,
        })
        ranker_path = args.agent_dir / "ranker_model.json"
        ranker_path.write_text(
            json.dumps(compact_ranker, separators=(",", ":")), encoding="utf-8"
        )

        all_variable = all_decisions[
            arrays["minimums"] < arrays["maximums"]
        ]
        final_count = lgb.LGBMRegressor(
            **_count_params(args.seed, best_count_trees)
        )
        final_count.fit(
            count_features[all_variable],
            arrays["chosen_counts"][all_variable],
            sample_weight=_episode_recency(
                episodes[all_variable], args.recency_floor, args.recency_power
            ),
            feature_name=count_feature_names,
            categorical_feature=_categorical_columns(count_feature_names),
        )
        compact_count = compact_booster(final_count.booster_, "regressor")
        compact_count.update({
            "tree_count": best_count_trees,
            "tree_count_selected_by": "validation_variable_count_accuracy",
            "runtime_scope": "variable_pick_count",
            "training_decisions": int(len(all_variable)),
        })
        count_path = args.agent_dir / "count_model.json"
        count_path.write_text(
            json.dumps(compact_count, separators=(",", ":")), encoding="utf-8"
        )
        for path, role, trees in (
            (ranker_path, "candidate_ranker", best_ranker_trees),
            (count_path, "pick_count_regressor", best_count_trees),
        ):
            exported.append({
                "file": path.name,
                "role": role,
                "trees": trees,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })

    report = {
        "cache": str(args.cache.resolve()),
        "agent_dir": str(args.agent_dir.resolve()),
        "seed": args.seed,
        "label": args.label,
        "graded_label_counts": graded_counts,
        "split_decisions": {
            split: int(len(values)) for split, values in decisions.items()
        },
        "rankable_train_decisions": int(len(rankable["train"])),
        "variable_count_train_decisions": int(len(variable_train)),
        "features": {
            "raw": len(feature_names),
            "varying_on_train": len(selected_names),
            "categorical": len(categorical),
        },
        "recency": {"floor": args.recency_floor, "power": args.recency_power},
        "selection": {
            "ranker": {
                "budget": args.ranker_trees,
                "selected_trees": best_ranker_trees,
                "selected_by": "validation_nonforced_semantic_exact",
                "curve": ranker_curve,
            },
            "count": {
                "budget": args.count_trees,
                "selected_trees": best_count_trees,
                "selected_by": "validation_variable_count_accuracy",
                "curve": count_curve,
            },
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_baselines": baselines,
        "target": {
            "metric": "nonforced_semantic_exact",
            "value": 0.90,
            "met_on_test": bool(test_metrics["nonforced_semantic_exact"] >= 0.90),
        },
        "exported": exported,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "validation": validation_metrics,
        "test": test_metrics,
        "target": report["target"],
        "exported": exported,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
