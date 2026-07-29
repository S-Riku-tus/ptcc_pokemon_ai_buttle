"""Rerank the teacher ranker's top candidates with explicit pair comparisons.

LambdaRank learns a separable score f(state, candidate).  Expert sequencing
often depends on a direct comparison such as "play card A before card B".
This probe trains g(state, left, right) and uses a small round-robin among the
base model's top candidates.  The split remains chronological and frozen.
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
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


CRITICAL_CANDIDATE_NAMES = (
    "option_type",
    "candidate_card_id",
    "candidate_attack_id",
    "candidate_area",
    "candidate_inplay_area",
    "candidate_target_id",
    "action_type",
    "fallback_selected",
    "v29_selected",
)


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    return starts, ends


def _pair_schema(
    feature_names: list[str],
) -> tuple[int, list[int], list[str]]:
    candidate_start = feature_names.index("option_type")
    critical = [
        feature_names.index(name)
        for name in CRITICAL_CANDIDATE_NAMES
        if name in feature_names
    ]
    names = (
        [f"state__{name}" for name in feature_names[:candidate_start]]
        + [f"delta__{name}" for name in feature_names[candidate_start:]]
        + [f"left__{feature_names[index]}" for index in critical]
        + [f"right__{feature_names[index]}" for index in critical]
        + [
            "base_left",
            "base_right",
            "base_delta",
            "base_rank_left",
            "base_rank_right",
        ]
    )
    return candidate_start, critical, names


def _pair_row(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_score: float,
    right_score: float,
    left_rank: int,
    right_rank: int,
    candidate_start: int,
    critical: list[int],
) -> np.ndarray:
    return np.concatenate((
        left[:candidate_start],
        left[candidate_start:] - right[candidate_start:],
        left[critical],
        right[critical],
        np.asarray(
            [
                left_score,
                right_score,
                left_score - right_score,
                left_rank,
                right_rank,
            ],
            dtype=np.float32,
        ),
    ))


def _training_pairs(
    arrays: dict[str, Any],
    decision_indices: np.ndarray,
    base_scores: np.ndarray,
    *,
    top_k: int,
    candidate_start: int,
    critical: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends = _ranges(arrays["groups"])
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for decision in decision_indices:
        start, end = starts[decision], ends[decision]
        labels_in_group = arrays["labels"][start:end]
        positive = int(np.flatnonzero(labels_in_group == 1)[0])
        order = np.argsort(-base_scores[start:end], kind="stable")
        ranks = np.empty(len(order), dtype=np.int32)
        ranks[order] = np.arange(len(order))
        negatives = [
            int(index)
            for index in order
            if int(index) != positive
        ][:top_k]
        decision_weight = float(arrays["weights"][start])
        for negative in negatives:
            positive_row = arrays["features"][start + positive]
            negative_row = arrays["features"][start + negative]
            positive_score = float(base_scores[start + positive])
            negative_score = float(base_scores[start + negative])
            rows.append(_pair_row(
                positive_row,
                negative_row,
                left_score=positive_score,
                right_score=negative_score,
                left_rank=int(ranks[positive]),
                right_rank=int(ranks[negative]),
                candidate_start=candidate_start,
                critical=critical,
            ))
            labels.append(1)
            weights.append(decision_weight)
            rows.append(_pair_row(
                negative_row,
                positive_row,
                left_score=negative_score,
                right_score=positive_score,
                left_rank=int(ranks[negative]),
                right_rank=int(ranks[positive]),
                candidate_start=candidate_start,
                critical=critical,
            ))
            labels.append(0)
            weights.append(decision_weight)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float32),
    )


def _fit_pair_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str],
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=127,
        min_child_samples=35,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=741,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        x,
        y,
        sample_weight=weights,
        feature_name=feature_names,
    )
    return model


def _candidate_scores(
    model: lgb.LGBMClassifier,
    arrays: dict[str, Any],
    decision_indices: np.ndarray,
    base_scores: np.ndarray,
    *,
    top_k: int,
    candidate_start: int,
    critical: list[int],
) -> list[dict[str, Any]]:
    starts, ends = _ranges(arrays["groups"])
    decisions = []
    for decision in decision_indices:
        start, end = starts[decision], ends[decision]
        group_scores = base_scores[start:end]
        order = np.argsort(-group_scores, kind="stable")[:top_k]
        ranks = np.empty(end - start, dtype=np.int32)
        ranks[np.argsort(-group_scores, kind="stable")] = np.arange(end - start)
        pair_rows = []
        pair_coordinates = []
        for left_position, left in enumerate(order):
            for right_position, right in enumerate(order):
                if left_position >= right_position:
                    continue
                pair_rows.append(_pair_row(
                    arrays["features"][start + left],
                    arrays["features"][start + right],
                    left_score=float(group_scores[left]),
                    right_score=float(group_scores[right]),
                    left_rank=int(ranks[left]),
                    right_rank=int(ranks[right]),
                    candidate_start=candidate_start,
                    critical=critical,
                ))
                pair_coordinates.append((left_position, right_position))
        pair_totals = np.zeros(len(order), dtype=np.float32)
        if pair_rows:
            probabilities = model.predict_proba(
                np.asarray(pair_rows, dtype=np.float32)
            )[:, 1]
            for probability, (left, right) in zip(
                probabilities, pair_coordinates
            ):
                probability = float(
                    np.clip(probability, 1e-5, 1.0 - 1e-5)
                )
                logit = np.log(probability / (1.0 - probability))
                pair_totals[left] += logit
                pair_totals[right] -= logit
        base_top = group_scores[order].astype(np.float32)
        base_top = (
            (base_top - float(base_top.mean()))
            / max(float(base_top.std()), 1e-5)
        )
        decisions.append({
            "labels": arrays["labels"][start:end][order],
            "base": base_top,
            "pair": pair_totals,
            "base_top1": int(arrays["labels"][start + order[0]] == 1),
        })
    return decisions


def _accuracy(
    decisions: list[dict[str, Any]],
    alpha: float,
) -> float:
    return float(np.mean([
        int(
            row["labels"][
                int(np.argmax(row["pair"] + alpha * row["base"]))
            ]
            == 1
        )
        for row in decisions
    ]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays: dict[str, Any] = {
            key: cached[key]
            for key in (
                "features",
                "labels",
                "weights",
                "groups",
                "fallback_correct",
                "teacher_action_types",
            )
        }
        arrays["splits"] = cached["splits"].astype(str)
        feature_names = cached["feature_names"].astype(str).tolist()
    split_values = np.asarray(arrays["splits"])
    train_indices = np.flatnonzero(split_values == "train")
    validation_indices = np.flatnonzero(split_values == "validation")
    test_indices = np.flatnonzero(split_values == "test")

    ranker = teacher._fit(
        arrays,
        feature_names,
        train_indices,
        n_estimators=900,
        validation_indices=validation_indices,
    )
    base_scores = ranker.predict(arrays["features"]).astype(np.float32)
    candidate_start, critical, pair_names = _pair_schema(feature_names)
    pair_x, pair_y, pair_weights = _training_pairs(
        arrays,
        train_indices,
        base_scores,
        top_k=args.top_k,
        candidate_start=candidate_start,
        critical=critical,
    )
    pair_model = _fit_pair_model(
        pair_x,
        pair_y,
        pair_weights,
        pair_names,
    )
    validation = _candidate_scores(
        pair_model,
        arrays,
        validation_indices,
        base_scores,
        top_k=args.top_k,
        candidate_start=candidate_start,
        critical=critical,
    )
    test = _candidate_scores(
        pair_model,
        arrays,
        test_indices,
        base_scores,
        top_k=args.top_k,
        candidate_start=candidate_start,
        critical=critical,
    )
    alpha_grid = np.arange(0.0, 3.01, 0.10)
    validation_grid = [
        {"alpha": float(alpha), "top1": _accuracy(validation, float(alpha))}
        for alpha in alpha_grid
    ]
    best = max(validation_grid, key=lambda row: row["top1"])
    report = {
        "cache": str(args.cache.resolve()),
        "top_k": args.top_k,
        "pair_rows": int(len(pair_y)),
        "pair_features": len(pair_names),
        "base_ranker_best_iteration": int(ranker.best_iteration_ or 900),
        "validation_base_top1": float(np.mean([
            row["base_top1"] for row in validation
        ])),
        "test_base_top1": float(np.mean([
            row["base_top1"] for row in test
        ])),
        "validation_grid": validation_grid,
        "selected_alpha": best["alpha"],
        "validation_pairwise_top1": best["top1"],
        "test_pairwise_top1": _accuracy(test, float(best["alpha"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
