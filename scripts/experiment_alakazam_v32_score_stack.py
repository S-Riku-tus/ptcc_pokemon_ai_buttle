"""Train a compact out-of-sample score stack for the v32 model trio."""

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

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponentials = np.exp(np.clip(shifted, -30.0, 30.0))
    return exponentials / max(float(exponentials.sum()), 1e-12)


def _features(
    scores: list[np.ndarray],
    groups: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    starts, ends = ensemble._ranges(groups)
    names = [
        value
        for model in range(len(scores))
        for value in (
            f"model_{model}_score",
            f"model_{model}_rank",
            f"model_{model}_rank_fraction",
            f"model_{model}_selected",
            f"model_{model}_group_margin",
            f"model_{model}_group_entropy",
        )
    ] + [
        "score_mean",
        "score_std",
        "score_max",
        "score_min",
        "large_minus_numeric",
        "large_minus_deep",
        "numeric_minus_deep",
        "top_vote_count",
        "candidate_position",
        "candidate_position_fraction",
        "group_size",
    ]
    output = np.zeros((len(scores[0]), len(names)), dtype=np.float32)
    for start, end in zip(starts, ends):
        count = end - start
        selected = []
        for model_index, model_scores in enumerate(scores):
            values = model_scores[start:end]
            order = np.argsort(-values, kind="stable")
            ranks = np.empty(count, dtype=np.float32)
            ranks[order] = np.arange(count, dtype=np.float32)
            probabilities = _softmax(values)
            entropy = float(
                -(probabilities * np.log(
                    np.maximum(probabilities, 1e-12)
                )).sum()
            )
            margin = (
                float(values[order[0]] - values[order[1]])
                if count > 1 else 0.0
            )
            offset = model_index * 6
            output[start:end, offset] = values
            output[start:end, offset + 1] = ranks
            output[start:end, offset + 2] = ranks / max(count - 1, 1)
            output[start:end, offset + 3] = (ranks == 0)
            output[start:end, offset + 4] = margin
            output[start:end, offset + 5] = entropy
            selected.append(int(order[0]))
        score_values = np.column_stack([
            model_scores[start:end] for model_scores in scores
        ])
        base = len(scores) * 6
        output[start:end, base] = score_values.mean(axis=1)
        output[start:end, base + 1] = score_values.std(axis=1)
        output[start:end, base + 2] = score_values.max(axis=1)
        output[start:end, base + 3] = score_values.min(axis=1)
        output[start:end, base + 4] = score_values[:, 0] - score_values[:, 1]
        output[start:end, base + 5] = score_values[:, 0] - score_values[:, 2]
        output[start:end, base + 6] = score_values[:, 1] - score_values[:, 2]
        votes = np.zeros(count, dtype=np.float32)
        for local in selected:
            votes[local] += 1.0
        output[start:end, base + 7] = votes
        positions = np.arange(count, dtype=np.float32)
        output[start:end, base + 8] = positions
        output[start:end, base + 9] = positions / max(count - 1, 1)
        output[start:end, base + 10] = count
    return output, names


def _fit(
    arrays: dict[str, Any],
    names: list[str],
    train: np.ndarray,
    validation: np.ndarray | None,
    *,
    leaves: int,
    minimum: int,
    iterations: int,
) -> lgb.LGBMRanker:
    x, y, weights, groups = teacher._select_decisions(arrays, train)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=iterations,
        learning_rate=0.025,
        num_leaves=leaves,
        min_child_samples=minimum,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.4,
        reg_lambda=2.0,
        random_state=743,
        n_jobs=4,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {}
    if validation is not None:
        vx, vy, _, validation_groups = teacher._select_decisions(
            arrays, validation
        )
        fit_kwargs.update({
            "eval_set": [(vx, vy)],
            "eval_group": [validation_groups],
            "callbacks": [lgb.early_stopping(40, verbose=False)],
        })
    model.fit(
        x,
        y,
        group=groups,
        sample_weight=weights,
        feature_name=names,
        **fit_kwargs,
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.scores, allow_pickle=False) as saved:
        val_scores = [
            saved["validation_large"],
            saved["validation_numeric"],
            saved["validation_deep"],
        ]
        test_scores = [
            saved["test_large"],
            saved["test_numeric"],
            saved["test_deep"],
        ]
        val_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        val_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
    with np.load(args.cache, allow_pickle=False) as cached:
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
    validation_decisions = np.flatnonzero(splits == "validation")
    validation_episodes = episode_ids[validation_decisions]
    ordered = np.unique(validation_episodes)
    ordered.sort()
    cut = int(len(ordered) * 0.60)
    train_episodes = set(ordered[:cut].tolist())
    meta_train = np.flatnonzero(np.asarray([
        episode in train_episodes for episode in validation_episodes
    ]))
    meta_validation = np.flatnonzero(np.asarray([
        episode not in train_episodes for episode in validation_episodes
    ]))

    val_x, names = _features(val_scores, val_groups)
    test_x, _ = _features(test_scores, test_groups)
    val_arrays: dict[str, Any] = {
        "features": val_x,
        "labels": val_labels,
        "weights": np.ones(len(val_labels), dtype=np.float32),
        "groups": val_groups,
    }
    experiments = []
    for leaves, minimum in ((7, 25), (15, 25), (31, 30), (63, 35)):
        model = _fit(
            val_arrays,
            names,
            meta_train,
            meta_validation,
            leaves=leaves,
            minimum=minimum,
            iterations=600,
        )
        vx, vy, _, vg = teacher._select_decisions(
            val_arrays, meta_validation
        )
        experiments.append({
            "leaves": leaves,
            "min_child_samples": minimum,
            "best_iteration": int(model.best_iteration_ or 600),
            "meta_validation_top1": ensemble._accuracy(
                model.predict(vx), vy, vg
            ),
        })
        print(experiments[-1], flush=True)
    selected = max(
        experiments,
        key=lambda row: (
            row["meta_validation_top1"],
            -row["leaves"],
        ),
    )
    final_model = _fit(
        val_arrays,
        names,
        np.arange(len(val_groups), dtype=np.int64),
        None,
        leaves=selected["leaves"],
        minimum=selected["min_child_samples"],
        iterations=selected["best_iteration"],
    )
    test_top1 = ensemble._accuracy(
        final_model.predict(test_x),
        test_labels,
        test_groups.tolist(),
    )
    report = {
        "meta_train_decisions": len(meta_train),
        "meta_validation_decisions": len(meta_validation),
        "features": len(names),
        "experiments": experiments,
        "selected": selected,
        "test_top1": test_top1,
        "v32_weighted_blend_reference": 0.7699600798403193,
        "target_top1": 0.9,
        "target_met": test_top1 >= 0.9,
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
