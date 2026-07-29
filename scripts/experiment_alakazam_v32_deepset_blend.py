"""Blend the v32 DeepSets checkpoint with leakage-free v31 tree rankers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_deepset as deepset  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _deep_scores(
    checkpoint_path: Path,
    features: np.ndarray,
    feature_names: list[str],
    groups: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    decisions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    continuous_columns = [
        feature_names.index(name)
        for name in checkpoint["continuous_feature_names"]
    ]
    categorical_columns = [
        feature_names.index(name)
        for name in checkpoint["categorical_feature_names"]
    ]
    mean = checkpoint["continuous_mean"]
    std = checkpoint["continuous_std"]
    continuous = np.empty(
        (len(features), len(continuous_columns)),
        dtype=np.float16,
    )
    for start in range(0, len(features), 50_000):
        end = min(start + 50_000, len(features))
        values = (
            features[start:end, continuous_columns].astype(np.float32)
            - mean
        ) / std
        continuous[start:end] = np.clip(
            values, -8.0, 8.0
        ).astype(np.float16)
    categorical = np.zeros(
        (len(features), len(categorical_columns)),
        dtype=np.int16,
    )
    for output_column, (source_column, vocabulary) in enumerate(zip(
        categorical_columns,
        checkpoint["categorical_vocabularies"],
    )):
        raw = features[:, source_column]
        positions = np.searchsorted(vocabulary, raw)
        clipped = np.minimum(positions, len(vocabulary) - 1)
        known = (
            (positions < len(vocabulary))
            & (vocabulary[clipped] == raw)
        )
        categorical[:, output_column] = np.where(
            known,
            positions + 1,
            len(vocabulary) + 1,
        ).astype(np.int16)
    store = deepset.GroupStore(
        continuous,
        categorical,
        groups,
        labels,
        weights,
    )
    model = (
        deepset.SetAttentionPolicy(
            len(continuous_columns),
            checkpoint["categorical_sizes"],
            checkpoint["hidden"],
            checkpoint["dropout"],
            checkpoint.get("attention_layers", 2),
        )
        if checkpoint.get("architecture") == "attention"
        else deepset.DeepSetPolicy(
            len(continuous_columns),
            checkpoint["categorical_sizes"],
            checkpoint["hidden"],
            checkpoint["dropout"],
        )
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            batch_decisions = decisions[start:start + batch_size]
            continuous_batch, categorical_batch, mask, _, _ = store.batch(
                batch_decisions, torch.device("cpu")
            )
            scores = model(
                continuous_batch,
                categorical_batch,
                mask,
            ).numpy()
            for row, decision in enumerate(batch_decisions):
                output.append(scores[row, :groups[decision]].copy())
    return np.concatenate(output).astype(np.float32)


def _oracle(
    score_sets: list[np.ndarray],
    labels: np.ndarray,
    groups: list[int],
) -> float:
    starts, ends = ensemble._ranges(groups)
    correct = 0
    for start, end in zip(starts, ends):
        correct += int(any(
            labels[start + int(np.argmax(scores[start:end]))] == 1
            for scores in score_sets
        ))
    return correct / len(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    with np.load(args.schema_cache, allow_pickle=False) as schema:
        desired = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        original_names = cached["feature_names"].astype(str).tolist()
        columns = [original_names.index(name) for name in desired]
        arrays: dict[str, Any] = {
            "features": cached["features"][:, columns],
            "labels": cached["labels"],
            "weights": cached["weights"],
            "groups": cached["groups"],
        }
        splits = cached["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    train_data = teacher._select_decisions(arrays, train)
    validation_data = teacher._select_decisions(arrays, validation)
    test_data = teacher._select_decisions(arrays, test)
    categorical = [
        index
        for index, name in enumerate(desired)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ]
    configs = [
        {
            "name": "lambda_large_leaf",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.025,
            "num_leaves": 255,
            "min_child_samples": 55,
            "colsample_bytree": 0.80,
            "seed": 1086,
        },
        {
            "name": "lambda_numeric_ids",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 35,
            "categorical": False,
            "seed": 305,
        },
    ]
    val_x, val_y, _, val_groups = validation_data
    test_x, test_y, _, test_groups = test_data
    val_tree_scores = []
    test_tree_scores = []
    tree_rows = []
    for config in configs:
        model = ensemble._ranker(
            config,
            train_data,
            validation_data,
            desired,
            categorical,
        )
        val_scores = ensemble._normalize(
            model.predict(val_x), val_groups
        )
        test_scores = ensemble._normalize(
            model.predict(test_x), test_groups
        )
        val_tree_scores.append(val_scores)
        test_tree_scores.append(test_scores)
        row = {
            "name": config["name"],
            "best_iteration": int(model.best_iteration_ or 1200),
            "validation_top1": ensemble._accuracy(
                val_scores, val_y, val_groups
            ),
            "test_top1": ensemble._accuracy(
                test_scores, test_y, test_groups
            ),
        }
        tree_rows.append(row)
        print(json.dumps(row), flush=True)

    deep_scores = _deep_scores(
        args.checkpoint,
        arrays["features"],
        desired,
        arrays["groups"],
        arrays["labels"],
        arrays["weights"],
        np.concatenate([validation, test]),
        128,
    )
    validation_rows = int(np.sum(val_groups))
    val_deep = ensemble._normalize(
        deep_scores[:validation_rows],
        val_groups,
    )
    test_deep = ensemble._normalize(
        deep_scores[validation_rows:],
        test_groups,
    )
    deep_row = {
        "name": "deepset",
        "validation_top1": ensemble._accuracy(
            val_deep, val_y, val_groups
        ),
        "test_top1": ensemble._accuracy(
            test_deep, test_y, test_groups
        ),
    }
    print(json.dumps(deep_row), flush=True)

    grid = []
    for numeric_weight in np.arange(0.5, 2.01, 0.1):
        base_validation = (
            val_tree_scores[0]
            + float(numeric_weight) * val_tree_scores[1]
        )
        base_test = (
            test_tree_scores[0]
            + float(numeric_weight) * test_tree_scores[1]
        )
        for deep_weight in np.arange(0.0, 2.01, 0.05):
            validation_scores = (
                base_validation + float(deep_weight) * val_deep
            )
            grid.append({
                "numeric_weight": float(numeric_weight),
                "deep_weight": float(deep_weight),
                "validation_top1": ensemble._accuracy(
                    validation_scores, val_y, val_groups
                ),
                "_test_scores": (
                    base_test + float(deep_weight) * test_deep
                ),
            })
    selected = max(
        grid,
        key=lambda row: (
            row["validation_top1"],
            -abs(row["numeric_weight"] - 1.3),
            -row["deep_weight"],
        ),
    )
    test_top1 = ensemble._accuracy(
        selected["_test_scores"], test_y, test_groups
    )
    report = {
        "tree_models": tree_rows,
        "deep_model": deep_row,
        "selected": {
            key: value
            for key, value in selected.items()
            if not key.startswith("_")
        },
        "test_top1": test_top1,
        "validation_oracle_any_model": _oracle(
            [*val_tree_scores, val_deep],
            val_y,
            val_groups,
        ),
        "test_oracle_any_model": _oracle(
            [*test_tree_scores, test_deep],
            test_y,
            test_groups,
        ),
        "v31_reference_top1": 0.7630988023952096,
        "target_top1": 0.9,
        "target_met": test_top1 >= 0.9,
    }
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        validation_large=val_tree_scores[0],
        validation_numeric=val_tree_scores[1],
        validation_deep=val_deep,
        validation_labels=val_y,
        validation_groups=np.asarray(val_groups),
        test_large=test_tree_scores[0],
        test_numeric=test_tree_scores[1],
        test_deep=test_deep,
        test_labels=test_y,
        test_groups=np.asarray(test_groups),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
