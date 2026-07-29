"""Add the set-attention checkpoint to the current v32 score ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_deepset_blend as deep_blend  # noqa: E402


def _oracle(
    scores: list[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    starts, ends = ensemble._ranges(groups)
    return float(np.mean([
        any(
            labels[start + int(np.argmax(values[start:end]))] == 1
            for values in scores
        )
        for start, end in zip(starts, ends)
    ]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
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
        model_names = ["large", "numeric", "deep"]
        if "validation_history" in saved.files:
            val_scores.append(saved["validation_history"])
            test_scores.append(saved["test_history"])
            model_names.append("history")
        val_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        val_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        names = cached["feature_names"].astype(str).tolist()
        groups = cached["groups"]
        labels = cached["labels"]
        weights = cached["weights"]
        splits = cached["splits"].astype(str)
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    combined = deep_blend._deep_scores(
        args.checkpoint,
        features,
        names,
        groups,
        labels,
        weights,
        np.concatenate([validation, test]),
        128,
    )
    validation_rows = int(val_groups.sum())
    val_attention = ensemble._normalize(
        combined[:validation_rows], val_groups.tolist()
    )
    test_attention = ensemble._normalize(
        combined[validation_rows:], test_groups.tolist()
    )
    val_scores.append(val_attention)
    test_scores.append(test_attention)
    model_names.append("attention")
    attention_metrics = {
        "validation_top1": ensemble._accuracy(
            val_attention, val_labels, val_groups.tolist()
        ),
        "test_top1": ensemble._accuracy(
            test_attention, test_labels, test_groups.tolist()
        ),
    }
    print(attention_metrics, flush=True)

    # Coordinate search starts from the validation-selected four-model blend.
    weights_vector = (
        np.asarray([1.0, 0.5, 0.6, 1.1, 0.0])
        if len(val_scores) == 5
        else np.asarray([1.0, 0.5, 1.45, 0.0])
    )
    best = 0.0
    for _ in range(4):
        changed = False
        for coordinate in range(1, len(weights_vector)):
            selected_value = weights_vector[coordinate]
            selected_accuracy = -1.0
            for value in np.arange(0.0, 2.01, 0.05):
                candidate = weights_vector.copy()
                candidate[coordinate] = value
                blended = sum(
                    float(weight) * scores
                    for weight, scores in zip(candidate, val_scores)
                )
                accuracy = ensemble._accuracy(
                    blended, val_labels, val_groups.tolist()
                )
                if accuracy > selected_accuracy:
                    selected_accuracy = accuracy
                    selected_value = float(value)
            if selected_value != weights_vector[coordinate]:
                changed = True
            weights_vector[coordinate] = selected_value
            best = selected_accuracy
        if not changed:
            break
    test_blended = sum(
        float(weight) * scores
        for weight, scores in zip(weights_vector, test_scores)
    )
    test_top1 = ensemble._accuracy(
        test_blended, test_labels, test_groups.tolist()
    )
    report = {
        "attention_model": attention_metrics,
        "model_order": model_names,
        "selected_weights": weights_vector.tolist(),
        "validation_top1": best,
        "test_top1": test_top1,
        "validation_oracle_any_model": _oracle(
            val_scores, val_labels, val_groups
        ),
        "test_oracle_any_model": _oracle(
            test_scores, test_labels, test_groups
        ),
        "previous_v32_blend_test_top1": 0.7708333333333334,
        "target_top1": 0.9,
        "target_met": test_top1 >= 0.9,
    }
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        **{
            f"validation_{name}": scores
            for name, scores in zip(model_names, val_scores)
        },
        validation_labels=val_labels,
        validation_groups=val_groups,
        **{
            f"test_{name}": scores
            for name, scores in zip(model_names, test_scores)
        },
        test_labels=test_labels,
        test_groups=test_groups,
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
