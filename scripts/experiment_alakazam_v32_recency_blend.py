"""Blend the strongest recency challenger into the frozen v32 ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_attention_blend as attention  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_scores", type=Path)
    parser.add_argument("recency_scores", type=Path)
    parser.add_argument("--recency-name", default="weighted_floor_0.1_power_1.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.base_scores, allow_pickle=False) as base:
        model_names = [
            "large",
            "numeric",
            "deep",
            "history",
            "attention",
        ]
        validation_scores = [
            base[f"validation_{name}"] for name in model_names
        ]
        test_scores = [
            base[f"test_{name}"] for name in model_names
        ]
        validation_labels = base["validation_labels"]
        test_labels = base["test_labels"]
        validation_groups = base["validation_groups"]
        test_groups = base["test_groups"]
    with np.load(args.recency_scores, allow_pickle=False) as recency:
        names = recency["names"].astype(str).tolist()
        recency_index = names.index(args.recency_name)
        validation_scores.append(
            recency["validation_scores"][recency_index]
        )
        test_scores.append(recency["test_scores"][recency_index])
    model_names.append(args.recency_name)

    weights = np.asarray([1.0, 0.55, 0.60, 1.10, 0.0, 0.0])
    best = ensemble._accuracy(
        sum(
            float(weight) * scores
            for weight, scores in zip(weights, validation_scores)
        ),
        validation_labels,
        validation_groups.tolist(),
    )
    for _ in range(6):
        changed = False
        for coordinate in range(1, len(weights)):
            selected_value = float(weights[coordinate])
            selected_accuracy = best
            for value in np.arange(0.0, 2.001, 0.025):
                candidate = weights.copy()
                candidate[coordinate] = value
                accuracy = ensemble._accuracy(
                    sum(
                        float(weight) * scores
                        for weight, scores in zip(
                            candidate,
                            validation_scores,
                        )
                    ),
                    validation_labels,
                    validation_groups.tolist(),
                )
                if accuracy > selected_accuracy:
                    selected_accuracy = accuracy
                    selected_value = float(value)
            if selected_value != weights[coordinate]:
                changed = True
            weights[coordinate] = selected_value
            best = selected_accuracy
        if not changed:
            break

    test_blend = sum(
        float(weight) * scores
        for weight, scores in zip(weights, test_scores)
    )
    test_top1 = ensemble._accuracy(
        test_blend,
        test_labels,
        test_groups.tolist(),
    )
    report = {
        "model_order": model_names,
        "selected_weights": weights.tolist(),
        "validation_top1": best,
        "test_top1": test_top1,
        "validation_oracle_any_model": attention._oracle(
            validation_scores,
            validation_labels,
            validation_groups,
        ),
        "test_oracle_any_model": attention._oracle(
            test_scores,
            test_labels,
            test_groups,
        ),
        "previous_v32_best_test_top1": 0.7715818363273453,
        "target_top1": 0.9,
        "target_met": test_top1 >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.scores_output,
        **{
            f"validation_{name}": scores
            for name, scores in zip(model_names, validation_scores)
        },
        **{
            f"test_{name}": scores
            for name, scores in zip(model_names, test_scores)
        },
        validation_labels=validation_labels,
        validation_groups=validation_groups,
        test_labels=test_labels,
        test_groups=test_groups,
        selected_weights=weights,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
