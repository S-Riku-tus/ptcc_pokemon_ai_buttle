"""Add one neural checkpoint to an existing leakage-free score ensemble."""

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
import experiment_alakazam_v32_attention_blend as blend  # noqa: E402
import experiment_alakazam_v32_deepset_blend as deep_blend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()

    previous = json.loads(
        args.blend_report.read_text(encoding="utf-8")
    )
    model_names = list(previous["model_order"])
    weights = np.asarray(
        previous["selected_weights"] + [0.0],
        dtype=np.float64,
    )
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_scores = [
            saved[f"validation_{name}"] for name in model_names
        ]
        test_scores = [
            saved[f"test_{name}"] for name in model_names
        ]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        names = cached["feature_names"].astype(str).tolist()
        groups = cached["groups"]
        labels = cached["labels"]
        row_weights = cached["weights"]
        splits = cached["splits"].astype(str)
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    combined = deep_blend._deep_scores(
        args.checkpoint,
        features,
        names,
        groups,
        labels,
        row_weights,
        np.concatenate([validation, test]),
        128,
    )
    validation_rows = int(validation_groups.sum())
    validation_new = ensemble._normalize(
        combined[:validation_rows],
        validation_groups.tolist(),
    )
    test_new = ensemble._normalize(
        combined[validation_rows:],
        test_groups.tolist(),
    )
    validation_scores.append(validation_new)
    test_scores.append(test_new)
    model_names.append(args.checkpoint_name)

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

    test_top1 = ensemble._accuracy(
        sum(
            float(weight) * scores
            for weight, scores in zip(weights, test_scores)
        ),
        test_labels,
        test_groups.tolist(),
    )
    report = {
        "new_model": {
            "name": args.checkpoint_name,
            "validation_top1": ensemble._accuracy(
                validation_new,
                validation_labels,
                validation_groups.tolist(),
            ),
            "test_top1": ensemble._accuracy(
                test_new,
                test_labels,
                test_groups.tolist(),
            ),
        },
        "model_order": model_names,
        "selected_weights": weights.tolist(),
        "validation_top1": best,
        "test_top1": test_top1,
        "validation_oracle_any_model": blend._oracle(
            validation_scores,
            validation_labels,
            validation_groups,
        ),
        "test_oracle_any_model": blend._oracle(
            test_scores,
            test_labels,
            test_groups,
        ),
        "previous_test_top1": previous["test_top1"],
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
