"""Add one precomputed leakage-free score vector to a v32 ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_attention_blend as blend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("new_scores", type=Path)
    parser.add_argument("--new-index", type=int, required=True)
    parser.add_argument("--new-name", required=True)
    parser.add_argument(
        "--anchor-new",
        action="store_true",
        help="Fix the new score at weight 1 and allow the old anchor to vary.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()

    previous = json.loads(args.blend_report.read_text(encoding="utf-8"))
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
    with np.load(args.new_scores, allow_pickle=False) as added:
        validation_new = added["validation_scores"][args.new_index]
        test_new = added["test_scores"][args.new_index]
        source_name = str(added["names"][args.new_index])
    validation_scores.append(validation_new)
    test_scores.append(test_new)
    model_names.append(args.new_name)
    anchor = 0
    if args.anchor_new:
        weights[0] = 0.0
        weights[-1] = 1.0
        anchor = len(weights) - 1

    def score(candidate: np.ndarray) -> float:
        return ensemble._accuracy(
            sum(
                float(weight) * values
                for weight, values in zip(candidate, validation_scores)
            ),
            validation_labels,
            validation_groups.tolist(),
        )

    best = score(weights)
    for step in (0.05, 0.025, 0.01):
        for _ in range(8):
            changed = False
            for coordinate in range(len(weights)):
                if coordinate == anchor:
                    continue
                current = float(weights[coordinate])
                candidates = np.unique(np.clip(
                    np.r_[
                        np.arange(0.0, 2.001, step),
                        current,
                    ],
                    0.0,
                    2.0,
                ))
                selected = current
                selected_accuracy = best
                for value in candidates:
                    candidate = weights.copy()
                    candidate[coordinate] = float(value)
                    accuracy = score(candidate)
                    if accuracy > selected_accuracy:
                        selected_accuracy = accuracy
                        selected = float(value)
                if selected != current:
                    changed = True
                weights[coordinate] = selected
                best = selected_accuracy
            if not changed:
                break

    test_top1 = ensemble._accuracy(
        sum(
            float(weight) * values
            for weight, values in zip(weights, test_scores)
        ),
        test_labels,
        test_groups.tolist(),
    )
    report = {
        "new_model": {
            "name": args.new_name,
            "source_name": source_name,
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
            f"validation_{name}": values
            for name, values in zip(model_names, validation_scores)
        },
        **{
            f"test_{name}": values
            for name, values in zip(model_names, test_scores)
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
