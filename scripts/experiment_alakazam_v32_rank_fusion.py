"""Compare validation-selected rank/probability fusion for frozen v32 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402


def _transform(
    scores: list[np.ndarray],
    groups: np.ndarray,
    kind: str,
    parameter: float,
) -> list[np.ndarray]:
    starts, ends = ensemble._ranges(groups)
    transformed = [np.empty_like(values) for values in scores]
    for start, end in zip(starts, ends):
        size = end - start
        for index, source in enumerate(scores):
            values = source[start:end]
            if kind == "softmax":
                shifted = np.clip(
                    (values - float(values.max())) / parameter,
                    -40.0,
                    0.0,
                )
                result = np.exp(shifted)
                result /= max(float(result.sum()), 1e-9)
            else:
                ranks = np.argsort(
                    np.argsort(-values, kind="stable"),
                    kind="stable",
                ).astype(np.float32)
                if kind == "percentile":
                    result = 1.0 - ranks / max(size - 1, 1)
                elif kind == "reciprocal":
                    result = 1.0 / (ranks + parameter)
                elif kind == "exponential_rank":
                    result = np.exp(-ranks / parameter)
                else:
                    raise ValueError(kind)
            transformed[index][start:end] = result
    return transformed


def _blend(score_sets: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(
        float(weight) * scores
        for weight, scores in zip(weights, score_sets)
    )


def _tune(
    score_sets: list[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, float]:
    weights = initial.astype(np.float64).copy()
    best = ensemble._accuracy(
        _blend(score_sets, weights),
        labels,
        groups.tolist(),
    )
    grid = np.arange(0.0, 2.51, 0.10)
    for _ in range(2):
        changed = False
        for index in range(len(weights)):
            candidate_best = (best, -float(weights[index]), weights[index])
            for value in grid:
                trial = weights.copy()
                trial[index] = float(value)
                accuracy = ensemble._accuracy(
                    _blend(score_sets, trial),
                    labels,
                    groups.tolist(),
                )
                candidate_best = max(
                    candidate_best,
                    (accuracy, -float(value), float(value)),
                )
            if candidate_best[0] > best:
                changed = True
            best = candidate_best[0]
            weights[index] = candidate_best[2]
        if not changed:
            break
    return weights, best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_report = json.loads(
        args.blend_report.read_text(encoding="utf-8")
    )
    names = list(base_report["model_order"])
    fixed = np.asarray(
        base_report["selected_weights"],
        dtype=np.float64,
    )
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_scores = [
            saved[f"validation_{name}"] for name in names
        ]
        test_scores = [saved[f"test_{name}"] for name in names]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]

    definitions: list[tuple[str, float]] = [
        ("percentile", 1.0),
        ("reciprocal", 0.5),
        ("reciprocal", 1.0),
        ("reciprocal", 2.0),
        ("reciprocal", 5.0),
        ("exponential_rank", 0.5),
        ("exponential_rank", 1.0),
        ("exponential_rank", 2.0),
        ("exponential_rank", 5.0),
        ("softmax", 0.25),
        ("softmax", 0.5),
        ("softmax", 1.0),
        ("softmax", 2.0),
        ("softmax", 4.0),
    ]
    transformed_validation: dict[
        tuple[str, float],
        list[np.ndarray],
    ] = {}
    candidates = [{
        "kind": "raw",
        "parameter": 1.0,
        "weight_source": "fixed",
        "weights": fixed.tolist(),
        "validation_top1": float(base_report["validation_top1"]),
    }]
    for kind, parameter in definitions:
        score_sets = _transform(
            validation_scores,
            validation_groups,
            kind,
            parameter,
        )
        transformed_validation[(kind, parameter)] = score_sets
        for weight_name, weights in (
            ("uniform", np.ones(len(names), dtype=np.float64)),
            ("fixed", fixed),
        ):
            candidates.append({
                "kind": kind,
                "parameter": parameter,
                "weight_source": weight_name,
                "weights": weights.tolist(),
                "validation_top1": ensemble._accuracy(
                    _blend(score_sets, weights),
                    validation_labels,
                    validation_groups.tolist(),
                ),
            })

    top_definitions = []
    for row in sorted(
        candidates,
        key=lambda item: -item["validation_top1"],
    ):
        key = (str(row["kind"]), float(row["parameter"]))
        if key[0] == "raw":
            continue
        if key not in top_definitions:
            top_definitions.append(key)
        if len(top_definitions) == 3:
            break
    tuned = []
    for key in top_definitions:
        weights, accuracy = _tune(
            transformed_validation[key],
            validation_labels,
            validation_groups,
            fixed,
        )
        tuned.append({
            "kind": key[0],
            "parameter": key[1],
            "weight_source": "coordinate_validation",
            "weights": weights.tolist(),
            "validation_top1": accuracy,
        })
    selected = max(
        [*candidates, *tuned],
        key=lambda row: (
            row["validation_top1"],
            row["weight_source"] != "coordinate_validation",
        ),
    )
    selected_key = (
        str(selected["kind"]),
        float(selected["parameter"]),
    )
    if selected_key[0] == "raw":
        strict_test = float(base_report["test_top1"])
    else:
        transformed_test = _transform(
            test_scores,
            test_groups,
            *selected_key,
        )
        selected_weights = np.asarray(selected["weights"])
        strict_test = ensemble._accuracy(
            _blend(transformed_test, selected_weights),
            test_labels,
            test_groups.tolist(),
        )
    report = {
        "model_order": names,
        "base_validation_top1": base_report["validation_top1"],
        "base_test_top1": base_report["test_top1"],
        "selected_on_validation": selected,
        "strict_test_top1": strict_test,
        "top_validation_candidates": sorted(
            [*candidates, *tuned],
            key=lambda row: -row["validation_top1"],
        )[:15],
        "target_top1": 0.9,
        "target_met": strict_test >= 0.9,
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
