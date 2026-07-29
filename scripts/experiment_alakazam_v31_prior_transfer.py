"""Add an earlier same-team submission while freezing current-policy splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


KEYS = (
    "labels",
    "weights",
    "groups",
    "fallback_correct",
    "teacher_action_types",
    "episode_ids",
    "ranks",
)


def _load(path: Path, desired: list[str] | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cached:
        names = cached["feature_names"].astype(str).tolist()
        desired = names if desired is None else desired
        columns = [names.index(name) for name in desired]
        output: dict[str, Any] = {
            "features": cached["features"][:, columns],
            "splits": cached["splits"].astype(str),
            "names": desired,
        }
        output.update({key: cached[key] for key in KEYS})
    return output


def _metrics(
    model: Any,
    arrays: dict[str, Any],
    decisions: np.ndarray,
) -> dict[str, Any]:
    scores, labels, groups = teacher._predict_for_decisions(
        model, arrays, decisions
    )
    result, _ = teacher._evaluate(
        scores,
        labels,
        groups,
        arrays["fallback_correct"][decisions],
        arrays["teacher_action_types"][decisions],
        1.0,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current_cache", type=Path)
    parser.add_argument("prior_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = _load(args.prior_cache)
    current = _load(args.current_cache, prior["names"])
    current_count = len(current["groups"])
    arrays: dict[str, Any] = {
        "features": np.concatenate((current["features"], prior["features"])),
    }
    arrays.update({
        key: np.concatenate((current[key], prior[key]))
        for key in KEYS
    })
    current_train = np.flatnonzero(current["splits"] == "train")
    current_validation = np.flatnonzero(current["splits"] == "validation")
    current_test = np.flatnonzero(current["splits"] == "test")
    prior_all = current_count + np.arange(len(prior["groups"]))
    experiments = []
    best_model = None
    best_row = None
    original_weights = arrays["weights"].copy()
    for transfer_weight in (0.0, 0.25, 0.5, 1.0):
        if transfer_weight == 0:
            train = current_train
        else:
            train = np.concatenate((current_train, prior_all))
        arrays["weights"] = original_weights.copy()
        if transfer_weight:
            current_rows = len(current["labels"])
            arrays["weights"][current_rows:] *= transfer_weight
        model = teacher._fit(
            arrays,
            prior["names"],
            train,
            n_estimators=1200,
            validation_indices=current_validation,
        )
        validation = _metrics(model, arrays, current_validation)
        row = {
            "transfer_weight": transfer_weight,
            "best_iteration": int(model.best_iteration_ or 1200),
            "validation": validation,
            "test": _metrics(model, arrays, current_test),
        }
        experiments.append(row)
        print({
            "transfer_weight": transfer_weight,
            "validation_top1": validation["semantic_top1"],
            "test_top1": row["test"]["semantic_top1"],
        }, flush=True)
        if (
            best_row is None
            or validation["semantic_top1"]
            > best_row["validation"]["semantic_top1"]
        ):
            best_row = row
            best_model = model
    assert best_row is not None and best_model is not None
    report = {
        "current_decisions": current_count,
        "prior_decisions": len(prior["groups"]),
        "features": len(prior["names"]),
        "selection_rule": "highest current-policy validation Top-1",
        "experiments": experiments,
        "selected_transfer_weight": best_row["transfer_weight"],
        "selected_validation_top1": best_row["validation"]["semantic_top1"],
        "selected_test_top1": best_row["test"]["semantic_top1"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "selected_transfer_weight": report["selected_transfer_weight"],
        "selected_validation_top1": report["selected_validation_top1"],
        "selected_test_top1": report["selected_test_top1"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
