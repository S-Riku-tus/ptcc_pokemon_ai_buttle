"""Ablate v31 order/serial features without repeating replay extraction."""

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


def _metrics(
    model: Any,
    arrays: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, float]:
    scores, labels, groups = teacher._predict_for_decisions(
        model, arrays, indices
    )
    result, _ = teacher._evaluate(
        scores,
        labels,
        groups,
        arrays["fallback_correct"][indices],
        arrays["teacher_action_types"][indices],
        1.0,
    )
    return {
        "top1": result["semantic_top1"],
        "top2": result["semantic_top2"],
        "top3": result["semantic_top3"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_cache", type=Path)
    parser.add_argument("order_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.base_cache, allow_pickle=False) as base:
        base_names = set(base["feature_names"].astype(str).tolist())
    with np.load(args.order_cache, allow_pickle=False) as cached:
        source_features = cached["features"]
        source_names = cached["feature_names"].astype(str).tolist()
        common: dict[str, Any] = {
            key: cached[key]
            for key in (
                "labels",
                "weights",
                "groups",
                "fallback_correct",
                "teacher_action_types",
            )
        }
        splits = cached["splits"].astype(str)

    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    new_names = set(source_names) - base_names
    variants = {
        "v30_schema": lambda name: name in base_names,
        "step_only": lambda name: (
            name in base_names or name == "observation_step"
        ),
        "option_order_no_serial": lambda name: (
            name in base_names
            or (
                name in new_names
                and name.startswith("candidate_")
                and "serial" not in name
            )
        ),
        "hand_order_ids_no_serial": lambda name: (
            name in base_names
            or name == "observation_step"
            or (
                name.startswith("self_hand_slot_")
                and name.endswith("_id")
            )
        ),
        "all_no_serial": lambda name: "serial" not in name,
        "all_features": lambda name: True,
    }
    reports = []
    for variant, keep in variants.items():
        columns = [
            index for index, name in enumerate(source_names) if keep(name)
        ]
        feature_names = [source_names[index] for index in columns]
        arrays = {
            **common,
            "features": source_features[:, columns],
        }
        model = teacher._fit(
            arrays,
            feature_names,
            train,
            n_estimators=900,
            validation_indices=validation,
        )
        reports.append({
            "variant": variant,
            "features": len(feature_names),
            "best_iteration": int(model.best_iteration_ or 900),
            "validation": _metrics(model, arrays, validation),
            "test": _metrics(model, arrays, test),
        })
        del arrays
        del model

    report = {
        "base_cache": str(args.base_cache.resolve()),
        "order_cache": str(args.order_cache.resolve()),
        "new_features": sorted(new_names),
        "experiments": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "experiments": [
            {
                "variant": row["variant"],
                "features": row["features"],
                "validation": row["validation"]["top1"],
                "test": row["test"]["top1"],
            }
            for row in reports
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
