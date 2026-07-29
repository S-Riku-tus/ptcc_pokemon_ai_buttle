"""Ablate matchup/history feature families from the v31 history cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _top1(
    model: Any,
    arrays: dict[str, Any],
    indices: np.ndarray,
) -> float:
    scores, labels, groups = teacher._predict_for_decisions(
        model, arrays, indices
    )
    metrics, _ = teacher._evaluate(
        scores,
        labels,
        groups,
        arrays["fallback_correct"][indices],
        arrays["teacher_action_types"][indices],
        1.0,
    )
    return float(metrics["semantic_top1"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_cache", type=Path)
    parser.add_argument("history_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.base_cache, allow_pickle=False) as base:
        base_names = set(base["feature_names"].astype(str).tolist())
    with np.load(args.history_cache, allow_pickle=False) as cached:
        source = cached["features"]
        names = cached["feature_names"].astype(str).tolist()
        shared: dict[str, Any] = {
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

    def family(prefixes: tuple[str, ...]) -> Callable[[str], bool]:
        return lambda name: (
            name in base_names
            or name.startswith("candidate_")
            or any(name.startswith(prefix) for prefix in prefixes)
        )

    variants = {
        "base": lambda name: name in base_names,
        "option_order": family(()),
        "option_plus_opp_discard": family(("opp_discard_slot_",)),
        "option_plus_long_recent": family(("long_recent_log_",)),
        "option_plus_turn_open": family(("turn_open_log_",)),
        "option_plus_all_logs": family((
            "long_recent_log_",
            "turn_open_log_",
        )),
        "all": lambda name: True,
    }
    reports = []
    for variant, keep in variants.items():
        columns = [
            index for index, name in enumerate(names) if keep(name)
        ]
        selected_names = [names[index] for index in columns]
        arrays = {**shared, "features": source[:, columns]}
        model = teacher._fit(
            arrays,
            selected_names,
            train,
            n_estimators=900,
            validation_indices=validation,
        )
        reports.append({
            "variant": variant,
            "features": len(selected_names),
            "best_iteration": int(model.best_iteration_ or 900),
            "validation_top1": _top1(model, arrays, validation),
            "test_top1": _top1(model, arrays, test),
        })
        del arrays
        del model

    report = {
        "base_cache": str(args.base_cache.resolve()),
        "history_cache": str(args.history_cache.resolve()),
        "experiments": reports,
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
