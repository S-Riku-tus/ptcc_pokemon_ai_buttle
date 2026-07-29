"""Train one ranker on coherent teachers and report each held-out cohort."""

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


ARRAY_KEYS = (
    "labels",
    "weights",
    "groups",
    "fallback_correct",
    "teacher_action_types",
    "episode_ids",
    "ranks",
)


def _load(path: Path, desired_names: list[str] | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cached:
        names = cached["feature_names"].astype(str).tolist()
        if desired_names is None:
            desired_names = names
        missing = sorted(set(desired_names) - set(names))
        if missing:
            raise ValueError(f"{path} lacks {missing[:10]}")
        columns = [names.index(name) for name in desired_names]
        result: dict[str, Any] = {
            "features": cached["features"][:, columns],
            "splits": cached["splits"].astype(str),
            "feature_names": desired_names,
        }
        result.update({key: cached[key] for key in ARRAY_KEYS})
    return result


def _subset_metrics(
    model: Any,
    arrays: dict[str, Any],
    decisions: np.ndarray,
) -> dict[str, Any]:
    scores, labels, groups = teacher._predict_for_decisions(
        model, arrays, decisions
    )
    metrics, _ = teacher._evaluate(
        scores,
        labels,
        groups,
        arrays["fallback_correct"][decisions],
        arrays["teacher_action_types"][decisions],
        1.0,
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("majkel_cache", type=Path)
    parser.add_argument("rmy_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rmy = _load(args.rmy_cache)
    names = rmy["feature_names"]
    majkel = _load(args.majkel_cache, names)
    arrays: dict[str, Any] = {
        "features": np.concatenate((majkel["features"], rmy["features"])),
        "splits": np.concatenate((majkel["splits"], rmy["splits"])),
    }
    arrays.update({
        key: np.concatenate((majkel[key], rmy[key]))
        for key in ARRAY_KEYS
    })
    cohort = np.concatenate((
        np.full(len(majkel["groups"]), "majkel"),
        np.full(len(rmy["groups"]), "rmy"),
    ))
    train = np.flatnonzero(arrays["splits"] == "train")
    validation = np.flatnonzero(arrays["splits"] == "validation")
    test = np.flatnonzero(arrays["splits"] == "test")
    model = teacher._fit(
        arrays,
        names,
        train,
        n_estimators=1200,
        validation_indices=validation,
    )
    report = {
        "features": len(names),
        "best_iteration": int(model.best_iteration_ or 1200),
        "decisions": len(arrays["groups"]),
        "validation": _subset_metrics(model, arrays, validation),
        "test": _subset_metrics(model, arrays, test),
        "by_cohort": {},
    }
    for name in ("majkel", "rmy"):
        report["by_cohort"][name] = {
            "validation": _subset_metrics(
                model,
                arrays,
                validation[cohort[validation] == name],
            ),
            "test": _subset_metrics(
                model,
                arrays,
                test[cohort[test] == name],
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "best_iteration": report["best_iteration"],
        "aggregate_validation": report["validation"]["semantic_top1"],
        "aggregate_test": report["test"]["semantic_top1"],
        "majkel_test": report["by_cohort"]["majkel"]["test"]["semantic_top1"],
        "rmy_test": report["by_cohort"]["rmy"]["test"]["semantic_top1"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
