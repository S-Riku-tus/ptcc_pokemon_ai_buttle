"""Train a chronological score-only stack for an arbitrary v32 ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_score_stack as stack  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    previous = json.loads(
        args.blend_report.read_text(encoding="utf-8")
    )
    model_names = list(previous["model_order"])
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
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
    validation_decisions = np.flatnonzero(splits == "validation")
    validation_episodes = episode_ids[validation_decisions]
    ordered = np.unique(validation_episodes)
    ordered.sort()
    cut = max(1, int(len(ordered) * 0.60))
    early = set(ordered[:cut].tolist())
    meta_train = np.flatnonzero(np.asarray([
        episode in early for episode in validation_episodes
    ]))
    meta_validation = np.flatnonzero(np.asarray([
        episode not in early for episode in validation_episodes
    ]))

    validation_x, names = stack._features(
        validation_scores,
        validation_groups,
    )
    test_x, _ = stack._features(test_scores, test_groups)
    arrays: dict[str, Any] = {
        "features": validation_x,
        "labels": validation_labels,
        "weights": np.ones(
            len(validation_labels),
            dtype=np.float32,
        ),
        "groups": validation_groups,
    }
    experiments = []
    for leaves, minimum in (
        (3, 80),
        (7, 60),
        (15, 60),
        (31, 80),
    ):
        model = stack._fit(
            arrays,
            names,
            meta_train,
            meta_validation,
            leaves=leaves,
            minimum=minimum,
            iterations=800,
        )
        late_x, late_y, _, late_groups = teacher._select_decisions(
            arrays,
            meta_validation,
        )
        row = {
            "leaves": leaves,
            "min_child_samples": minimum,
            "best_iteration": int(model.best_iteration_ or 800),
            "meta_validation_top1": ensemble._accuracy(
                model.predict(late_x),
                late_y,
                late_groups,
            ),
        }
        experiments.append(row)
        print(json.dumps(row), flush=True)
    selected = max(
        experiments,
        key=lambda row: (
            row["meta_validation_top1"],
            -row["leaves"],
        ),
    )
    final = stack._fit(
        arrays,
        names,
        np.arange(len(validation_groups), dtype=np.int64),
        None,
        leaves=selected["leaves"],
        minimum=selected["min_child_samples"],
        iterations=selected["best_iteration"],
    )
    test_top1 = ensemble._accuracy(
        final.predict(test_x),
        test_labels,
        test_groups.tolist(),
    )
    report = {
        "model_order": model_names,
        "features": len(names),
        "meta_train_decisions": len(meta_train),
        "meta_validation_decisions": len(meta_validation),
        "experiments": experiments,
        "selected": selected,
        "test_top1": test_top1,
        "weighted_blend_reference": previous["test_top1"],
        "target_top1": 0.9,
        "target_met": test_top1 >= 0.9,
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
