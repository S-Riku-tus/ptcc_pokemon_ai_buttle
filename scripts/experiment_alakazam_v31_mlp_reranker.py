"""Nonlinear MLP challenger using leakage-free OOF base-ranker scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_oof_reranker as reranker  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("oof_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays: dict[str, Any] = {
            key: cached[key]
            for key in (
                "features",
                "labels",
                "weights",
                "groups",
                "episode_ids",
            )
        }
        splits = cached["splits"].astype(str)
        names = cached["feature_names"].astype(str).tolist()
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    base = teacher._fit(
        arrays,
        names,
        train,
        n_estimators=900,
        validation_indices=validation,
    )
    base_scores = base.predict(arrays["features"]).astype(np.float32)
    oof = np.load(args.oof_cache, allow_pickle=False)
    candidate_start, critical, _ = reranker._schema(names)
    train_x, train_y, train_weights = reranker._examples(
        arrays, train, oof, candidate_start, critical
    )
    models = []
    for hidden, alpha in (
        ((128, 64), 1e-4),
        ((256, 128), 3e-4),
    ):
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=alpha,
                batch_size=512,
                learning_rate_init=5e-4,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=10,
                random_state=741,
                verbose=False,
            ),
        )
        model.fit(train_x, train_y, mlpclassifier__sample_weight=train_weights)
        validation_predictions = reranker._predictions(
            model,
            arrays,
            validation,
            base_scores,
            candidate_start,
            critical,
        )
        grid = [
            {
                "threshold": float(threshold),
                "top1": reranker._accuracy(
                    validation_predictions, float(threshold)
                ),
            }
            for threshold in np.arange(0.30, 0.951, 0.025)
        ]
        selected = max(grid, key=lambda row: row["top1"])
        models.append({
            "model": model,
            "hidden": hidden,
            "alpha": alpha,
            "iterations": int(
                model.named_steps["mlpclassifier"].n_iter_
            ),
            **selected,
        })
        print({
            key: models[-1][key]
            for key in ("hidden", "alpha", "iterations", "threshold", "top1")
        }, flush=True)
    selected = max(models, key=lambda row: row["top1"])
    test_predictions = reranker._predictions(
        selected["model"],
        arrays,
        test,
        base_scores,
        candidate_start,
        critical,
    )
    report = {
        "cache": str(args.cache.resolve()),
        "train_pairs": len(train_y),
        "positive_rate": float(train_y.mean()),
        "experiments": [
            {key: value for key, value in row.items() if key != "model"}
            for row in models
        ],
        "selected": {
            key: value
            for key, value in selected.items()
            if key != "model"
        },
        "test_base": reranker._accuracy(test_predictions, 1.1),
        "test_top1": reranker._accuracy(
            test_predictions, float(selected["threshold"])
        ),
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
