"""Train diverse candidate scorers and greedily ensemble on validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(groups, dtype=np.int64)
    ends = np.cumsum(values)
    return np.r_[0, ends[:-1]], ends


def _normalize(scores: np.ndarray, groups: list[int]) -> np.ndarray:
    result = scores.astype(np.float32).copy()
    starts, ends = _ranges(groups)
    for start, end in zip(starts, ends):
        values = result[start:end]
        result[start:end] = (
            (values - float(values.mean()))
            / max(float(values.std()), 1e-5)
        )
    return result


def _accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: list[int],
) -> float:
    starts, ends = _ranges(groups)
    return float(np.mean([
        labels[start + int(np.argmax(scores[start:end]))] == 1
        for start, end in zip(starts, ends)
    ]))


def _ranker(
    config: dict[str, Any],
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    names: list[str],
    categorical: list[int],
) -> Any:
    train_x, train_y, train_w, train_groups = train_data
    val_x, val_y, _, val_groups = validation_data
    model = lgb.LGBMRanker(
        objective=config["objective"],
        metric="ndcg",
        n_estimators=1200,
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        max_depth=config.get("max_depth", -1),
        min_child_samples=config["min_child_samples"],
        subsample=config.get("subsample", 0.9),
        subsample_freq=1,
        colsample_bytree=config.get("colsample_bytree", 0.88),
        reg_alpha=config.get("reg_alpha", 0.2),
        reg_lambda=config.get("reg_lambda", 1.0),
        random_state=config["seed"],
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        group=train_groups,
        sample_weight=train_w,
        feature_name=names,
        categorical_feature=(
            categorical if config.get("categorical", True) else []
        ),
        eval_set=[(val_x, val_y)],
        eval_group=[val_groups],
        callbacks=[lgb.early_stopping(55, verbose=False)],
    )
    return model


def _classifier(
    config: dict[str, Any],
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    names: list[str],
    categorical: list[int],
) -> Any:
    train_x, train_y, train_w, _ = train_data
    val_x, val_y, _, _ = validation_data
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=config.get("colsample_bytree", 0.88),
        reg_alpha=0.2,
        reg_lambda=1.0,
        scale_pos_weight=config["scale_pos_weight"],
        random_state=config["seed"],
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=train_w,
        feature_name=names,
        categorical_feature=categorical,
        eval_set=[(val_x, val_y)],
        callbacks=[lgb.early_stopping(55, verbose=False)],
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.schema_cache, allow_pickle=False) as schema:
        desired = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        names = cached["feature_names"].astype(str).tolist()
        columns = [names.index(name) for name in desired]
        arrays: dict[str, Any] = {
            "features": cached["features"][:, columns],
            "labels": cached["labels"],
            "weights": cached["weights"],
            "groups": cached["groups"],
        }
        splits = cached["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    train_data = teacher._select_decisions(arrays, train)
    validation_data = teacher._select_decisions(arrays, validation)
    test_data = teacher._select_decisions(arrays, test)
    categorical = [
        index
        for index, name in enumerate(desired)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ]
    configs = [
        {
            "name": "lambda_standard",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 40,
            "seed": 741,
        },
        {
            "name": "lambda_small_leaf",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.025,
            "num_leaves": 63,
            "min_child_samples": 20,
            "colsample_bytree": 0.95,
            "seed": 19,
        },
        {
            "name": "lambda_large_leaf",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.025,
            "num_leaves": 255,
            "min_child_samples": 55,
            "colsample_bytree": 0.80,
            "seed": 1086,
        },
        {
            "name": "lambda_numeric_ids",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 35,
            "categorical": False,
            "seed": 305,
        },
        {
            "name": "xendcg",
            "kind": "ranker",
            "objective": "rank_xendcg",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 35,
            "seed": 743,
        },
        {
            "name": "binary_balanced",
            "kind": "classifier",
            "learning_rate": 0.025,
            "num_leaves": 127,
            "min_child_samples": 35,
            "scale_pos_weight": 7.0,
            "seed": 1152,
        },
    ]
    val_x, val_y, _, val_groups = validation_data
    test_x, test_y, _, test_groups = test_data
    models = []
    for config in configs:
        if config["kind"] == "ranker":
            model = _ranker(
                config,
                train_data,
                validation_data,
                desired,
                categorical,
            )
            val_scores = model.predict(val_x)
            test_scores = model.predict(test_x)
        else:
            model = _classifier(
                config,
                train_data,
                validation_data,
                desired,
                categorical,
            )
            val_scores = model.predict_proba(val_x)[:, 1]
            test_scores = model.predict_proba(test_x)[:, 1]
        val_scores = _normalize(val_scores, val_groups)
        test_scores = _normalize(test_scores, test_groups)
        row = {
            "name": config["name"],
            "best_iteration": int(model.best_iteration_ or 1200),
            "validation_top1": _accuracy(val_scores, val_y, val_groups),
            "test_top1": _accuracy(test_scores, test_y, test_groups),
            "validation_scores": val_scores,
            "test_scores": test_scores,
        }
        models.append(row)
        print({
            key: row[key]
            for key in (
                "name",
                "best_iteration",
                "validation_top1",
                "test_top1",
            )
        }, flush=True)

    selected = max(models, key=lambda row: row["validation_top1"])
    ensemble_val = selected["validation_scores"].copy()
    ensemble_test = selected["test_scores"].copy()
    components = [{"name": selected["name"], "weight": 1.0}]
    best_accuracy = _accuracy(ensemble_val, val_y, val_groups)
    for _ in range(8):
        candidate = None
        for model in models:
            for weight in np.arange(0.1, 2.01, 0.1):
                score = _accuracy(
                    ensemble_val + float(weight) * model["validation_scores"],
                    val_y,
                    val_groups,
                )
                if candidate is None or score > candidate["top1"]:
                    candidate = {
                        "model": model,
                        "weight": float(weight),
                        "top1": score,
                    }
        assert candidate is not None
        if candidate["top1"] <= best_accuracy:
            break
        ensemble_val += (
            candidate["weight"] * candidate["model"]["validation_scores"]
        )
        ensemble_test += (
            candidate["weight"] * candidate["model"]["test_scores"]
        )
        best_accuracy = candidate["top1"]
        components.append({
            "name": candidate["model"]["name"],
            "weight": candidate["weight"],
        })
    report = {
        "features": len(desired),
        "models": [
            {
                key: value
                for key, value in row.items()
                if not key.endswith("_scores")
            }
            for row in models
        ],
        "ensemble": {
            "components": components,
            "validation_top1": best_accuracy,
            "test_top1": _accuracy(
                ensemble_test, test_y, test_groups
            ),
        },
        "validation_oracle_any_model": float(np.mean([
            any(
                val_y[start + int(np.argmax(model["validation_scores"][start:end]))]
                == 1
                for model in models
            )
            for start, end in zip(*_ranges(val_groups))
        ])),
        "test_oracle_any_model": float(np.mean([
            any(
                test_y[start + int(np.argmax(model["test_scores"][start:end]))]
                == 1
                for model in models
            )
            for start, end in zip(*_ranges(test_groups))
        ])),
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
