"""Compare CatBoost groupwise ranking on the frozen Majkel split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostRanker, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _select(
    arrays: dict[str, Any],
    decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts, ends = _ranges(arrays["groups"])
    rows = np.concatenate([
        np.arange(starts[index], ends[index], dtype=np.int64)
        for index in decisions
    ])
    groups = arrays["groups"][decisions]
    group_ids = np.repeat(np.arange(len(decisions), dtype=np.int64), groups)
    return (
        arrays["features"][rows],
        arrays["labels"][rows].astype(np.float32),
        arrays["weights"][rows],
        group_ids,
    )


def _accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    starts, ends = _ranges(groups)
    top1 = top2 = top3 = 0
    for start, end in zip(starts, ends):
        order = np.argsort(-scores[start:end], kind="stable")
        group_labels = labels[start:end]
        top1 += int(group_labels[order[0]] == 1)
        top2 += int(np.any(group_labels[order[:2]] == 1))
        top3 += int(np.any(group_labels[order[:3]] == 1))
    count = len(groups)
    return {
        "semantic_top1": top1 / count,
        "semantic_top2": top2 / count,
        "semantic_top3": top3 / count,
    }


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
    baseline = teacher._fit(
        arrays,
        desired,
        train,
        n_estimators=900,
        validation_indices=validation,
    )
    importance = baseline.booster_.feature_importance(importance_type="gain")
    selected_columns = set(
        np.argsort(-importance, kind="stable")[:160].tolist()
    )
    selected_columns.update(
        desired.index(name)
        for name in (
            "action_type",
            "option_type",
            "candidate_card_id",
            "candidate_attack_id",
            "candidate_target_id",
            "fallback_selected",
            "v29_selected",
        )
    )
    selected_columns = sorted(selected_columns)
    desired = [desired[index] for index in selected_columns]
    arrays["features"] = arrays["features"][:, selected_columns]
    train_x, train_y, train_w, train_group = _select(arrays, train)
    val_x, val_y, val_w, val_group = _select(arrays, validation)
    test_x, test_y, _, _ = _select(arrays, test)
    train_pool = Pool(
        train_x,
        label=train_y,
        weight=train_w,
        group_id=train_group,
        feature_names=desired,
    )
    validation_pool = Pool(
        val_x,
        label=val_y,
        weight=val_w,
        group_id=val_group,
        feature_names=desired,
    )
    experiments = []
    best_model = None
    best_row = None
    for loss, depth in (
        ("QuerySoftMax", 6),
    ):
        model = CatBoostRanker(
            loss_function=loss,
            eval_metric="NDCG:top=1",
            iterations=250,
            learning_rate=0.07,
            depth=depth,
            l2_leaf_reg=5.0,
            random_seed=741,
            thread_count=4,
            random_strength=0.3,
            od_type="Iter",
            od_wait=35,
            verbose=10,
            allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=validation_pool)
        validation_scores = model.predict(val_x)
        validation_metrics = _accuracy(
            validation_scores,
            val_y,
            arrays["groups"][validation],
        )
        row = {
            "loss": loss,
            "depth": depth,
            "best_iteration": int(model.get_best_iteration()),
            "validation": validation_metrics,
        }
        experiments.append(row)
        print(json.dumps(row), flush=True)
        if (
            best_row is None
            or validation_metrics["semantic_top1"]
            > best_row["validation"]["semantic_top1"]
        ):
            best_row = row
            best_model = model
    assert best_model is not None and best_row is not None
    test_metrics = _accuracy(
        best_model.predict(test_x),
        test_y,
        arrays["groups"][test],
    )
    report = {
        "cache": str(args.cache.resolve()),
        "features": len(desired),
        "selection_rule": "highest frozen validation Top-1",
        "experiments": experiments,
        "selected": best_row,
        "test": test_metrics,
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
