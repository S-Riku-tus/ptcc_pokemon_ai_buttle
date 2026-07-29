"""Stack diverse base-ranker scores using only the validation-era labels."""

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

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _meta_features(
    original: np.ndarray,
    scores: list[np.ndarray],
    groups: list[int],
) -> tuple[np.ndarray, list[str]]:
    starts, ends = ensemble._ranges(groups)
    extras = np.zeros((len(original), len(scores) * 3 + 2), dtype=np.float32)
    for model_index, model_scores in enumerate(scores):
        for start, end in zip(starts, ends):
            order = np.argsort(-model_scores[start:end], kind="stable")
            ranks = np.empty(end - start, dtype=np.float32)
            ranks[order] = np.arange(end - start)
            extras[start:end, model_index * 3] = model_scores[start:end]
            extras[start:end, model_index * 3 + 1] = ranks
            extras[start:end, model_index * 3 + 2] = (
                ranks == 0
            ).astype(np.float32)
    score_columns = extras[:, np.arange(0, len(scores) * 3, 3)]
    extras[:, -2] = score_columns.mean(axis=1)
    extras[:, -1] = score_columns.std(axis=1)
    names = [
        value
        for index in range(len(scores))
        for value in (
            f"stack_model_{index}_score",
            f"stack_model_{index}_rank",
            f"stack_model_{index}_selected",
        )
    ] + ["stack_score_mean", "stack_score_std"]
    return np.concatenate((original, extras), axis=1), names


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
            "episode_ids": cached["episode_ids"],
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
            "name": "standard",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 40,
            "seed": 741,
        },
        {
            "name": "large",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.025,
            "num_leaves": 255,
            "min_child_samples": 55,
            "colsample_bytree": 0.80,
            "seed": 1086,
        },
        {
            "name": "numeric",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 35,
            "categorical": False,
            "seed": 305,
        },
    ]
    validation_scores = []
    test_scores = []
    base_report = []
    val_x, val_y, val_w, val_groups = validation_data
    test_x, test_y, test_w, test_groups = test_data
    for config in configs:
        model = ensemble._ranker(
            config,
            train_data,
            validation_data,
            desired,
            categorical,
        )
        val_score = ensemble._normalize(model.predict(val_x), val_groups)
        test_score = ensemble._normalize(model.predict(test_x), test_groups)
        validation_scores.append(val_score)
        test_scores.append(test_score)
        base_report.append({
            "name": config["name"],
            "best_iteration": int(model.best_iteration_ or 1200),
            "validation_top1": ensemble._accuracy(
                val_score, val_y, val_groups
            ),
            "test_top1": ensemble._accuracy(
                test_score, test_y, test_groups
            ),
        })
        print(base_report[-1], flush=True)

    meta_val_x, extra_names = _meta_features(
        val_x, validation_scores, val_groups
    )
    meta_test_x, _ = _meta_features(
        test_x, test_scores, test_groups
    )
    ordered_episodes = sorted(set(arrays["episode_ids"][validation]))
    cut = int(len(ordered_episodes) * 0.60)
    meta_train_episodes = set(ordered_episodes[:cut])
    meta_train_decisions = np.flatnonzero(np.asarray([
        episode in meta_train_episodes
        for episode in arrays["episode_ids"][validation]
    ]))
    meta_validation_decisions = np.flatnonzero(np.asarray([
        episode not in meta_train_episodes
        for episode in arrays["episode_ids"][validation]
    ]))
    meta_arrays: dict[str, Any] = {
        "features": meta_val_x,
        "labels": val_y,
        "weights": val_w,
        "groups": np.asarray(val_groups),
    }
    meta_names = desired + extra_names
    meta_categorical = categorical
    experiments = []
    selected_model = None
    selected = None
    for leaves, child in ((31, 20), (63, 25), (127, 30)):
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=leaves,
            min_child_samples=child,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=741,
            n_jobs=4,
            verbosity=-1,
        )
        mx, my, mw, mgroups = teacher._select_decisions(
            meta_arrays, meta_train_decisions
        )
        vx, vy, _, vgroups = teacher._select_decisions(
            meta_arrays, meta_validation_decisions
        )
        model.fit(
            mx,
            my,
            group=mgroups,
            sample_weight=mw,
            feature_name=meta_names,
            categorical_feature=meta_categorical,
            eval_set=[(vx, vy)],
            eval_group=[vgroups],
            callbacks=[lgb.early_stopping(45, verbose=False)],
        )
        validation_top1 = ensemble._accuracy(
            model.predict(vx), vy, vgroups
        )
        row = {
            "leaves": leaves,
            "min_child_samples": child,
            "best_iteration": int(model.best_iteration_ or 900),
            "meta_validation_top1": validation_top1,
        }
        experiments.append(row)
        print(row, flush=True)
        if selected is None or validation_top1 > selected[
            "meta_validation_top1"
        ]:
            selected = row
            selected_model = model
    assert selected is not None and selected_model is not None
    test_top1 = ensemble._accuracy(
        selected_model.predict(meta_test_x),
        test_y,
        test_groups,
    )
    weighted_ensemble = test_scores[1] + 1.3 * test_scores[2]
    report = {
        "base_models": base_report,
        "meta_train_decisions": len(meta_train_decisions),
        "meta_validation_decisions": len(meta_validation_decisions),
        "experiments": experiments,
        "selected": selected,
        "test_top1": test_top1,
        "reference_weighted_ensemble_test": ensemble._accuracy(
            weighted_ensemble, test_y, test_groups
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
