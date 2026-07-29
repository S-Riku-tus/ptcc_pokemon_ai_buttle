"""Score an external teacher corpus with the frozen-split v32 model family."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_deepset_blend as deep_blend  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_cache", type=Path)
    parser.add_argument("external_cache", type=Path)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--wide-deep", type=Path, required=True)
    parser.add_argument("--deep-seed305", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.training_cache, allow_pickle=False) as saved:
        names = saved["feature_names"].astype(str).tolist()
        training_arrays: dict[str, Any] = {
            "features": saved["features"],
            "labels": saved["labels"],
            "weights": saved["weights"],
            "groups": saved["groups"],
        }
        splits = saved["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    train_data = teacher._select_decisions(training_arrays, train)
    validation_data = teacher._select_decisions(
        training_arrays,
        validation,
    )
    categorical = [
        index
        for index, name in enumerate(names)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ]
    configs = [
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
    models = []
    for config in configs:
        model = ensemble._ranker(
            config,
            train_data,
            validation_data,
            names,
            categorical,
        )
        models.append(model)
        print(json.dumps({
            "model": config["name"],
            "best_iteration": int(model.best_iteration_ or 1200),
        }), flush=True)
    del train_data, validation_data, training_arrays, splits
    gc.collect()

    with np.load(args.external_cache, allow_pickle=False) as saved:
        external_names = saved["feature_names"].astype(str).tolist()
        columns = [external_names.index(name) for name in names]
        features = saved["features"][:, columns]
        labels = saved["labels"]
        weights = saved["weights"]
        groups = saved["groups"]
        external_splits = saved["splits"].astype(str)
        episode_ids = saved["episode_ids"]
    decisions = np.arange(len(groups), dtype=np.int64)
    external_arrays = {
        "features": features,
        "labels": labels,
        "weights": weights,
        "groups": groups,
    }
    external_x, external_y, external_w, external_groups = (
        teacher._select_decisions(external_arrays, decisions)
    )
    score_sets = {}
    for config, model in zip(configs, models):
        scores = ensemble._normalize(
            model.predict(external_x),
            external_groups,
        )
        score_sets[config["name"]] = scores
        print(json.dumps({
            "model": config["name"],
            "external_top1": ensemble._accuracy(
                scores,
                external_y,
                external_groups,
            ),
        }), flush=True)
    del external_x, models
    gc.collect()

    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    checkpoints = [
        ("deep", args.deep),
        ("attention", args.attention),
        ("wide_deep", args.wide_deep),
        ("deep_seed305", args.deep_seed305),
    ]
    for name, checkpoint in checkpoints:
        scores = deep_blend._deep_scores(
            checkpoint,
            features,
            names,
            groups,
            labels,
            weights,
            decisions,
            128,
        )
        scores = ensemble._normalize(scores, external_groups)
        score_sets[name] = scores
        print(json.dumps({
            "model": name,
            "external_top1": ensemble._accuracy(
                scores,
                external_y,
                external_groups,
            ),
        }), flush=True)
        gc.collect()

    output = {
        f"external_{name}": score_sets[name]
        for name in (
            "large",
            "numeric",
            "deep",
            "attention",
            "wide_deep",
            "deep_seed305",
        )
    }
    output.update({
        "external_labels": external_y,
        "external_weights": external_w,
        "external_groups": np.asarray(external_groups),
        "external_splits": external_splits,
        "external_episode_ids": episode_ids,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(json.dumps({
        "output": str(args.output),
        "decisions": len(groups),
        "candidate_rows": len(labels),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
