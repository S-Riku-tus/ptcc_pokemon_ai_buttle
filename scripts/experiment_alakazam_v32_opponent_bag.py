"""Evaluate permutation-invariant opponent-discard evidence for v32.

The v31 base policy knows ordered opponent Bench slots but only a few
hand-picked opponent discard identities.  This challenger turns every card ID
seen in the chronological training split into a count feature, allowing the
ranker to infer matchup/archetype without depending on discard order.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _bag(
    values: np.ndarray,
    card_ids: list[int],
) -> np.ndarray:
    output = np.empty(
        (len(values), len(card_ids)),
        dtype=np.float32,
    )
    for column, card_id in enumerate(card_ids):
        output[:, column] = (values == card_id).sum(axis=1)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("--max-card-ids", type=int, default=206)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.schema_cache, allow_pickle=False) as schema:
        desired = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        original_names = cached["feature_names"].astype(str).tolist()
        desired_columns = [
            original_names.index(name) for name in desired
        ]
        discard_columns = [
            index
            for index, name in enumerate(original_names)
            if (
                name.startswith("opp_discard_slot_")
                and name.endswith("_id")
            )
        ]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        starts, _ = _ranges(groups)
        train_decisions = np.flatnonzero(splits == "train")
        train_rows = starts[train_decisions]
        discard_at_decisions = cached["features"][
            train_rows
        ][:, discard_columns].astype(np.int16)
        frequencies = Counter(
            int(value)
            for value in discard_at_decisions.ravel()
            if value >= 0
        )
        card_ids = [
            card_id
            for card_id, _ in frequencies.most_common(
                args.max_card_ids
            )
        ]
        base = cached["features"][:, desired_columns]
        discard_values = cached["features"][
            :, discard_columns
        ].astype(np.int16)
        arrays: dict[str, Any] = {
            "labels": cached["labels"],
            "weights": cached["weights"],
            "groups": groups,
        }
        episode_ids = cached["episode_ids"]

    bag = _bag(discard_values, card_ids)
    features = np.column_stack((base, bag))
    del base
    del discard_values
    del bag
    names = desired + [
        f"opp_discard_count_{card_id}"
        for card_id in card_ids
    ]
    arrays["features"] = features
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    categorical = [
        index
        for index, name in enumerate(names)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ]
    configs = [
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
            "name": "lambda_medium_bag",
            "kind": "ranker",
            "objective": "lambdarank",
            "learning_rate": 0.025,
            "num_leaves": 127,
            "min_child_samples": 45,
            "colsample_bytree": 0.85,
            "seed": 741,
        },
    ]
    validation_data = teacher._select_decisions(arrays, validation)
    test_x, test_y, _, test_groups = teacher._select_decisions(
        arrays,
        test,
    )
    experiments = []
    models = []
    for config in configs:
        model = ensemble._ranker(
            config,
            teacher._select_decisions(arrays, train),
            validation_data,
            names,
            categorical,
        )
        validation_x, validation_y, _, validation_groups = (
            validation_data
        )
        row = {
            "name": config["name"],
            "best_iteration": int(model.best_iteration_ or 1200),
            "validation_top1": ensemble._accuracy(
                model.predict(validation_x),
                validation_y,
                validation_groups,
            ),
            "test_top1": ensemble._accuracy(
                model.predict(test_x),
                test_y,
                test_groups,
            ),
        }
        experiments.append(row)
        models.append(model)
        print(json.dumps(row), flush=True)

    selected_index = max(
        range(len(experiments)),
        key=lambda index: experiments[index]["validation_top1"],
    )
    selected_model = models[selected_index]
    split_ranges = {
        split: (
            int(episode_ids[np.flatnonzero(splits == split)].min()),
            int(episode_ids[np.flatnonzero(splits == split)].max()),
        )
        for split in ("train", "validation", "test")
    }
    report = {
        "features": len(names),
        "opponent_discard_card_ids": card_ids,
        "opponent_discard_features": len(card_ids),
        "split_episode_ranges": split_ranges,
        "experiments": experiments,
        "selected": experiments[selected_index],
        "target_top1": 0.9,
        "target_met": experiments[selected_index]["test_top1"] >= 0.9,
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
