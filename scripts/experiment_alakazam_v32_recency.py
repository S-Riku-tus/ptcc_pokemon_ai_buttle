"""Test chronological recency weighting for the v32 teacher ranker."""

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
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _weighted_arrays(
    arrays: dict[str, Any],
    train: np.ndarray,
    *,
    floor: float,
    power: float,
) -> dict[str, Any]:
    episodes = arrays["episode_ids"][train]
    ordered = np.unique(episodes)
    positions = {
        int(episode): index / max(len(ordered) - 1, 1)
        for index, episode in enumerate(ordered)
    }
    decision_multipliers = np.ones(
        len(arrays["groups"]),
        dtype=np.float32,
    )
    decision_multipliers[train] = np.asarray([
        floor + (1.0 - floor) * positions[int(episode)] ** power
        for episode in episodes
    ], dtype=np.float32)
    starts, ends = _ranges(arrays["groups"])
    row_multipliers = np.ones(
        len(arrays["weights"]),
        dtype=np.float32,
    )
    for decision in train:
        row_multipliers[starts[decision]:ends[decision]] = (
            decision_multipliers[decision]
        )
    output = dict(arrays)
    output["weights"] = arrays["weights"] * row_multipliers
    return output


def _recent_window(
    train: np.ndarray,
    episode_ids: np.ndarray,
    fraction: float,
) -> np.ndarray:
    ordered = np.unique(episode_ids[train])
    ordered.sort()
    keep = set(
        ordered[max(0, int(len(ordered) * (1.0 - fraction))):].tolist()
    )
    return train[np.asarray([
        int(episode_ids[index]) in keep for index in train
    ])]


def _oracle(
    score_sets: list[np.ndarray],
    labels: np.ndarray,
    groups: list[int],
) -> float:
    starts, ends = _ranges(np.asarray(groups))
    correct = 0
    for start, end in zip(starts, ends):
        correct += int(any(
            labels[start + int(np.argmax(scores[start:end]))] == 1
            for scores in score_sets
        ))
    return correct / len(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument(
        "--weighted",
        action="append",
        default=[],
        metavar="FLOOR:POWER",
        help="Additional or replacement recency weighting variant.",
    )
    parser.add_argument(
        "--only-weighted",
        action="store_true",
        help="Run only variants supplied through --weighted.",
    )
    parser.add_argument("--base-seed", type=int, default=1086)
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="Use --base-seed for every variant to isolate weighting effects.",
    )
    args = parser.parse_args()

    with np.load(args.schema_cache, allow_pickle=False) as schema:
        desired = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        original_names = cached["feature_names"].astype(str).tolist()
        columns = [original_names.index(name) for name in desired]
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
    validation_data = teacher._select_decisions(arrays, validation)
    test_data = teacher._select_decisions(arrays, test)
    validation_x, validation_y, _, validation_groups = validation_data
    test_x, test_y, _, test_groups = test_data
    categorical = [
        index
        for index, name in enumerate(desired)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ]
    base_config = {
        "kind": "ranker",
        "objective": "lambdarank",
        "learning_rate": 0.025,
        "num_leaves": 255,
        "min_child_samples": 55,
        "colsample_bytree": 0.80,
        "seed": 1086,
    }
    defaults = [
        {
            "name": "uniform",
            "train": train,
            "arrays": arrays,
        },
        *[
            {
                "name": f"weighted_floor_{floor}_power_{power}",
                "train": train,
                "arrays": _weighted_arrays(
                    arrays,
                    train,
                    floor=floor,
                    power=power,
                ),
            }
            for floor, power in (
                (0.10, 1.0),
                (0.25, 1.0),
                (0.50, 1.0),
                (0.10, 2.0),
                (0.25, 2.0),
            )
        ],
        *[
            {
                "name": f"recent_window_{fraction}",
                "train": _recent_window(
                    train,
                    arrays["episode_ids"],
                    fraction,
                ),
                "arrays": arrays,
            }
            for fraction in (0.8, 0.6, 0.4)
        ],
    ]
    custom = []
    for value in args.weighted:
        floor_text, power_text = value.split(":", 1)
        floor = float(floor_text)
        power = float(power_text)
        custom.append({
            "name": f"weighted_floor_{floor}_power_{power}",
            "train": train,
            "arrays": _weighted_arrays(
                arrays,
                train,
                floor=floor,
                power=power,
            ),
        })
    variants = ([] if args.only_weighted else defaults) + custom
    experiments = []
    validation_scores = []
    test_scores = []
    for index, variant in enumerate(variants):
        config = dict(base_config)
        config["name"] = variant["name"]
        config["seed"] = (
            args.base_seed
            if args.same_seed
            else args.base_seed + index
        )
        model = ensemble._ranker(
            config,
            teacher._select_decisions(
                variant["arrays"],
                variant["train"],
            ),
            validation_data,
            desired,
            categorical,
        )
        validation_score = ensemble._normalize(
            model.predict(validation_x),
            validation_groups,
        )
        test_score = ensemble._normalize(
            model.predict(test_x),
            test_groups,
        )
        row = {
            "name": variant["name"],
            "train_decisions": len(variant["train"]),
            "best_iteration": int(model.best_iteration_ or 1200),
            "seed": int(config["seed"]),
            "validation_top1": ensemble._accuracy(
                validation_score,
                validation_y,
                validation_groups,
            ),
            "test_top1": ensemble._accuracy(
                test_score,
                test_y,
                test_groups,
            ),
        }
        experiments.append(row)
        validation_scores.append(validation_score)
        test_scores.append(test_score)
        print(json.dumps(row), flush=True)

    selected_index = max(
        range(len(experiments)),
        key=lambda index: experiments[index]["validation_top1"],
    )
    selected = experiments[selected_index]
    oracle = _oracle(
        validation_scores,
        validation_y,
        validation_groups,
    )
    test_oracle = _oracle(
        test_scores,
        test_y,
        test_groups,
    )
    report = {
        "experiments": experiments,
        "selected": selected,
        "validation_variant_oracle": oracle,
        "test_variant_oracle": test_oracle,
        "target_top1": 0.9,
        "target_met": selected["test_top1"] >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.scores_output,
        validation_scores=np.stack(validation_scores),
        test_scores=np.stack(test_scores),
        validation_labels=validation_y,
        test_labels=test_y,
        validation_groups=np.asarray(validation_groups),
        test_groups=np.asarray(test_groups),
        names=np.asarray([row["name"] for row in experiments]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
