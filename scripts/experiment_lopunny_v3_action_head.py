"""Validation-only hierarchical next-action head for Lopunny v3.

MAIN decisions are decomposed into (1) the semantic action family/card/attack
that Majkel performs next and (2) the concrete legal target/copy.  A
multiclass LightGBM model predicts stage (1) from the public state and ordered
legal-menu summary; the base ranker breaks ties for stage (2).  Test rows are
never accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402
from scripts.experiment_lopunny_v3_contextual_topk import (  # noqa: E402
    SEED,
    _fit_base,
    _predict_global,
    _rankable,
)


SLOT_FIELDS = (
    "option_type", "action_type", "candidate_card_id", "candidate_attack_id",
    "candidate_area", "candidate_inplay_area", "candidate_target_id",
)


def _signature(row: np.ndarray, columns: dict[str, int]) -> tuple[int, int, int]:
    return (
        int(round(float(row[columns["action_type"]]))),
        int(round(float(row[columns["candidate_card_id"]]))),
        int(round(float(row[columns["candidate_attack_id"]]))),
    )


def _decision_schema(
    names: list[str], count_names: list[str], slots: int
) -> tuple[list[int], list[str]]:
    slot_columns = [names.index(name) for name in SLOT_FIELDS]
    output = list(count_names)
    for slot in range(slots):
        output += [f"legal_slot_{slot}__{name}" for name in SLOT_FIELDS]
    return slot_columns, output


def _decision_matrix(
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    names: list[str],
    slots: int,
) -> tuple[np.ndarray, list[int], list[tuple[int, int, int]], list[str]]:
    count_names = arrays["count_feature_names"].astype(str).tolist()
    slot_columns, output_names = _decision_schema(names, count_names, slots)
    starts, ends = v1._group_ranges(arrays["groups"])
    rows: list[np.ndarray] = []
    kept: list[int] = []
    targets: list[tuple[int, int, int]] = []
    columns = {name: names.index(name) for name in SLOT_FIELDS}
    for decision_value in decisions:
        decision = int(decision_value)
        if (
            int(arrays["select_contexts"][decision]) != 0
            or bool(arrays["forced"][decision])
            or int(arrays["chosen_counts"][decision]) != 1
        ):
            continue
        start, end = int(starts[decision]), int(ends[decision])
        options = arrays["features"][start:end]
        ordered = np.full((slots, len(slot_columns)), -1.0, dtype=np.float32)
        ordered[:min(slots, len(options))] = options[:slots, slot_columns]
        chosen = int(np.flatnonzero(arrays["labels"][start:end] == 1)[0])
        rows.append(np.concatenate((
            arrays["count_features"][decision], ordered.reshape(-1)
        )).astype(np.float32))
        kept.append(decision)
        targets.append(_signature(options[chosen], columns))
    return np.asarray(rows), kept, targets, output_names


def _categorical(names: list[str]) -> list[int]:
    return [
        index for index, name in enumerate(names)
        if name in v1.BASE_CATEGORICAL
        or name.endswith("_id") or "_id_" in name
        or "__action_type" in name or "__option_type" in name
        or name.endswith("_area")
    ]


def _scores(
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    base_global: np.ndarray,
    probabilities: np.ndarray,
    kept: list[int],
    class_by_signature: dict[tuple[int, int, int], int],
    names: list[str],
    alpha: float,
) -> np.ndarray:
    starts, ends = v1._group_ranges(arrays["groups"])
    base_rows = v1._rows_for(arrays["groups"], decisions)
    output = base_global[base_rows].copy()
    local_offset = {
        int(decision): int(offset)
        for decision, offset in zip(
            decisions, np.r_[0, np.cumsum(arrays["groups"][decisions])[:-1]]
        )
    }
    columns = {name: names.index(name) for name in SLOT_FIELDS}
    for row_index, decision in enumerate(kept):
        start, end = int(starts[decision]), int(ends[decision])
        block = base_global[start:end]
        z = (block - block.mean()) / max(float(block.std()), 1e-5)
        head = np.full(end - start, -20.0, dtype=np.float32)
        for local, candidate in enumerate(arrays["features"][start:end]):
            class_index = class_by_signature.get(_signature(candidate, columns))
            if class_index is not None:
                head[local] = np.log(max(float(probabilities[row_index, class_index]), 1e-9))
        offset = local_offset[decision]
        output[offset:offset + end - start] = head + alpha * z
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-trees", type=int, default=600)
    parser.add_argument("--head-trees", type=int, default=1000)
    parser.add_argument("--slots", type=int, default=24)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    starts, _ = v1._group_ranges(arrays["groups"])
    arrays["decision_turns"] = np.rint(
        arrays["features"][starts, names.index("turn")]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)
    varying = v1._varying_columns(
        arrays["features"],
        v1._rows_for(arrays["groups"], _rankable(arrays, train)),
    )

    started = time.perf_counter()
    base = _fit_base(arrays, names, train, varying, args.base_trees, SEED)
    validation_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    _predict_global(
        base, arrays, validation, varying, args.base_trees, validation_global
    )
    train_x, _, train_signatures, head_names = _decision_matrix(
        arrays, train, names, args.slots
    )
    validation_x, validation_kept, validation_signatures, _ = _decision_matrix(
        arrays, validation, names, args.slots
    )
    signature_counts = Counter(train_signatures)
    ordered_signatures = sorted(signature_counts)
    class_by_signature = {
        signature: index for index, signature in enumerate(ordered_signatures)
    }
    train_y = np.asarray(
        [class_by_signature[signature] for signature in train_signatures],
        dtype=np.int32,
    )

    head = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=args.head_trees,
        learning_rate=0.025, num_leaves=63, min_child_samples=28,
        max_depth=-1, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.82, reg_alpha=0.25, reg_lambda=1.8,
        random_state=SEED + 900, n_jobs=20, verbosity=-1,
    )
    head.fit(
        train_x, train_y,
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][np.asarray([
                decision for decision in train
                if int(arrays["select_contexts"][decision]) == 0
                and not bool(arrays["forced"][decision])
                and int(arrays["chosen_counts"][decision]) == 1
            ], dtype=np.int64)],
            0.35, 2.0,
        ),
        feature_name=head_names,
        categorical_feature=_categorical(head_names),
    )

    count_names = arrays["count_feature_names"].astype(str).tolist()
    variable = train[arrays["minimums"][train] < arrays["maximums"][train]]
    count = lgb.LGBMRegressor(**v1._count_params(SEED, 250))
    count.fit(
        arrays["count_features"][variable], arrays["chosen_counts"][variable],
        feature_name=count_names,
        categorical_feature=v1._categorical_columns(count_names),
    )
    counts = v1._predict_counts(
        count, arrays["count_features"], validation,
        arrays["minimums"], arrays["maximums"], num_iteration=250,
    )
    base_rows = v1._rows_for(arrays["groups"], validation)
    base_scores = validation_global[base_rows]
    base_metrics = v1.evaluate(base_scores, validation, arrays, counts)

    experiments: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    known_validation = np.asarray([
        int(signature in class_by_signature) for signature in validation_signatures
    ], dtype=np.int8)
    for trees in range(100, args.head_trees + 1, 100):
        probabilities = head.predict_proba(validation_x, num_iteration=trees)
        class_predictions = probabilities.argmax(axis=1)
        signature_accuracy = float(np.mean([
            bool(known_validation[index])
            and ordered_signatures[int(class_predictions[index])] == signature
            for index, signature in enumerate(validation_signatures)
        ]))
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            scores = _scores(
                arrays, validation, validation_global, probabilities,
                validation_kept, class_by_signature, names, alpha,
            )
            metrics = v1.evaluate(scores, validation, arrays, counts)
            row = {
                "trees": trees, "alpha": alpha,
                "signature_accuracy": signature_accuracy,
                "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
                "single_top1": metrics["single_choice_semantic_top1"],
                "main_top1": metrics["main_single_choice_semantic_top1"],
            }
            experiments.append(row)
            if best is None or (
                row["nonforced_semantic_exact"], row["main_top1"], -row["trees"]
            ) > (
                best["nonforced_semantic_exact"], best["main_top1"], -best["trees"]
            ):
                best = row

    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()),
        "test_read": False,
        "classes": [
            {"signature": list(signature), "train_count": signature_counts[signature]}
            for signature in ordered_signatures
        ],
        "validation_known_signature_rate": float(known_validation.mean()),
        "base_validation": base_metrics,
        "selected": best,
        "fit_seconds": time.perf_counter() - started,
        "target": {
            "metric": "validation_nonforced_semantic_exact", "value": 0.85,
            "met": bool(best and best["nonforced_semantic_exact"] >= 0.85),
        },
        "experiments": experiments,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "base": base_metrics["nonforced_semantic_exact"],
        "selected": best, "target": report["target"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
