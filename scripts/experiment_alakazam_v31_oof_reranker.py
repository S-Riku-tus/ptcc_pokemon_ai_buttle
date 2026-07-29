"""OOF-trained specialist that only challenges the base ranker's top choice."""

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


CRITICAL = (
    "option_type",
    "candidate_card_id",
    "candidate_attack_id",
    "candidate_area",
    "candidate_inplay_area",
    "candidate_target_id",
    "action_type",
    "fallback_selected",
    "v29_selected",
)


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _row(
    base: np.ndarray,
    challenger: np.ndarray,
    *,
    base_score: float,
    challenger_score: float,
    challenger_rank: int,
    candidate_start: int,
    critical: list[int],
) -> np.ndarray:
    return np.concatenate((
        base[:candidate_start],
        challenger[candidate_start:] - base[candidate_start:],
        base[critical],
        challenger[critical],
        np.asarray([
            base_score,
            challenger_score,
            challenger_score - base_score,
            challenger_rank,
        ], dtype=np.float32),
    ))


def _schema(names: list[str]) -> tuple[int, list[int], list[str]]:
    candidate_start = names.index("option_type")
    critical = [names.index(name) for name in CRITICAL]
    output = (
        [f"state__{name}" for name in names[:candidate_start]]
        + [f"delta__{name}" for name in names[candidate_start:]]
        + [f"base__{names[index]}" for index in critical]
        + [f"challenger__{names[index]}" for index in critical]
        + ["base_score", "challenger_score", "score_delta", "challenger_rank"]
    )
    return candidate_start, critical, output


def _oof_scores(
    arrays: dict[str, Any],
    names: list[str],
    train: np.ndarray,
    folds: int,
) -> np.ndarray:
    result = np.full(len(arrays["labels"]), np.nan, dtype=np.float32)
    episodes = arrays["episode_ids"]
    unique = np.unique(episodes[train])
    episode_fold = {
        int(episode): index % folds
        for index, episode in enumerate(unique)
    }
    for fold in range(folds):
        heldout = train[np.asarray([
            episode_fold[int(episodes[index])] == fold for index in train
        ])]
        fitted = train[np.asarray([
            episode_fold[int(episodes[index])] != fold for index in train
        ])]
        model = teacher._fit(
            arrays,
            names,
            fitted,
            n_estimators=360,
        )
        x, _, _, _ = teacher._select_decisions(arrays, heldout)
        scores = model.predict(x).astype(np.float32)
        starts, ends = _ranges(arrays["groups"])
        offset = 0
        for decision in heldout:
            size = ends[decision] - starts[decision]
            result[starts[decision]:ends[decision]] = scores[offset:offset + size]
            offset += size
    if np.any(np.isnan(np.concatenate([
        result[_ranges(arrays["groups"])[0][index]:_ranges(arrays["groups"])[1][index]]
        for index in train
    ]))):
        raise RuntimeError("OOF scores incomplete")
    return result


def _examples(
    arrays: dict[str, Any],
    decisions: np.ndarray,
    scores: np.ndarray,
    candidate_start: int,
    critical: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends = _ranges(arrays["groups"])
    rows = []
    labels = []
    weights = []
    for decision in decisions:
        start, end = starts[decision], ends[decision]
        order = np.argsort(-scores[start:end], kind="stable")[:3]
        positive = int(np.flatnonzero(arrays["labels"][start:end] == 1)[0])
        base = int(order[0])
        if positive not in order:
            continue
        for rank, challenger in enumerate(order[1:], start=2):
            if positive != base and int(challenger) != positive:
                continue
            rows.append(_row(
                arrays["features"][start + base],
                arrays["features"][start + challenger],
                base_score=float(scores[start + base]),
                challenger_score=float(scores[start + challenger]),
                challenger_rank=rank,
                candidate_start=candidate_start,
                critical=critical,
            ))
            labels.append(int(int(challenger) == positive))
            weights.append(float(arrays["weights"][start]))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float32),
    )


def _predictions(
    model: lgb.LGBMClassifier,
    arrays: dict[str, Any],
    decisions: np.ndarray,
    scores: np.ndarray,
    candidate_start: int,
    critical: list[int],
) -> dict[str, np.ndarray]:
    starts, ends = _ranges(arrays["groups"])
    rows = []
    groups = []
    base_correct = []
    challenger_correct = []
    for decision in decisions:
        start, end = starts[decision], ends[decision]
        order = np.argsort(-scores[start:end], kind="stable")[:3]
        base = int(order[0])
        decision_rows = [
            _row(
                arrays["features"][start + base],
                arrays["features"][start + challenger],
                base_score=float(scores[start + base]),
                challenger_score=float(scores[start + challenger]),
                challenger_rank=rank,
                candidate_start=candidate_start,
                critical=critical,
            )
            for rank, challenger in enumerate(order[1:], start=2)
        ]
        rows.extend(decision_rows)
        groups.append(len(decision_rows))
        base_correct.append(int(arrays["labels"][start + base] == 1))
        challenger_correct.extend(
            int(arrays["labels"][start + challenger] == 1)
            for challenger in order[1:]
        )
    probabilities = model.predict_proba(
        np.asarray(rows, dtype=np.float32)
    )[:, 1]
    return {
        "probabilities": probabilities,
        "groups": np.asarray(groups, dtype=np.int8),
        "base_correct": np.asarray(base_correct, dtype=np.int8),
        "challenger_correct": np.asarray(challenger_correct, dtype=np.int8),
    }


def _accuracy(predictions: dict[str, np.ndarray], threshold: float) -> float:
    starts, ends = _ranges(predictions["groups"])
    correct = 0
    for decision, (start, end) in enumerate(zip(starts, ends)):
        probabilities = predictions["probabilities"][start:end]
        challenger_index = int(np.argmax(probabilities))
        correct += int(
            predictions["challenger_correct"][start + challenger_index]
            if probabilities[challenger_index] >= threshold
            else predictions["base_correct"][decision]
        )
    return correct / len(predictions["groups"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
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
    oof_cache = args.output.with_suffix(".oof.npy")
    if oof_cache.exists():
        oof = np.load(oof_cache, allow_pickle=False)
        if oof.shape != (len(arrays["labels"]),):
            raise ValueError(f"Invalid OOF cache shape: {oof.shape}")
    else:
        oof = _oof_scores(arrays, names, train, args.folds)
        oof_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(oof_cache, oof, allow_pickle=False)
    candidate_start, critical, pair_names = _schema(names)
    train_x, train_y, train_weights = _examples(
        arrays, train, oof, candidate_start, critical
    )
    validation_x, validation_y, _ = _examples(
        arrays, validation, base_scores, candidate_start, critical
    )
    experiments = []
    best_model = None
    best = None
    for positive_weight in (1.0, 2.0, 4.0):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            learning_rate=0.025,
            num_leaves=127,
            min_child_samples=30,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.2,
            reg_lambda=1.0,
            scale_pos_weight=positive_weight,
            random_state=741,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(
            train_x,
            train_y,
            sample_weight=train_weights,
            feature_name=pair_names,
            eval_set=[(validation_x, validation_y)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        validation_predictions = _predictions(
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
                "top1": _accuracy(
                    validation_predictions,
                    float(threshold),
                ),
            }
            for threshold in np.arange(0.30, 0.951, 0.025)
        ]
        row = {
            "positive_weight": positive_weight,
            "best_iteration": int(model.best_iteration_ or 1000),
            **max(grid, key=lambda value: value["top1"]),
        }
        experiments.append(row)
        print(row, flush=True)
        if best is None or row["top1"] > best["top1"]:
            best = row
            best_model = model
    assert best is not None and best_model is not None
    validation_predictions = _predictions(
        best_model,
        arrays,
        validation,
        base_scores,
        candidate_start,
        critical,
    )
    test_predictions = _predictions(
        best_model,
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
        "base_best_iteration": int(base.best_iteration_ or 900),
        "experiments": experiments,
        "selected": best,
        "validation_base": _accuracy(validation_predictions, 1.1),
        "test_base": _accuracy(test_predictions, 1.1),
        "test_top1": _accuracy(
            test_predictions, float(best["threshold"])
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
