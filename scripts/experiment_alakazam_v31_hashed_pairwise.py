"""Learn sparse state/action crosses with an online pairwise imitation model.

Tree rankers must rediscover every action timing conjunction.  This probe
hashes explicit ``candidate x state`` tokens and trains on expert-vs-legal
option differences while retaining the frozen chronological split.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-5:
        return str(rounded)
    return f"{float(value):.3g}"


class Tokenizer:
    def __init__(
        self,
        features: np.ndarray,
        names: list[str],
        state_columns: list[int],
    ) -> None:
        self.features = features
        self.names = names
        self.state_columns = state_columns
        self.option_type = names.index("option_type")
        self.action = names.index("action_type")
        self.card = names.index("candidate_card_id")
        self.target = names.index("candidate_target_id")
        self.area = names.index("candidate_inplay_area")

    def tokens(self, first_row: int, row: int) -> dict[str, float]:
        state = self.features[first_row]
        candidate = self.features[row]
        action = int(candidate[self.action])
        card = int(candidate[self.card])
        target = int(candidate[self.target])
        area = int(candidate[self.area])
        action_key = f"a{action}"
        card_key = f"a{action}c{card}"
        full_key = f"a{action}c{card}t{target}r{area}"
        out: dict[str, float] = {
            f"candidate:{action_key}": 1.0,
            f"candidate:{card_key}": 1.0,
            f"candidate:{full_key}": 1.0,
        }
        for column in range(self.option_type, len(self.names)):
            value = float(candidate[column])
            if value in (0.0, -1.0) or not np.isfinite(value):
                continue
            out[f"c:{self.names[column]}={_value(value)}"] = 1.0
        for column in self.state_columns:
            value = float(state[column])
            token = f"{self.names[column]}={_value(value)}"
            out[f"xa:{action_key}|{token}"] = 1.0
            out[f"xac:{card_key}|{token}"] = 1.0
        return out


def _difference(
    positive: dict[str, float],
    negative: dict[str, float],
    reverse: bool,
) -> dict[str, float]:
    left, right = (negative, positive) if reverse else (positive, negative)
    result = dict(left)
    for token, value in right.items():
        result[token] = result.get(token, 0.0) - value
        if result[token] == 0.0:
            del result[token]
    return result


def _batches(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _candidate_matrix(
    tokenizer: Tokenizer,
    hasher: FeatureHasher,
    starts: np.ndarray,
    ends: np.ndarray,
    decisions: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[dict[str, float]] = []
    groups = []
    matrices = []
    for batch in _batches(decisions, 500):
        rows.clear()
        for decision in batch:
            start, end = starts[decision], ends[decision]
            groups.append(end - start)
            rows.extend(
                tokenizer.tokens(start, row)
                for row in range(start, end)
            )
        matrices.append(hasher.transform(rows).tocsr())
    return sparse.vstack(matrices, format="csr"), np.asarray(groups)


def _accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    starts, ends = _ranges(groups)
    return float(np.mean([
        labels[start + int(np.argmax(scores[start:end]))] == 1
        for start, end in zip(starts, ends)
    ]))


def _normalized_base(
    scores: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    starts, ends = _ranges(groups)
    result = scores.copy().astype(np.float32)
    for start, end in zip(starts, ends):
        values = result[start:end]
        result[start:end] = (
            (values - float(values.mean()))
            / max(float(values.std()), 1e-5)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--state-features", type=int, default=120)
    parser.add_argument("--hash-bits", type=int, default=20)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays: dict[str, Any] = {
            key: cached[key]
            for key in ("features", "labels", "weights", "groups")
        }
        splits = cached["splits"].astype(str)
        names = cached["feature_names"].astype(str).tolist()
    starts, ends = _ranges(arrays["groups"])
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")

    ranker = teacher._fit(
        arrays,
        names,
        train,
        n_estimators=900,
        validation_indices=validation,
    )
    importance = ranker.booster_.feature_importance(importance_type="gain")
    candidate_start = names.index("option_type")
    state_columns = sorted(
        sorted(
            range(candidate_start),
            key=lambda column: float(importance[column]),
            reverse=True,
        )[:args.state_features]
    )
    tokenizer = Tokenizer(arrays["features"], names, state_columns)
    hasher = FeatureHasher(
        n_features=1 << args.hash_bits,
        input_type="dict",
        alternate_sign=True,
    )
    validation_x, validation_groups = _candidate_matrix(
        tokenizer, hasher, starts, ends, validation
    )
    test_x, test_groups = _candidate_matrix(
        tokenizer, hasher, starts, ends, test
    )
    validation_labels = np.concatenate([
        arrays["labels"][starts[index]:ends[index]]
        for index in validation
    ])
    test_labels = np.concatenate([
        arrays["labels"][starts[index]:ends[index]]
        for index in test
    ])
    validation_base = ranker.predict(
        np.concatenate([
            arrays["features"][starts[index]:ends[index]]
            for index in validation
        ])
    ).astype(np.float32)
    test_base = ranker.predict(
        np.concatenate([
            arrays["features"][starts[index]:ends[index]]
            for index in test
        ])
    ).astype(np.float32)
    validation_base = _normalized_base(validation_base, validation_groups)
    test_base = _normalized_base(test_base, test_groups)

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=2e-7,
        l1_ratio=0.03,
        fit_intercept=False,
        average=True,
        random_state=741,
        learning_rate="optimal",
    )
    rng = np.random.default_rng(741)
    history = []
    best_model = None
    best_row = None
    for epoch in range(args.epochs):
        shuffled = rng.permutation(train)
        for batch in _batches(shuffled, 700):
            pair_rows: list[dict[str, float]] = []
            pair_labels = []
            pair_weights = []
            for decision in batch:
                start, end = starts[decision], ends[decision]
                positive = int(
                    np.flatnonzero(arrays["labels"][start:end] == 1)[0]
                )
                positive_tokens = tokenizer.tokens(start, start + positive)
                for local, row in enumerate(range(start, end)):
                    if local == positive:
                        continue
                    reverse = bool((int(decision) + local + epoch) % 2)
                    pair_rows.append(_difference(
                        positive_tokens,
                        tokenizer.tokens(start, row),
                        reverse,
                    ))
                    pair_labels.append(0 if reverse else 1)
                    pair_weights.append(float(arrays["weights"][row]))
            pair_x = hasher.transform(pair_rows)
            classifier.partial_fit(
                pair_x,
                np.asarray(pair_labels, dtype=np.int8),
                classes=np.asarray([0, 1], dtype=np.int8),
                sample_weight=np.asarray(pair_weights, dtype=np.float64),
            )
        hashed = classifier.decision_function(validation_x).astype(np.float32)
        grid = []
        for alpha in np.arange(0.0, 3.01, 0.1):
            accuracy = _accuracy(
                hashed + float(alpha) * validation_base,
                validation_labels,
                validation_groups,
            )
            grid.append({"alpha": float(alpha), "top1": accuracy})
        row = {
            "epoch": epoch + 1,
            "hashed_top1": _accuracy(
                hashed, validation_labels, validation_groups
            ),
            **max(grid, key=lambda value: value["top1"]),
        }
        history.append(row)
        print(row, flush=True)
        if best_row is None or row["top1"] > best_row["top1"]:
            best_row = row
            best_model = copy.deepcopy(classifier)
    assert best_model is not None and best_row is not None
    test_hashed = best_model.decision_function(test_x).astype(np.float32)
    test_score = test_hashed + float(best_row["alpha"]) * test_base
    report = {
        "cache": str(args.cache.resolve()),
        "hash_features": 1 << args.hash_bits,
        "state_features": [names[column] for column in state_columns],
        "ranker_best_iteration": int(ranker.best_iteration_ or 900),
        "validation_history": history,
        "selected_epoch": best_row["epoch"],
        "selected_alpha": best_row["alpha"],
        "validation_top1": best_row["top1"],
        "test_top1": _accuracy(test_score, test_labels, test_groups),
        "test_hashed_only": _accuracy(
            test_hashed, test_labels, test_groups
        ),
        "test_base": _accuracy(test_base, test_labels, test_groups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "selected_epoch",
            "selected_alpha",
            "validation_top1",
            "test_top1",
            "test_hashed_only",
            "test_base",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
