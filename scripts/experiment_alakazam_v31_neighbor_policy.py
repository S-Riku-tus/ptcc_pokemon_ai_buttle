"""Retrieve teacher actions from nearby train-only strategic states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _decision_matrix(
    features: np.ndarray,
    groups: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[str]]:
    starts, _ = _ranges(groups)
    candidate_start = names.index("option_type")
    columns = list(range(candidate_start))
    columns.extend(
        index
        for index, name in enumerate(names)
        if name.startswith("offered_") and index not in columns
    )
    return features[starts][:, columns], [names[index] for index in columns]


def _signatures(
    arrays: dict[str, Any],
    names: list[str],
) -> list[tuple[int, int, int, int]]:
    starts, ends = _ranges(arrays["groups"])
    columns = [
        names.index("action_type"),
        names.index("candidate_card_id"),
        names.index("candidate_target_id"),
        names.index("candidate_inplay_area"),
    ]
    output = []
    for start, end in zip(starts, ends):
        positive = int(np.flatnonzero(arrays["labels"][start:end] == 1)[0])
        output.append(tuple(
            int(value)
            for value in arrays["features"][start + positive, columns]
        ))
    return output


def _base_scores(
    model: Any,
    arrays: dict[str, Any],
    decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, labels, groups = teacher._predict_for_decisions(
        model, arrays, decisions
    )
    starts, ends = _ranges(np.asarray(groups))
    for start, end in zip(starts, ends):
        values = scores[start:end]
        scores[start:end] = (
            (values - float(values.mean()))
            / max(float(values.std()), 1e-5)
        )
    return scores, labels, np.asarray(groups)


def _neighbor_scores(
    arrays: dict[str, Any],
    names: list[str],
    decisions: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    train_decisions: np.ndarray,
    teacher_signatures: list[tuple[int, int, int, int]],
) -> np.ndarray:
    starts, ends = _ranges(arrays["groups"])
    columns = [
        names.index("action_type"),
        names.index("candidate_card_id"),
        names.index("candidate_target_id"),
        names.index("candidate_inplay_area"),
    ]
    output = []
    for local, decision in enumerate(decisions):
        start, end = starts[decision], ends[decision]
        candidates = [
            tuple(int(value) for value in row)
            for row in arrays["features"][start:end, columns]
        ]
        votes = np.zeros(len(candidates), dtype=np.float32)
        distances = np.atleast_1d(neighbor_distances[local])
        neighbors = np.atleast_1d(neighbor_indices[local])
        scale = max(float(np.median(distances)), 1e-5)
        for distance, neighbor in zip(distances, neighbors):
            signature = teacher_signatures[int(train_decisions[neighbor])]
            weight = float(np.exp(-float(distance) / scale))
            for index, candidate in enumerate(candidates):
                if candidate == signature:
                    votes[index] += weight
                elif candidate[:2] == signature[:2]:
                    votes[index] += 0.45 * weight
                elif candidate[0] == signature[0]:
                    votes[index] += 0.08 * weight
        if votes.max() > 0:
            votes /= votes.max()
        output.append(votes)
    return np.concatenate(output)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.schema_cache, allow_pickle=False) as schema:
        desired = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        original_names = cached["feature_names"].astype(str).tolist()
        columns = [original_names.index(name) for name in desired]
        arrays: dict[str, Any] = {
            "features": cached["features"][:, columns],
        }
        for key in ("labels", "weights", "groups"):
            arrays[key] = cached[key]
        splits = cached["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    decision_x, decision_names = _decision_matrix(
        arrays["features"], arrays["groups"], desired
    )
    categorical = [
        index
        for index, name in enumerate(decision_names)
        if name.endswith("_id")
        or name in {"self_active_id", "opp_active_id", "stadium_id"}
    ]
    numeric = [
        index for index in range(len(decision_names))
        if index not in categorical
    ]
    preprocess = ColumnTransformer((
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ))
    transformed_train = preprocess.fit_transform(decision_x[train])
    transformed_validation = preprocess.transform(decision_x[validation])
    transformed_test = preprocess.transform(decision_x[test])
    svd = TruncatedSVD(n_components=64, random_state=741)
    train_embedding = svd.fit_transform(transformed_train)
    validation_embedding = svd.transform(transformed_validation)
    test_embedding = svd.transform(transformed_test)
    scale = np.maximum(train_embedding.std(axis=0), 1e-5)
    train_embedding /= scale
    validation_embedding /= scale
    test_embedding /= scale
    tree = cKDTree(train_embedding)
    validation_distance, validation_neighbor = tree.query(
        validation_embedding, k=50, workers=4
    )
    test_distance, test_neighbor = tree.query(
        test_embedding, k=50, workers=4
    )
    ranker = teacher._fit(
        arrays,
        desired,
        train,
        n_estimators=900,
        validation_indices=validation,
    )
    validation_base, validation_labels, validation_groups = _base_scores(
        ranker, arrays, validation
    )
    test_base, test_labels, test_groups = _base_scores(
        ranker, arrays, test
    )
    teacher_signatures = _signatures(arrays, desired)
    grid = []
    cached_validation_votes = {}
    for neighbors in (1, 3, 5, 10, 20, 50):
        votes = _neighbor_scores(
            arrays,
            desired,
            validation,
            validation_neighbor[:, :neighbors],
            validation_distance[:, :neighbors],
            train,
            teacher_signatures,
        )
        cached_validation_votes[neighbors] = votes
        for alpha in np.arange(0.0, 4.01, 0.10):
            grid.append({
                "neighbors": neighbors,
                "alpha": float(alpha),
                "top1": _accuracy(
                    validation_base + float(alpha) * votes,
                    validation_labels,
                    validation_groups,
                ),
            })
    selected = max(grid, key=lambda row: row["top1"])
    test_votes = _neighbor_scores(
        arrays,
        desired,
        test,
        test_neighbor[:, :int(selected["neighbors"])],
        test_distance[:, :int(selected["neighbors"])],
        train,
        teacher_signatures,
    )
    report = {
        "cache": str(args.cache.resolve()),
        "decision_features": len(decision_names),
        "embedding_features": 64,
        "svd_explained_variance": float(
            svd.explained_variance_ratio_.sum()
        ),
        "selected": selected,
        "validation_neighbor_only": _accuracy(
            cached_validation_votes[int(selected["neighbors"])],
            validation_labels,
            validation_groups,
        ),
        "test_base": _accuracy(test_base, test_labels, test_groups),
        "test_neighbor_only": _accuracy(
            test_votes, test_labels, test_groups
        ),
        "test_top1": _accuracy(
            test_base + float(selected["alpha"]) * test_votes,
            test_labels,
            test_groups,
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
