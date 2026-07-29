"""Compare exported v31/v32 tree policies on one untouched teacher cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _aligned_rows(cache: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    cached_names = cache["feature_names"].astype(str).tolist()
    source = cache["features"]
    by_name = {name: index for index, name in enumerate(cached_names)}
    rows = np.full((len(source), len(names)), -1.0, dtype=np.float32)
    for target, name in enumerate(names):
        source_index = by_name.get(name)
        if source_index is not None:
            rows[:, target] = source[:, source_index]
    return rows


def _score_node(
    node: dict[str, Any],
    rows: np.ndarray,
    indices: np.ndarray,
    output: np.ndarray,
) -> None:
    if not len(indices):
        return
    if "v" in node:
        output[indices] = float(node["v"])
        return
    values = rows[indices, int(node["f"])]
    missing = np.isnan(values)
    if node.get("d", "<=") == "==":
        go_left = np.isin(
            np.rint(values).astype(np.int64),
            np.asarray(node.get("c") or (), dtype=np.int64),
        )
    else:
        go_left = values <= float(node["t"])
    if bool(node.get("x", True)):
        go_left |= missing
    else:
        go_left &= ~missing
    _score_node(node["l"], rows, indices[go_left], output)
    _score_node(node["r"], rows, indices[~go_left], output)


def _score_model(model: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(rows), dtype=np.float64)
    all_indices = np.arange(len(rows), dtype=np.int64)
    tree_output = np.empty(len(rows), dtype=np.float64)
    for tree in model["trees"]:
        _score_node(tree, rows, all_indices, tree_output)
        scores += tree_output
    if model.get("average_output") and model["trees"]:
        scores /= len(model["trees"])
    return scores


def _normalize(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.empty_like(scores)
    start = 0
    for size in groups:
        end = start + int(size)
        values = scores[start:end]
        scale = max(float(values.std()), 1e-5)
        result[start:end] = (values - float(values.mean())) / scale
        start = end
    return result


def _accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    correct = 0
    start = 0
    for size in groups:
        end = start + int(size)
        correct += int(labels[start + int(np.argmax(scores[start:end]))] == 1)
        start = end
    return correct / max(1, len(groups))


def _load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cache",
        type=Path,
        help="Teacher cache not used by either final model, e.g. rmy_arrays.npz.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--skip-v32", action="store_true")
    args = parser.parse_args()

    v31_dir = ROOT / "agents" / "alakazam" / "alakazam_ml_v31"
    v32_dir = ROOT / "agents" / "alakazam" / "alakazam_ml_v32"
    models = {
        "v31_large": _load_model(v31_dir / "ranker_model.json"),
        "v31_numeric": _load_model(v31_dir / "ranker_numeric_model.json"),
    }
    if not args.skip_v32:
        models["v32_large"] = _load_model(v32_dir / "ranker_model.json")

    with np.load(args.cache, allow_pickle=False) as loaded:
        cache = {name: loaded[name] for name in loaded.files}
    decision_mask = cache["splits"].astype(str) == args.split
    row_mask = np.repeat(decision_mask, cache["groups"].astype(np.int64))
    groups = cache["groups"][decision_mask].astype(np.int64)
    labels = cache["labels"][row_mask]

    scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        rows = _aligned_rows(cache, model["feature_names"])[row_mask]
        scores[name] = _normalize(_score_model(model, rows), groups)

    v31 = scores["v31_large"] + 1.3 * scores["v31_numeric"]
    v32 = scores.get("v32_large")
    report = {
        "cache": str(args.cache.resolve()),
        "split": args.split,
        "decisions": int(len(groups)),
        "candidate_rows": int(len(labels)),
        "v31_exported_ensemble_top1": _accuracy(v31, labels, groups),
    }
    if v32 is not None:
        report.update({
            "v32_exported_large_leaf_top1": _accuracy(v32, labels, groups),
            "v32_minus_v31_points": (
                _accuracy(v32, labels, groups)
                - _accuracy(v31, labels, groups)
            ),
        })
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.scores_output:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.scores_output,
            v31_scores=v31,
            labels=labels,
            groups=groups,
        )


if __name__ == "__main__":
    main()
