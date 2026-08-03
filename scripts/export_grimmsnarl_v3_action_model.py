"""Export the v3 multiclass action prior to a stdlib-only JSON model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import _compact_node  # noqa: E402


def _tree_value(row: np.ndarray, tree: dict) -> float:
    node = tree
    while "v" not in node:
        value = float(row[node["f"]])
        if value != value:
            go_left = bool(node.get("x", True))
        elif node.get("d", "<=") == "==":
            go_left = int(round(value)) in node.get("c", ())
        else:
            go_left = value <= float(node["t"])
        node = node["l"] if go_left else node["r"]
    return float(node["v"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--guard-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    guard = json.loads(args.guard_report.read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(args.model))
    dumped = booster.dump_model()
    classes = [int(value) for value in training["classes"]]
    class_count = int(dumped.get("num_tree_per_iteration", 1))
    if class_count != len(classes):
        raise SystemExit(
            f"class mismatch: booster={class_count}, report={len(classes)}"
        )
    best_iteration = int(training["best_iteration"])
    tree_info = dumped["tree_info"][:best_iteration * class_count]
    class_trees = [[] for _ in classes]
    for index, tree in enumerate(tree_info):
        class_trees[index % class_count].append(
            _compact_node(tree["tree_structure"])
        )
    if any(len(trees) != best_iteration for trees in class_trees):
        raise SystemExit("incomplete multiclass tree group")

    # Exporting multiclass trees is easy to get subtly wrong because LightGBM
    # interleaves classes by iteration. Compare native raw logits with the
    # exact class grouping that the stdlib runtime will walk.
    rng = np.random.default_rng(20260803)
    probes = rng.integers(
        -1, 7, size=(24, len(booster.feature_name()))
    ).astype(np.float32)
    native = np.asarray(
        booster.predict(
            probes, raw_score=True, num_iteration=best_iteration
        )
    )
    compact = np.asarray([
        [sum(_tree_value(row, tree) for tree in trees)
         for trees in class_trees]
        for row in probes
    ])
    max_abs_error = float(np.max(np.abs(native - compact)))
    if max_abs_error > 1e-9:
        raise SystemExit(
            f"multiclass export parity failed: {max_abs_error:.3e}"
        )

    selected = guard["selected"]
    model = {
        "format": "lightgbm_multiclass_tree_v1",
        "kind": "grimmsnarl_elite_action_prior",
        "feature_names": list(booster.feature_name()),
        "classes": classes,
        "class_trees": class_trees,
        # log-softmax differs from raw class score by one decision-wide
        # constant, so argmax blending can use the cheaper raw logits.
        "blend_alpha": float(selected["alpha"]),
        "selection_rule": guard["selection_rule"],
        "elite_validation_delta": float(selected["elite_delta"]),
        "pinned_validation_delta": float(selected["pinned_delta"]),
        "elite_test_delta": float(guard["test"]["elite"]["delta"]),
        "pinned_test_delta": float(guard["test"]["pinned"]["delta"]),
        "best_iteration": best_iteration,
        "export_probe_max_abs_error": max_abs_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "features": len(model["feature_names"]),
        "classes": classes,
        "trees_per_class": [len(trees) for trees in class_trees],
        "alpha": model["blend_alpha"],
        "max_abs_error": max_abs_error,
        "bytes": args.output.stat().st_size,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
