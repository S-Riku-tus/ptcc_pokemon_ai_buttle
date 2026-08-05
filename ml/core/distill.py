from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _categorical_values(threshold: Any) -> list[int]:
    if isinstance(threshold, str):
        return [int(value) for value in threshold.split("||") if value != ""]
    if isinstance(threshold, (int, float)):
        return [int(threshold)]
    return []


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    if "leaf_value" in node:
        return {"v": float(node["leaf_value"])}
    decision = str(node.get("decision_type", "<="))
    compact = {
        "f": int(node["split_feature"]),
        "d": decision,
        "l": _compact_node(node["left_child"]),
        "r": _compact_node(node["right_child"]),
        "x": bool(node.get("default_left", True)),
    }
    if decision == "==":
        compact["c"] = _categorical_values(node.get("threshold"))
    else:
        compact["t"] = float(node["threshold"])
    return compact


def compact_booster(
    booster,
    kind: str = "ranker",
    num_iteration: int | None = None,
) -> dict[str, Any]:
    dumped = booster.dump_model(num_iteration=num_iteration)
    return {
        "format": "lightgbm_tree_v2",
        "kind": kind,
        "feature_names": list(booster.feature_name()),
        "average_output": bool(dumped.get("average_output", False)),
        "trees": [_compact_node(tree["tree_structure"]) for tree in dumped["tree_info"]],
    }


def tree_score(features: list[float], model: dict[str, Any]) -> float:
    total = 0.0
    for tree in model["trees"]:
        node = tree
        while "v" not in node:
            value = features[node["f"]]
            if value != value:
                go_left = node.get("x", True)
            elif node.get("d", "<=") == "==":
                go_left = int(round(value)) in set(node.get("c", []))
            else:
                go_left = value <= node["t"]
            node = node["l"] if go_left else node["r"]
        total += node["v"]
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


def write_compact_model(booster, path: Path, kind: str = "ranker") -> dict[str, Any]:
    model = compact_booster(booster, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    args = parser.parse_args()
    import lightgbm as lgb
    booster = lgb.Booster(model_file=args.model)
    write_compact_model(booster, Path(args.output))


if __name__ == "__main__":
    main()
