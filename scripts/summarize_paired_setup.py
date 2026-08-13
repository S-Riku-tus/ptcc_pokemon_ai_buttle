"""Summarize setup-funnel changes from a traced CRN gauntlet report."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


Metric = tuple[str, Callable[[dict[str, Any]], float | None]]


def _initial(field: str) -> Callable[[dict[str, Any]], float | None]:
    def read(funnel: dict[str, Any]) -> float | None:
        initial = funnel.get("initial")
        return float(initial.get(field, 0)) if isinstance(initial, dict) else None
    return read


def _number(field: str) -> Callable[[dict[str, Any]], float | None]:
    def read(funnel: dict[str, Any]) -> float | None:
        value = funnel.get(field)
        return float(value) if isinstance(value, (int, float)) else None
    return read


def _shadow_by(turn: int) -> Callable[[dict[str, Any]], float | None]:
    def read(funnel: dict[str, Any]) -> float | None:
        value = funnel.get("first_shadow_own_turn")
        return float(value is not None and int(value) <= turn)
    return read


METRICS: tuple[Metric, ...] = (
    ("initial_impidimp", _initial("impidimp")),
    ("initial_impidimp_present", lambda f: float((_initial("impidimp")(f) or 0) > 0)),
    ("impidimp_by_turn2", _number("impidimp_by_turn2")),
    ("grimmsnarl_by_turn2", lambda f: float((_number("grimmsnarl_by_turn2")(f) or 0) > 0)),
    ("ready_grimmsnarl_by_turn2", lambda f: float((_number("ready_grimmsnarl_by_turn2")(f) or 0) > 0)),
    ("spikemuth_by_turn2", lambda f: float((_number("spikemuth_by_turn2")(f) or 0) > 0)),
    ("max_dark_energy_by_turn2", _number("max_dark_energy_by_turn2")),
    ("shadow_by_turn2", _shadow_by(2)),
    ("shadow_by_turn3", _shadow_by(3)),
)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        row for row in rows
        if row.get("repeat") == 0 and not row.get("error")
        and isinstance(row.get("setup_funnel"), dict)
    ]
    treatments: dict[str, dict[str, Any]] = {}
    for treatment in ("champion", "challenger"):
        subset = [row for row in primary if row.get("treatment") == treatment]
        treatments[treatment] = {
            "games": len(subset),
            "win_rate": _mean([float(row["evaluated_win"]) for row in subset]),
            "metrics": {
                name: _mean([
                    value for row in subset
                    if (value := reader(row["setup_funnel"])) is not None
                ])
                for name, reader in METRICS
            },
        }

    pairs: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in primary:
        key = (str(row["opponent"]), int(row["seed"]), int(row["evaluated_seat"]))
        pairs[key][str(row["treatment"])] = row

    effects: dict[str, Any] = {}
    for name, reader in METRICS:
        clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
        for (opponent, seed, _seat), values in pairs.items():
            if set(values) != {"champion", "challenger"}:
                continue
            left = reader(values["champion"]["setup_funnel"])
            right = reader(values["challenger"]["setup_funnel"])
            if left is not None and right is not None:
                clusters[(opponent, seed)].append(right - left)
        cluster_values = [statistics.fmean(values) for values in clusters.values()]
        effect = _mean(cluster_values)
        if len(cluster_values) >= 2:
            se = statistics.stdev(cluster_values) / math.sqrt(len(cluster_values))
            ci = [effect - 1.96 * se, effect + 1.96 * se]
        else:
            se, ci = None, [None, None]
        effects[name] = {
            "seed_clusters": len(cluster_values),
            "challenger_minus_champion": effect,
            "cluster_standard_error": se,
            "cluster_95ci": ci,
        }
    return {"treatments": treatments, "paired_effects": effects}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8"))
    report = {
        "source": str(args.report),
        "overall": summarize(source.get("rows", [])),
        "by_opponent": {
            label: summarize([
                row for row in source.get("rows", []) if row.get("opponent") == label
            ])
            for label in source.get("by_opponent", {})
        },
        "by_evaluated_order": {
            label: summarize([
                row for row in source.get("rows", [])
                if row.get("evaluated_went_first") == value
            ])
            for label, value in (("first", 1), ("second", 0))
        },
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"report: {args.out.resolve()}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
