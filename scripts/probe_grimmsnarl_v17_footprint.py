"""Isolate v17's wall-break changes on all 110 stored v15 games.

v16 and v17 have the same base policy; only ``wall_break.py`` differs.  This
probe therefore feeds the action actually chosen by v15 through each guard,
advancing both guards over the complete replay.  It measures the exact extra
footprint without paying to rescore thousands of unchanged decisions through
the 2,000-tree ranker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts",
             ROOT / "agents/grimmsnarl/grimmsnarl_ml_v16"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402
from probe_grimmsnarl_v16_footprint import episodes, shape, single  # noqa: E402


def load_guard(version: str) -> Any:
    path = (ROOT / "agents/grimmsnarl" / f"grimmsnarl_ml_{version}"
            / "wall_break.py")
    spec = importlib.util.spec_from_file_location(
        f"wall_break_{version}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WallBreakGuard()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    totals: Counter[str] = Counter()
    by_matchup: dict[str, Counter[str]] = defaultdict(Counter)
    by_shape: Counter[str] = Counter()
    touched: list[dict[str, Any]] = []
    stats16: Counter[str] = Counter()
    stats17: Counter[str] = Counter()

    for episode_id, replay, seat, matchup in episodes():
        left, right = load_guard("v16"), load_guard("v17")
        differences = 0
        steps = replay.get("steps") or []
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single((steps[index + 1][seat] or {}).get("action"))
            if played is None:
                continue
            left.note(observation)
            right.note(observation)
            a = left.adjust(observation, select, played, [played])
            b = right.adjust(observation, select, played, [played])
            totals["decisions"] += 1
            by_matchup[matchup]["decisions"] += 1
            if a != b:
                differences += 1
                totals["differences"] += 1
                by_matchup[matchup]["differences"] += 1
                by_shape[
                    f"{shape(observation, a)} -> {shape(observation, b)}"
                ] += 1
        totals["games"] += 1
        stats16.update(left.stats)
        stats17.update(right.stats)
        if differences:
            touched.append({
                "episode_id": episode_id,
                "matchup": matchup,
                "differences": differences,
            })

    output = {
        "comparison": "v16 wall_break -> v17 wall_break",
        "totals": dict(totals),
        "by_matchup": {
            name: dict(block) for name, block in sorted(by_matchup.items())
        },
        "by_shape": dict(by_shape.most_common()),
        "episodes_touched": sorted(
            touched, key=lambda row: (-row["differences"], row["episode_id"])
        ),
        "v16_guard": dict(stats16),
        "v17_guard": dict(stats17),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
