"""Replay the exported Lopunny runtime and audit legality/agreement/latency.

The exported deployment model is refit on all 386 trajectories, so agreement
reported here is a *runtime/resubstitution parity* check, not the generalization
estimate.  The honest chronological estimate remains ``training_report.json``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.agent_loader import load_dir_agent  # noqa: E402


DEFAULT_RUN = (
    ROOT / "data" / "runs" / "leaderboard_top1" / "non_alakazam"
    / "20260801_rank01_sub55137818"
)
DEFAULT_AGENT = ROOT / "agents" / "lopunny" / "majkel_lopunny_ml_v1"


def _load_features(agent_dir: Path):
    path = agent_dir / "imitation_features.py"
    spec = importlib.util.spec_from_file_location("lopunny_eval_features", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic_counter(
    module: Any,
    observation: dict[str, Any],
    action: list[int],
) -> Counter[tuple[int | float, ...]]:
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    options = list(select.get("option") or [])
    return Counter(
        module.semantic_option_key(current, select, options[index], index)
        for index in action
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        episode_ids = cached["episode_ids"]
        splits = cached["splits"].astype(str)
    allowed = (
        set(map(int, np.unique(episode_ids)))
        if args.split == "all"
        else set(map(int, np.unique(episode_ids[splits == args.split])))
    )

    feature_module = _load_features(args.agent_dir.resolve())
    _, _, agent_module = load_dir_agent(args.agent_dir.resolve())
    with (args.run_dir / "manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))

    totals: Counter[str] = Counter()
    latencies: list[float] = []
    for row in manifest:
        if row.get("error"):
            continue
        episode_id = int(row["episode_id"])
        if episode_id not in allowed:
            continue
        seat = int(row["detected_submission_agent_index"])
        replay_path = (
            args.run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not replay_path.exists():
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        agent_module.agent({"select": None})
        totals["episodes"] += 1
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select")
            teacher = (steps[step_index + 1][seat] or {}).get("action")
            if (
                record.get("status") != "ACTIVE"
                or not isinstance(select, dict)
                or not isinstance(teacher, list)
                or not select.get("option")
            ):
                continue
            started = time.perf_counter()
            predicted = agent_module.agent(observation)
            latencies.append((time.perf_counter() - started) * 1000.0)
            options = list(select.get("option") or [])
            minimum = max(0, min(int(select.get("minCount") or 0), len(options)))
            maximum = max(
                minimum, min(int(select.get("maxCount") or 0), len(options))
            )
            legal = (
                isinstance(predicted, list)
                and minimum <= len(predicted) <= maximum
                and len(set(predicted)) == len(predicted)
                and all(isinstance(index, int) and 0 <= index < len(options) for index in predicted)
            )
            totals["decisions"] += 1
            totals["legal"] += int(legal)
            if not legal:
                continue
            totals["raw_exact"] += int(set(predicted) == set(teacher))
            totals["count_correct"] += int(len(predicted) == len(teacher))
            totals["semantic_exact"] += int(
                _semantic_counter(feature_module, observation, predicted)
                == _semantic_counter(feature_module, observation, teacher)
            )

    snapshot = agent_module.diag_snapshot()
    count = max(1, totals["decisions"])
    sorted_latency = sorted(latencies)
    p95_index = min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95))
    report = {
        "scope": args.split,
        "interpretation": "deployment-refit runtime parity/resubstitution; not holdout evidence",
        "episodes": totals["episodes"],
        "decisions": totals["decisions"],
        "legal_rate": totals["legal"] / count,
        "semantic_exact": totals["semantic_exact"] / count,
        "raw_exact": totals["raw_exact"] / count,
        "count_accuracy": totals["count_correct"] / count,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": sorted_latency[p95_index] if sorted_latency else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "diagnostics": snapshot,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
