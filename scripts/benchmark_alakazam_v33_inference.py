"""Measure v33 decision latency against v32 on real holdout observations.

v33 ships 1,005 trees where v32 shipped 692, and the submission runtime walks
them in pure Python, so the extra accuracy has to be paid for in wall clock.
Kaggle stops the agent on ``remainingOverageTime``, and the runtime already
falls back below two seconds, so this checks the margin is comfortable.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402


def read_replay(row: dict[str, str], archives: dict[str, Any]) -> dict:
    path = row["storage_path"]
    if row["storage_type"] == "zip":
        archive = archives.get(path)
        if archive is None:
            archive = zipfile.ZipFile(path)
            archives[path] = archive
        return json.loads(archive.read(row["replay_path"]))
    return json.loads(
        (Path(path) / row["replay_path"]).read_text(encoding="utf-8")
    )


def collect_observations(index: Path, episodes: int) -> list[dict]:
    with index.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(
            csv.DictReader(handle),
            key=lambda r: int(r["episode_id"]),
            reverse=True,
        )
    archives: dict[str, Any] = {}
    observations = []
    for row in rows[:episodes]:
        replay = read_replay(row, archives)
        seat = int(row["seat_index"])
        for step in (replay.get("steps") or [])[:-1]:
            if seat >= len(step):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            if (
                record.get("status") == "ACTIVE"
                and int(select.get("context", -1)) == 0
                and int(select.get("type", -1)) == 0
                and int(select.get("minCount") or 0) == 1
                and int(select.get("maxCount") or 0) == 1
                and len(options) >= 2
            ):
                observations.append(observation)
    for archive in archives.values():
        archive.close()
    return observations


def measure(agent_dir: Path, observations: list[dict]) -> dict[str, Any]:
    _, _, main_module = load_dir_agent(agent_dir.resolve())
    agent = main_module.agent
    main_module.diag_reset()
    timings = []
    for observation in observations:
        started = time.perf_counter()
        agent(observation)
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    snapshot = main_module.diag_snapshot()
    return {
        "agent": agent_dir.name,
        "decisions": len(timings),
        "mean_ms": statistics.fmean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": timings[int(len(timings) * 0.95)],
        "max_ms": timings[-1],
        "total_seconds": sum(timings) / 1000.0,
        "ml_model_rate": snapshot["ml"].get("model_rate"),
        "ml_memory_rate": snapshot["ml"].get("memory_rate"),
        "ml_fallback_rate": snapshot["ml"].get("fallback_rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("--agent", action="append", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations = collect_observations(args.teacher_index, args.episodes)
    print(f"collected {len(observations)} scoped decisions", flush=True)

    results = []
    for agent_dir in args.agent:
        # Each agent has to run in its own process: the loader installs the
        # agent directory on sys.path and the module names collide.
        import subprocess
        completed = subprocess.run(
            [
                sys.executable, __file__, str(args.teacher_index),
                "--agent", str(agent_dir), "--episodes", str(args.episodes),
                "--output", str(args.output.with_suffix(".partial.json")),
                "--_single",
            ],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-2000:])
        results.append(
            json.loads(
                args.output.with_suffix(".partial.json").read_text("utf-8")
            )
        )
        print(json.dumps(results[-1]), flush=True)

    args.output.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def single() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--_single", action="store_true")
    args = parser.parse_args()
    observations = collect_observations(args.teacher_index, args.episodes)
    result = measure(args.agent, observations)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(single() if "--_single" in sys.argv else main())
