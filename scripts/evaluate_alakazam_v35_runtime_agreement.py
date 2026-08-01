"""Run a packaged agent over teacher states and score what it actually plays.

The offline pipeline scores a ranker; the ladder scores an agent. Between them
sit the memory fast path, the v29 baseline, and six safety guards that discard
the ranker's pick. This driver closes that gap by replaying the teacher's own
observations through a real agent directory and comparing the returned action
with the action the teacher took, so the reported agreement is end to end.

It doubles as the parity check for the simulated shell audit. Counts of guards
that fire before the ranker is consulted -- the lethal guard above all -- do
not depend on which model is loaded, so they must match the simulation
decision for decision. Guards that depend on the ranker's pick will not match
a model fitted on a different episode set, and are reported for shape only.

The agent runs in a subprocess because the loader installs the agent directory
on ``sys.path`` and the module names collide between versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))


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


def episode_rows(index: Path, min_episode: int, limit: int):
    with index.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unique: dict[int, dict[str, str]] = {}
    for row in rows:
        unique.setdefault(int(row["episode_id"]), row)
    picked = [
        unique[key] for key in sorted(unique) if key >= min_episode
    ]
    return picked[:limit] if limit else picked


def collect(index: Path, min_episode: int, limit: int):
    """Scoped MAIN decisions with the action the teacher recorded next."""
    archives: dict[str, Any] = {}
    samples = []
    for row in episode_rows(index, min_episode, limit):
        replay = read_replay(row, archives)
        seat = int(row["seat_index"])
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            recorded = (steps[step_index + 1][seat] or {}).get("action")
            if (
                record.get("status") != "ACTIVE"
                or int(select.get("context", -1)) != 0
                or int(select.get("type", -1)) != 0
                or int(select.get("minCount") or 0) != 1
                or int(select.get("maxCount") or 0) != 1
                or len(options) < 2
                or not isinstance(recorded, list)
                or len(recorded) != 1
                or not isinstance(recorded[0], int)
                or not 0 <= recorded[0] < len(options)
            ):
                continue
            samples.append({
                "episode_id": int(row["episode_id"]),
                "observation": observation,
                "teacher": recorded[0],
            })
    for archive in archives.values():
        archive.close()
    return samples


def option_key(current, select, option, features_module) -> tuple:
    card = features_module.candidate_card(current, option, select) or {}
    target = features_module.candidate_target(current, option) or {}
    return (
        int(option.get("type", -1)),
        int(card.get("id", -1)),
        int(option.get("attackId", -1)),
        int(target.get("id", -1)),
        int(option.get("inPlayArea", -1)),
    )


def single(args) -> int:
    from agent_loader import load_dir_agent

    samples = collect(args.index, args.min_episode, args.episodes)
    _, _, main_module = load_dir_agent(args.agent.resolve())
    features_module = sys.modules["ml_features"]
    main_module.diag_reset()

    matched = 0
    turn_matched = 0
    per_episode_turns: dict[tuple[int, int], set[tuple]] = {}
    for sample in samples:
        current = sample["observation"].get("current") or {}
        select = sample["observation"].get("select") or {}
        options = list(select.get("option") or [])
        key = (sample["episode_id"], int(current.get("turn", -1)))
        per_episode_turns.setdefault(key, set()).add(
            option_key(current, select, options[sample["teacher"]],
                       features_module)
        )
    # The turn pick set has to be complete before any decision in that turn is
    # scored, so the agent runs in a second pass.
    for sample in samples:
        observation = sample["observation"]
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        action = main_module.agent(observation)
        if not isinstance(action, list) or len(action) != 1:
            continue
        played = int(action[0])
        if not 0 <= played < len(options):
            continue
        exact = played == sample["teacher"]
        matched += int(exact)
        key = (sample["episode_id"], int(current.get("turn", -1)))
        turn_matched += int(exact or option_key(
            current, select, options[played], features_module
        ) in per_episode_turns[key])

    snapshot = main_module.diag_snapshot()
    total = max(1, len(samples))
    result = {
        "agent": args.agent.name,
        "episodes": len({s["episode_id"] for s in samples}),
        "decisions": len(samples),
        "played_top1": matched / total,
        "played_turn_set": turn_matched / total,
        "diag": {
            key: value for key, value in snapshot["ml"].items()
            if isinstance(value, (int, float))
        },
        "fallback_reasons": {
            key: value for key, value in snapshot["ml"].items()
            if key.startswith(("fallback_", "candidate_blocked_", "memory_"))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        k: v for k, v in result.items() if k != "diag"
    }, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--agent", action="append", type=Path, required=True)
    parser.add_argument("--min-episode", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--_single", action="store_true")
    args = parser.parse_args()
    if args._single:
        args.agent = args.agent[0]
        return single(args)

    results = []
    for agent_dir in args.agent:
        partial = args.output.with_suffix(f".{agent_dir.name}.json")
        completed = subprocess.run(
            [
                sys.executable, __file__, str(args.index),
                "--agent", str(agent_dir),
                "--min-episode", str(args.min_episode),
                "--episodes", str(args.episodes),
                "--output", str(partial), "--_single",
            ],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-3000:])
        print(completed.stdout.strip(), flush=True)
        results.append(json.loads(partial.read_text(encoding="utf-8")))

    counts = Counter()
    for result in results:
        counts[result["agent"]] = result["played_top1"]
    args.output.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
