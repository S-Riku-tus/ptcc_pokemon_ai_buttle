"""Counterfactual probe: what would a candidate take on the *stored* boards?

The two v5 changes both live in a multi-pick select the ranker never sees, so
neither shows up in a Top-1 agreement number and neither is measurable from a
ladder run whose rating noise is larger than the effect (an identical agent has
scored 842.8 and 804 on this ladder). What is measurable is the count itself:
replay a stored run decision by decision, ask the candidate at every Punk Up
deck search and Buddy-Buddy Poffin search, advance the game with the action
that was actually taken, and compare the candidate's counts against both the
run's own and the same-deck top-50 corpus.

Teacher forcing keeps the boards on-distribution and keeps the runtime's
intra-turn history describing the turn the replay described, so a candidate and
its parent are compared on identical states.

    python scripts/probe_grimmsnarl_v5_search_budget.py \
        --agent-dir agents/grimmsnarl/grimmsnarl_ml_v5 \
        --run data/runs/grimmsnarl/20260805_grimmsnarl_ml_v4_sub55253296 \
        --submission 55253296 \
        --report experiments/grimmsnarl_ml_v5/probe_v5_on_v4_run.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402

PUNK_CONTEXT = 22
POFFIN_CONTEXT = 5
GRIMMSNARL_EX_ID = 648
POFFIN_ID = 1086


def _effect_id(select: dict[str, Any]) -> int:
    effect = select.get("effect")
    if isinstance(effect, dict):
        try:
            return int(effect.get("id", -1))
        except (TypeError, ValueError):
            return -1
    return -1


def _played(step_next: list[Any], seat: int, option_count: int) -> list[int]:
    action = (step_next[seat] or {}).get("action")
    if not isinstance(action, list):
        return []
    return [
        value for value in action
        if isinstance(value, int) and 0 <= value < option_count
    ]


def _candidate(agent, observation: dict[str, Any], count: int) -> list[int]:
    answer = agent(observation)
    if not isinstance(answer, list):
        return []
    return [v for v in answer if isinstance(v, int) and 0 <= v < count]


def probe(agent, observe_external, replay_path: Path, seat: int,
          counts: Counter, examples: list[dict[str, Any]]) -> None:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    steps = payload.get("steps") or []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        played = _played(steps[index + 1], seat, len(options))
        if not select:
            continue
        context = int(select.get("context", -1))
        effect = _effect_id(select)
        group = None
        if context == PUNK_CONTEXT and effect == GRIMMSNARL_EX_ID:
            group = "punk"
        elif context == POFFIN_CONTEXT and effect == POFFIN_ID:
            group = "poffin"

        if group is not None:
            current = observation.get("current") or {}
            maximum = int(select.get("maxCount", 0) or 0)
            mine = _candidate(agent, observation, len(options))
            counts[f"{group}_events"] += 1
            counts[f"{group}_played_sum"] += len(played)
            counts[f"{group}_agent_sum"] += len(mine)
            counts[f"{group}_max_sum"] += maximum
            counts[f"{group}_played_max"] += int(len(played) == maximum)
            counts[f"{group}_agent_max"] += int(len(mine) == maximum)
            counts[f"{group}_agent_count_{min(len(mine), 5)}"] += 1
            counts[f"{group}_played_count_{min(len(played), 5)}"] += 1
            counts[f"{group}_agent_over_legal"] += int(len(mine) > maximum)
            counts[f"{group}_agent_under_min"] += int(
                len(mine) < int(select.get("minCount", 0) or 0)
            )
            if len(mine) != len(played) and len(examples) < 12:
                examples.append({
                    "group": group,
                    "replay": replay_path.name,
                    "turn": int(current.get("turn", -1)),
                    "max": maximum,
                    "played": len(played),
                    "agent": len(mine),
                })
        else:
            # Every other select still has to be asked, or the ranker's
            # intra-turn columns and the planner's heal budget stop describing
            # the turn the replay described.
            _candidate(agent, observation, len(options))
        if played:
            observe_external(observation, played[0])


def run_jobs(run_dir: Path, submission: str):
    rows = list(csv.DictReader(
        open(run_dir / "episodes.csv", encoding="utf-8-sig")))
    for row in rows:
        seat = 0 if row["agent_0_submission_id"] == submission else 1
        path = (run_dir / "episodes" / row["episode_id"] / "replay"
                / f"episode_{row['episode_id']}.json")
        if path.exists():
            yield path, seat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    agent, _diag, module = load_dir_agent(args.agent_dir)
    observe_external = getattr(module, "observe_external", lambda *_: None)
    reset = getattr(module, "diag_reset", lambda: None)

    counts: Counter = Counter()
    examples: list[dict[str, Any]] = []
    jobs = list(run_jobs(args.run, args.submission))
    if args.limit:
        jobs = jobs[:args.limit]
    for path, seat in jobs:
        reset()
        probe(agent, observe_external, path, seat, counts, examples)

    report: dict[str, Any] = {
        "agent_dir": str(args.agent_dir),
        "run": str(args.run),
        "replays": len(jobs),
        "examples": examples,
    }
    for group in ("punk", "poffin"):
        n = counts[f"{group}_events"]
        if not n:
            continue
        report[group] = {
            "events": n,
            "mean_offered_max": counts[f"{group}_max_sum"] / n,
            "mean_played": counts[f"{group}_played_sum"] / n,
            "mean_agent": counts[f"{group}_agent_sum"] / n,
            "played_took_max_pct": 100 * counts[f"{group}_played_max"] / n,
            "agent_took_max_pct": 100 * counts[f"{group}_agent_max"] / n,
            "agent_counts": {
                k: counts[f"{group}_agent_count_{k}"] for k in range(6)
            },
            "played_counts": {
                k: counts[f"{group}_played_count_{k}"] for k in range(6)
            },
            "illegal_over_max": counts[f"{group}_agent_over_legal"],
            "illegal_under_min": counts[f"{group}_agent_under_min"],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"},
                     indent=2))


if __name__ == "__main__":
    main()
