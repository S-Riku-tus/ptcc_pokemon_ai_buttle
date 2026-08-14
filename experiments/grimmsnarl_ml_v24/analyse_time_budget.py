"""How much of the per-episode compute bank does the agent actually spend?

v22/v24 are a one-ply ranker plus rule fallback: `main.py` has no deadline, no
lookahead and no timing code at all.  If Kaggle grants a large per-episode
overage bank and we return in microseconds, that unused compute is the single
largest untouched resource in the project - and the mirror at >=950, which is
decided one prize apart, is exactly the kind of position a search can win.

`remainingOverageTime` is reported per seat on every step of the replay, so
the spend is directly observable without instrumenting the agent.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/runs/grimmsnarl"
GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/time_budget.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (
                    run_dir, int(row["detected_submission_agent_index"]))

    rows: list[dict[str, Any]] = []
    configuration: dict[str, Any] | None = None
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith("v24"):
            continue
        entry = index.get(raw["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (run_dir / "episodes" / raw["episode_id"] / "replay"
                / f"episode_{raw['episode_id']}.json")
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        if configuration is None:
            configuration = replay.get("configuration")
        steps = replay.get("steps") or []
        ours, theirs = [], []
        for step in steps:
            for actor, sink in ((seat, ours), (1 - seat, theirs)):
                if actor < len(step) and isinstance(step[actor], dict):
                    # The field lives on the observation, not the step record.
                    obs = step[actor].get("observation") or {}
                    value = obs.get("remainingOverageTime")
                    if isinstance(value, (int, float)):
                        sink.append(float(value))
        if not ours:
            continue
        rows.append({
            "episode_id": raw["episode_id"],
            "won": raw["won"] == "True",
            "steps": len(steps),
            "our_start": ours[0],
            "our_end": ours[-1],
            "our_spent": round(ours[0] - ours[-1], 3),
            "opp_start": theirs[0] if theirs else None,
            "opp_end": theirs[-1] if theirs else None,
            "opp_spent": round(theirs[0] - theirs[-1], 3) if theirs else None,
        })

    if not rows:
        print("no timing fields found in the replays")
        return 1

    ours = [r["our_spent"] for r in rows]
    opps = [r["opp_spent"] for r in rows if r["opp_spent"] is not None]
    bank = rows[0]["our_start"]
    print("configuration:")
    print(json.dumps(configuration, ensure_ascii=False, indent=2)[:1200])
    print()
    print(f"episodes measured: {len(rows)}")
    print(f"per-episode overage bank at step 0: {bank}")
    print(f"our spend:      mean {sum(ours) / len(ours):8.3f}s  "
          f"max {max(ours):8.3f}s  "
          f"({sum(ours) / len(ours) / bank:.2%} of bank)" if bank else "")
    if opps:
        print(f"opponent spend: mean {sum(opps) / len(opps):8.3f}s  "
              f"max {max(opps):8.3f}s  "
              f"({sum(opps) / len(opps) / bank:.2%} of bank)" if bank else "")
    print()
    print("top 10 episodes by our spend:")
    for r in sorted(rows, key=lambda r: -r["our_spent"])[:10]:
        print(f"  {r['episode_id']}  {'W' if r['won'] else 'L'}  "
              f"steps {r['steps']:>3}  ours {r['our_spent']:8.3f}s  "
              f"opp {r['opp_spent']:8.3f}s")

    OUT.write_text(json.dumps(
        {"configuration": configuration, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
