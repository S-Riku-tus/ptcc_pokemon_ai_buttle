"""Every decision the v10 residual changes, and every one it declines to.

The behaviour probe reports rates; this reports the ledger. A residual whose
override count is not a measured number is the shell the Alakazam line lost
5.16 points of agreement to, so each firing is written out with the board that
triggered it, what v8 would have taken, what the panel replaced it with, and
how many pilots voted.

The two agents are walked side by side on the identical stored decisions, both
teacher-forced, so a difference between them is the residual and nothing else.
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


def load(agent_dir: Path):
    agent, _, module = load_dir_agent(agent_dir)
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        raise SystemExit(f"{agent_dir}: no ranker loaded")
    ranker.teacher_forced = True
    return agent, module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8",
    )
    parser.add_argument(
        "--candidate", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v10",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base_agent, base_module = load(args.base)
    cand_agent, cand_module = load(args.candidate)
    mf = sys.modules["ml_features"]

    episodes: list[tuple[int, int, Path]] = []
    for raw in csv.DictReader(
        (args.run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
            continue
        episode_id = int(raw["episode_id"])
        path = (
            args.run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if path.exists():
            episodes.append(
                (episode_id, 0 if a0 == args.submission else 1, path)
            )
    if args.limit:
        episodes = episodes[: args.limit]

    residual_totals: Counter[str] = Counter()
    differences: list[dict[str, Any]] = []
    decisions = 0

    for episode_id, seat, path in episodes:
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        won = bool(rewards[seat] > (other if other is not None else 0))
        for module in (base_module, cand_module):
            snapshot = (module.diag_snapshot() or {}).get("residual") or {}
            for key, value in snapshot.items():
                if isinstance(value, (int, float)):
                    residual_totals[key] += int(value)
            module.diag_reset()

        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            action = (steps[index + 1][seat] or {}).get("action")
            if not options or not isinstance(action, list) or len(action) != 1:
                continue
            if not isinstance(action[0], int):
                continue
            if not 0 <= action[0] < len(options):
                continue
            current = observation.get("current") or {}
            if len(current.get("players") or []) < 2:
                continue
            decisions += 1
            base_answer = base_agent(observation)
            cand_answer = cand_agent(observation)
            if base_answer != cand_answer:
                def describe(answer):
                    if not (isinstance(answer, list) and len(answer) == 1):
                        return None
                    slot = answer[0]
                    if not 0 <= slot < len(options):
                        return None
                    card = mf.resolve_option(
                        current, select, options[slot]
                    )[0] or {}
                    return {"slot": slot, "card": int(card.get("id", -1))}

                differences.append({
                    "episode_id": episode_id,
                    "won": won,
                    "turn": int(current.get("turn", -1)),
                    "context": int(select.get("context", -1)),
                    "options": len(options),
                    "v8": describe(base_answer),
                    "v10": describe(cand_answer),
                    "played": action[0],
                })
            base_module.observe_external(observation, action[0])
            cand_module.observe_external(observation, action[0])
        print(f"{episode_id} decisions={decisions} "
              f"differences={len(differences)}", flush=True)

    for module in (base_module, cand_module):
        snapshot = (module.diag_snapshot() or {}).get("residual") or {}
        for key, value in snapshot.items():
            if isinstance(value, (int, float)):
                residual_totals[key] += int(value)

    report = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "episodes": len(episodes),
        "decisions": decisions,
        "differences": len(differences),
        "difference_rate": (
            round(len(differences) / decisions, 6) if decisions else None
        ),
        "distinct_episodes_changed": len(
            {row["episode_id"] for row in differences}
        ),
        "changed_in_lost_games": len(
            {row["episode_id"] for row in differences if not row["won"]}
        ),
        "contexts_changed": dict(
            Counter(row["context"] for row in differences)
        ),
        "residual_counters": dict(residual_totals),
        "ledger": differences,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {k: v for k, v in report.items() if k != "ledger"},
        ensure_ascii=False, indent=2,
    ))
    for row in differences:
        print(f"  ep={row['episode_id']} turn={row['turn']} "
              f"ctx={row['context']} v8={row['v8']} v10={row['v10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
