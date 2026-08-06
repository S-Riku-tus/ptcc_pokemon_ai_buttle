"""What else is on the menu when the Froslass evolve is, in our own games.

Scoring the whole select as a different pilot is only narrow if the select is
narrow. The class-mode escalation fires on every MAIN decision that offers a
Froslass evolve, so anything else offered at that moment is also being decided
by the escalation pilot - and a global pin to that pilot was measured to drop
the Grimmsnarl evolve from 62.3% of turns to 43.1%, which is a regression we
must not import through the side door.

This counts, over our own stored ladder games, how many decisions the class is,
what share of all decisions that is, and what the co-offered actions are.

Usage:
    python experiments/grimmsnarl_ml_v6/measure_escalation_scope.py \
        --run-dir data/runs/grimmsnarl/20260806_grimmsnarl_ml_v5_sub55275642 \
        --submission 55275642
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v5"))

import ml_features as mf  # noqa: E402

FROSLASS_ID = 104
GRIMMSNARL_EX_ID = 648
MAIN = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    co_offered: Counter[str] = Counter()
    rows = list(csv.DictReader(
        (args.run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ))
    for row in rows:
        episode = row["episode_id"]
        seat = 0 if row["agent_0_submission_id"] == args.submission else 1
        path = (
            args.run_dir / "episodes" / episode / "replay"
            / f"episode_{episode}.json"
        )
        if not path.exists():
            continue
        steps = (json.loads(path.read_text(encoding="utf-8")).get("steps") or [])
        counts["episodes"] += 1
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            if not options:
                continue
            counts["decisions"] += 1
            if int(select.get("context", -1)) != MAIN:
                continue
            if int(select.get("maxCount") or 0) != 1:
                continue
            counts["main_decisions"] += 1
            current = observation.get("current") or {}
            actions = [mf.action_type(current, o, select) for o in options]
            cards = [
                int((mf.candidate_card(current, o, select) or {}).get("id", -1))
                for o in options
            ]
            froslass = [
                slot for slot, action in enumerate(actions)
                if action == "evolve" and cards[slot] == FROSLASS_ID
            ]
            if not froslass:
                continue
            counts["escalated_decisions"] += 1
            counts["escalated_options"] += len(options)
            for slot, action in enumerate(actions):
                if slot in froslass:
                    continue
                if action == "evolve" and cards[slot] == GRIMMSNARL_EX_ID:
                    co_offered["grimmsnarl_evolve"] += 1
                elif action == "evolve":
                    co_offered[f"evolve_{cards[slot]}"] += 1
                else:
                    co_offered[action] += 1
            for name, present in (
                ("with_grimmsnarl_evolve", any(
                    actions[s] == "evolve" and cards[s] == GRIMMSNARL_EX_ID
                    for s in range(len(options))
                )),
                ("with_energy_attach", any(
                    actions[s] == "energy" and cards[s] == mf.DARK_ENERGY_ID
                    for s in range(len(options))
                )),
                ("with_attack", any(a == "attack" for a in actions)),
                ("with_boss", any(a == "boss" for a in actions)),
            ):
                if present:
                    counts[name] += 1

    report = {
        "run_dir": str(args.run_dir),
        "counts": dict(counts),
        "escalated_share_of_all_decisions": round(
            counts["escalated_decisions"] / counts["decisions"], 4
        ) if counts["decisions"] else None,
        "escalated_share_of_main": round(
            counts["escalated_decisions"] / counts["main_decisions"], 4
        ) if counts["main_decisions"] else None,
        "mean_options_when_escalated": round(
            counts["escalated_options"] / counts["escalated_decisions"], 2
        ) if counts["escalated_decisions"] else None,
        "co_offered_actions": dict(co_offered.most_common()),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
