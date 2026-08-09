"""How much of our own turn does a once-per-turn search actually cover?

Reads the v11.1 ladder run's replays from our seat and counts, per own turn,
how many MAIN-context selects carry a real choice (>1 option). The v11 search
fires on at most the first such select per turn, so everything after it is
pure v9 argmax.

Also reports the real compute headroom from ``remainingOverageTime`` and the
per-turn action-class profile of the first vs later MAIN decisions.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

RUN = ROOT / "data" / "runs" / "grimmsnarl" / "20260809_grimmsnarl_ml_v11_sub55353978"
SUBMISSION = "55353978"

CARDS = {
    int(c["cardId"]): c
    for c in json.loads((ROOT / "vendor" / "cg" / "cards.json").read_text("utf-8"))
}


def main() -> int:
    rows = list(csv.DictReader((RUN / "episodes.csv").open(encoding="utf-8-sig")))
    per_turn_counts = Counter()          # number of searchable MAIN selects in a turn
    turns_total = 0
    searchable_total = 0
    covered_total = 0                    # what once-per-turn actually sees
    first_slot_opts = Counter()
    later_slot_opts = Counter()
    overage_left = []
    games = 0
    main_selects_per_game = []
    turn_of_first = Counter()

    for raw in rows:
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or raw["state"] != "COMPLETED":
            continue
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if a0 == a1:
            continue
        seat = 0 if a0 == SUBMISSION else 1
        path = RUN / "episodes" / raw["episode_id"] / "replay" / f"episode_{raw['episode_id']}.json"
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        games += 1
        by_turn: dict[int, list[dict]] = defaultdict(list)
        last_overage = None
        for step in replay.get("steps") or []:
            if seat >= len(step) or not isinstance(step[seat], dict):
                continue
            entry = step[seat]
            if entry.get("status") != "ACTIVE":
                continue
            obs = entry.get("observation") or {}
            sel = obs.get("select") or {}
            cur = obs.get("current") or {}
            ov = obs.get("remainingOverageTime")
            if isinstance(ov, (int, float)):
                last_overage = float(ov)
            if int(sel.get("context", -1)) != 0:
                continue
            n = len(sel.get("option") or [])
            turn = int(cur.get("turn", -1))
            by_turn[turn].append({"n": n, "tac": int(cur.get("turnActionCount", 0) or 0)})
        if last_overage is not None:
            overage_left.append(last_overage)
        main_selects_per_game.append(sum(len(v) for v in by_turn.values()))
        for turn, decisions in by_turn.items():
            searchable = [d for d in decisions if d["n"] > 1]
            turns_total += 1
            per_turn_counts[len(searchable)] += 1
            searchable_total += len(searchable)
            covered_total += 1 if searchable else 0
            for i, d in enumerate(searchable):
                (first_slot_opts if i == 0 else later_slot_opts)[min(d["n"], 10)] += 1
            if searchable:
                turn_of_first[turn] += 1

    print(f"games={games} own turns={turns_total}")
    print(f"MAIN selects per game (mean)={sum(main_selects_per_game)/max(1,games):.1f}")
    print(f"searchable MAIN selects (>1 option) total={searchable_total}"
          f"  per own turn={searchable_total/max(1,turns_total):.2f}")
    print(f"covered by once-per-turn search={covered_total}"
          f"  ({covered_total/max(1,searchable_total):.1%} of searchable decisions)")
    print("\ndistribution: searchable MAIN selects in one own turn")
    for k in sorted(per_turn_counts):
        print(f"  {k:2d} decisions : {per_turn_counts[k]:5d} turns"
              f"  ({per_turn_counts[k]/turns_total:6.1%})")
    print("\noption-count of the FIRST searchable select vs the LATER ones")
    print("  n_options  first  later")
    for k in sorted(set(first_slot_opts) | set(later_slot_opts)):
        print(f"  {k:9d}  {first_slot_opts[k]:5d}  {later_slot_opts[k]:5d}")
    if overage_left:
        overage_left.sort()
        print(f"\nremainingOverageTime at game end: min={overage_left[0]:.1f}"
              f" p10={overage_left[len(overage_left)//10]:.1f}"
              f" median={overage_left[len(overage_left)//2]:.1f}"
              f" max={overage_left[-1]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
