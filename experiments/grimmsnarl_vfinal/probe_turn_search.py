"""Is there room inside our own turn that the greedy ranker does not take?

Paired design.  For each of our turns in a sample of stored ladder games:

1. determinize our hidden zones once and open one search tree;
2. walk it once following the v22 agent, action by action, to the end of the
   turn - this is exactly what v22 does, on this deck order;
3. enumerate the same tree and keep the best complete line.

Both walks descend from the same root, so the difference between them is not a
shuffle difference.  Anything the enumeration finds is a line v22 could have
played on the board it actually had.

Usage:
  python experiments/grimmsnarl_vfinal/probe_turn_search.py --games 40
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts", ROOT / "experiments" / "grimmsnarl_vfinal"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_loader import load_dir_agent_module  # noqa: E402
from turnsearch import SearchUnavailable, TurnSearch  # noqa: E402

AGENT_DIR = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22"
DECK = [
    int(line) for line in (AGENT_DIR / "deck.csv").read_text(encoding="utf-8").split()
    if line.strip()
]
VERSION_GAMES = (
    ROOT / "experiments" / "grimmsnarl_endgame_20260816" / "version_games.csv"
)
RUN_DIRS = {
    row["episode_id"]: row for row in
    csv.DictReader(VERSION_GAMES.open(encoding="utf-8-sig"))
}


def episode_paths() -> dict[str, pathlib.Path]:
    out: dict[str, pathlib.Path] = {}
    root = ROOT / "data" / "runs" / "grimmsnarl"
    for run in sorted(root.iterdir()):
        episodes = run / "episodes"
        if not episodes.is_dir():
            continue
        for episode in episodes.iterdir():
            replay = episode / "replay" / f"episode_{episode.name}.json"
            if replay.exists():
                out.setdefault(episode.name, replay)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--branch", type=int, default=6)
    parser.add_argument("--beam", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--min-turn", type=int, default=3)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--losses-only", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--report", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent
                        / "turn_search_probe.json")
    args = parser.parse_args()

    paths = episode_paths()
    pool = [
        (episode, row) for episode, row in RUN_DIRS.items()
        if episode in paths
        and (not args.family or row["opponent_family"] in args.family)
        and (not args.losses_only or row["won"] == "0")
    ]
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.games]
    print(f"sampled {len(pool)} games")

    module = load_dir_agent_module(AGENT_DIR)
    searcher = TurnSearch(
        DECK, max_nodes=args.nodes, max_seconds=args.seconds,
        branch_cap=args.branch, beam_width=args.beam,
    )

    turns = 0
    unavailable = Counter()
    better_value = 0
    better_prizes = 0
    better_damage = 0
    lethal_found = 0
    extra_prizes = 0
    truncations = 0
    per_family: dict[str, Counter] = {}
    records = []
    started = time.monotonic()

    for episode, row in pool:
        replay = json.loads(paths[episode].read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        seat = int(row["seat"])
        family = row["opponent_family"]
        stats = per_family.setdefault(family, Counter())
        seen_turns: set[int] = set()
        for step in steps:
            if seat >= len(step):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            if not observation.get("search_begin_input"):
                continue
            if int(select.get("context", -1)) != 0:
                continue
            turn = int(current.get("turn", -1))
            if turn < args.min_turn or turn in seen_turns:
                continue
            seen_turns.add(turn)

            shuffle = random.Random(args.seed ^ turn ^ hash(episode) & 0xFFFF).shuffle
            try:
                prepared = searcher.prepare(observation, shuffle)
            except SearchUnavailable as error:
                unavailable[str(error).split(":")[0]] += 1
                continue
            except Exception as error:  # noqa: BLE001
                unavailable[type(error).__name__] += 1
                continue

            root, our_seat, start_turn, base_prizes, base_damage = prepared
            module.diag_reset()
            try:
                greedy = searcher.walk(
                    root, our_seat, start_turn,
                    lambda obs: module.agent(obs),
                    base_prizes, base_damage,
                )
            except Exception as error:  # noqa: BLE001
                unavailable[f"walk:{type(error).__name__}"] += 1
                try:
                    searcher.api.end()
                except Exception:  # noqa: BLE001
                    pass
                continue

            lines = searcher.search(observation, prepared=prepared, close=True)
            complete = [line for line in lines if line.get("complete")]
            if not complete:
                continue
            best = max(complete, key=lambda line: line["value"])
            turns += 1
            stats["turns"] += 1
            truncations += int(searcher.stats.get("truncated", False))
            if best["value"] > greedy["value"] + 1e-6:
                better_value += 1
                stats["better_value"] += 1
            if best["prizes"] > greedy["prizes"]:
                better_prizes += 1
                stats["better_prizes"] += 1
                extra_prizes += best["prizes"] - greedy["prizes"]
            elif best["prizes"] == greedy["prizes"] and best["damage"] > greedy["damage"]:
                better_damage += 1
                stats["better_damage"] += 1
            if best["result"] == our_seat and greedy["result"] != our_seat:
                lethal_found += 1
                stats["lethal"] += 1
            records.append({
                "episode": episode, "family": family, "turn": turn,
                "won": row["won"],
                "greedy_value": greedy["value"], "best_value": best["value"],
                "greedy_prizes": greedy["prizes"], "best_prizes": best["prizes"],
                "greedy_damage": greedy["damage"], "best_damage": best["damage"],
                "greedy_result": greedy["result"], "best_result": best["result"],
                "nodes": searcher.stats.get("nodes"),
                "lines": len(lines),
                "truncated": bool(searcher.stats.get("truncated")),
            })

    elapsed = time.monotonic() - started
    print(f"\nour turns searched: {turns}   ({elapsed:.0f}s, "
          f"{elapsed / max(turns, 1):.2f}s per turn)")
    print(f"positions the search could not open: {sum(unavailable.values())}")
    for reason, count in unavailable.most_common(8):
        print(f"    {reason}: {count}")
    if turns:
        print(f"\nturns where a better line exists (leaf value): "
              f"{better_value} ({better_value / turns:.1%})")
        print(f"  ... strictly more prizes this turn:  {better_prizes} "
              f"({better_prizes / turns:.1%}), {extra_prizes} extra prizes total")
        print(f"  ... same prizes, more damage:        {better_damage} "
              f"({better_damage / turns:.1%})")
        print(f"  ... a win the greedy line missed:    {lethal_found}")
        print(f"  node-budget truncations: {truncations} ({truncations / turns:.1%})")
        print("\nby opponent family:")
        print(f"{'family':32} {'turns':>6} {'+value':>7} {'+prize':>7} {'+dmg':>6} {'win':>4}")
        for family, stats in sorted(
            per_family.items(), key=lambda kv: -kv[1]["turns"]
        ):
            print(f"{family[:32]:32} {stats['turns']:>6} "
                  f"{stats['better_value']:>7} {stats['better_prizes']:>7} "
                  f"{stats['better_damage']:>6} {stats['lethal']:>4}")

    args.report.write_text(json.dumps({
        "games": len(pool), "turns": turns, "elapsed_seconds": round(elapsed, 1),
        "node_budget": args.nodes, "branch_cap": args.branch,
        "better_value": better_value, "better_prizes": better_prizes,
        "better_damage": better_damage, "extra_prizes": extra_prizes,
        "lethal_found": lethal_found, "truncations": truncations,
        "unavailable": dict(unavailable),
        "per_family": {k: dict(v) for k, v in per_family.items()},
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
