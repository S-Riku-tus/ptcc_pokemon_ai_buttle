"""Does overriding one action, then handing the turn back to v22, help or hurt?

The 320-game mirror arena scored the prize-authority layer at 153-167 (0.478)
even though it found and played 480 strictly-more-prize openings.  There are
two candidate explanations and they need different fixes:

  A. the extra prize is not real - the determinized deck order invented it;
  B. the extra prize is real, but only at the end of the *line*, and vfinal
     plays only the line's first action before handing the turn back to the
     greedy ranker, which then walks somewhere else.

This measures B directly, on the same paired tree the offline probe uses:

  greedy  - v22 from the root to the end of the turn
  best    - the enumeration's best complete line
  hybrid  - the best line's first action, then v22 for the rest of the turn

``hybrid`` is exactly what the shipped agent does.  If hybrid tracks greedy
instead of best, the authority is being spent on openings whose payoff the
agent then declines to collect.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics
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


def episode_paths() -> dict[str, pathlib.Path]:
    out: dict[str, pathlib.Path] = {}
    for run in sorted((ROOT / "data" / "runs" / "grimmsnarl").iterdir()):
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
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--branch", type=int, default=30)
    parser.add_argument("--beam", type=int, default=48)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--report", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent
                        / "commitment_probe.json")
    args = parser.parse_args()

    rows = list(csv.DictReader(VERSION_GAMES.open(encoding="utf-8-sig")))
    paths = episode_paths()
    pool = [(row["episode_id"], row) for row in rows if row["episode_id"] in paths]
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.games]
    print(f"sampled {len(pool)} games")

    module = load_dir_agent_module(AGENT_DIR)
    searcher = TurnSearch(
        DECK, max_nodes=args.nodes, max_seconds=args.seconds,
        branch_cap=args.branch, beam_width=args.beam,
    )

    turns = 0
    tallies: Counter = Counter()
    prize_totals = {"greedy": 0, "best": 0, "hybrid": 0}
    records = []
    started = time.monotonic()

    for episode, row in pool:
        replay = json.loads(paths[episode].read_text(encoding="utf-8"))
        seat_hint = int(row["seat"])
        seen: set[int] = set()
        for step in replay.get("steps") or []:
            if seat_hint >= len(step):
                continue
            observation = (step[seat_hint] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            if not observation.get("search_begin_input"):
                continue
            if int(select.get("context", -1)) != 0:
                continue
            turn = int(current.get("turn", -1))
            if turn < 3 or turn in seen:
                continue
            seen.add(turn)

            shuffle = random.Random(
                args.seed ^ turn ^ (hash(episode) & 0xFFFF)
            ).shuffle
            try:
                prepared = searcher.prepare(observation, shuffle)
            except (SearchUnavailable, Exception):  # noqa: BLE001
                continue
            root, seat, start_turn, base_prizes, base_damage = prepared

            module.diag_reset()
            try:
                greedy = searcher.walk(
                    root, seat, start_turn, lambda obs: module.agent(obs),
                    base_prizes, base_damage,
                )
            except Exception:  # noqa: BLE001
                try:
                    searcher.api.end()
                except Exception:  # noqa: BLE001
                    pass
                continue

            lines = searcher.search(observation, prepared=prepared, close=False)
            complete = [line for line in lines if line.get("complete")]
            if not complete:
                try:
                    searcher.api.end()
                except Exception:  # noqa: BLE001
                    pass
                continue
            best = max(
                complete,
                key=lambda line: (
                    1 if line["result"] == seat else 0,
                    line["prizes"], line["damage"],
                ),
            )
            hybrid = None
            if best["path"] and best["prizes"] > greedy["prizes"]:
                # Replay the best line's opening, then hand the turn to v22 -
                # exactly the shipped behaviour.
                module.diag_reset()
                try:
                    first = searcher.api.step(root["searchId"], best["path"][0])
                    hybrid = searcher.walk(
                        first, seat, start_turn,
                        lambda obs: module.agent(obs), base_prizes, base_damage,
                    )
                    hybrid["depth"] += 1
                except Exception:  # noqa: BLE001
                    hybrid = None
            try:
                searcher.api.end()
            except Exception:  # noqa: BLE001
                pass

            turns += 1
            if hybrid is None:
                continue
            tallies["override_opportunities"] += 1
            prize_totals["greedy"] += greedy["prizes"]
            prize_totals["best"] += best["prizes"]
            prize_totals["hybrid"] += hybrid["prizes"]
            if hybrid["prizes"] >= best["prizes"]:
                tallies["hybrid_collects_the_prize"] += 1
            elif hybrid["prizes"] > greedy["prizes"]:
                tallies["hybrid_collects_some"] += 1
            elif hybrid["prizes"] == greedy["prizes"]:
                tallies["hybrid_equals_greedy"] += 1
            else:
                tallies["hybrid_worse_than_greedy"] += 1
            records.append({
                "episode": episode, "turn": turn,
                "greedy_prizes": greedy["prizes"], "best_prizes": best["prizes"],
                "hybrid_prizes": hybrid["prizes"],
                "greedy_damage": greedy["damage"], "best_damage": best["damage"],
                "hybrid_damage": hybrid["damage"],
                "best_depth": best["depth"], "greedy_depth": greedy["depth"],
            })

    elapsed = time.monotonic() - started
    print(f"\nturns searched: {turns}  ({elapsed:.0f}s)")
    opportunities = tallies["override_opportunities"]
    print(f"turns where the search would have overridden: {opportunities}")
    if opportunities:
        for key in (
            "hybrid_collects_the_prize", "hybrid_collects_some",
            "hybrid_equals_greedy", "hybrid_worse_than_greedy",
        ):
            print(f"   {key:32} {tallies[key]:4}  "
                  f"({tallies[key] / opportunities:5.1%})")
        print(f"\nprizes over those {opportunities} turns: "
              f"greedy {prize_totals['greedy']}, "
              f"hybrid {prize_totals['hybrid']}, "
              f"best-if-committed {prize_totals['best']}")
        damages = [
            (r["hybrid_damage"] - r["greedy_damage"]) for r in records
        ]
        if damages:
            print(f"mean damage delta hybrid - greedy: "
                  f"{statistics.mean(damages):+.1f}")
        depths = [r["best_depth"] for r in records]
        print(f"best-line depth: mean {statistics.mean(depths):.1f}, "
              f"median {statistics.median(depths)}")

    args.report.write_text(json.dumps({
        "games": len(pool), "turns": turns,
        "tallies": dict(tallies), "prize_totals": prize_totals,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
