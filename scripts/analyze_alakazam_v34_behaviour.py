"""Compare what our ladder agents actually DO, run against run.

The offline Top-1 metric says v33 beats v32 by 0.80 points, but the ladder
rating moved 45. That gap means agreement rate is a weak proxy, so v34 needs a
behavioural read as well: which concrete plays changed between versions, and do
the plays that previous measurement tied to winning actually happen more often.

Two levers are already measured and documented for this deck:

  Dudunsparce Run Away Draw   A reusable +3 cards per turn that shuffles itself
                              back into the deck. Earlier work measured our
                              agents at 1.69 uses/game against 2.86 for the
                              field on the identical 60-card list, and games
                              with zero uses had a 0% win rate.
  Powerful Hand               Alakazam's attack, 20 damage counters per card in
                              hand. Landing 4 or more decides the Grimmsnarl
                              matchup.

Usage:
  python scripts/analyze_alakazam_v34_behaviour.py --report out.json \\
      v31=data/runs/.../20260730_v31_sub55076863:55076863 \\
      v33=data/runs/.../20260731_v33_sub55129390:55129390
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CARDS: dict[int, dict[str, Any]] = {
    c["cardId"]: c
    for c in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}

ALAKAZAM, KADABRA, ABRA = 743, 742, 741
DUDUNSPARCE, DUNSPARCE = 66, 305

TURN_START = 2
EVOLVE = 12
ATTACK = 15
RESULT = 23


def seat_map(run: Path, submission_id: int) -> dict[str, int]:
    rows = list(csv.DictReader(
        (run / "episodes.csv").read_text(encoding="utf-8-sig").splitlines()
    ))
    mapping: dict[str, int] = {}
    for row in rows:
        for seat in (0, 1):
            if str(row.get(f"agent_{seat}_submission_id")) == str(submission_id):
                mapping[str(row["episode_id"])] = seat
    return mapping


def _stream(run: Path, episode_id: str, agent: int) -> list[dict[str, Any]]:
    path = (
        run / "episodes" / episode_id / f"agent_{agent}"
        / f"agent_{agent}_observation_logs.json"
    )
    if not path.exists():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    out: list[dict[str, Any]] = []
    previous: str | None = None
    for entry in entries:
        blob = json.dumps(entry["logs"], sort_keys=True)
        if blob == previous:
            continue
        previous = blob
        out.extend(entry["logs"])
    return out


def merged_logs(run: Path, episode_id: str, seat: int) -> list[dict[str, Any]]:
    ours = _stream(run, episode_id, seat)
    theirs = _stream(run, episode_id, 1 - seat)
    turns_ours = sum(1 for log in ours if log.get("type") == TURN_START)
    turns_theirs = sum(1 for log in theirs if log.get("type") == TURN_START)
    return theirs if turns_theirs > turns_ours else ours


def archetype(deck: list[int] | None) -> str:
    if not deck:
        return "unknown"
    pokes = Counter(
        cid for cid in deck
        if CARDS.get(cid) and CARDS[cid]["cardType"] == 0
    )
    if not pokes:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple:
        cid, count = item
        card = CARDS[cid]
        return (
            card["stage2"], card["megaEx"] or card["ex"],
            card["stage1"], count, card["hp"],
        )

    return CARDS[max(pokes.items(), key=key)[0]]["name"]


def analyse_episode(run: Path, episode_id: str, seat: int) -> dict[str, Any]:
    replay = json.loads(
        (run / "episodes" / episode_id / "replay"
         / f"episode_{episode_id}.json").read_text(encoding="utf-8")
    )
    steps = replay["steps"]
    decks: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for s in (0, 1):
            action = steps[1][s].get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[s] = action

    rewards = [steps[-1][s].get("reward") for s in (0, 1)]
    ours, theirs = rewards[seat], rewards[1 - seat]
    if ours is None or theirs is None or ours == theirs:
        result = "draw"
    else:
        result = "win" if ours > theirs else "loss"

    logs = merged_logs(run, episode_id, seat)
    our_turns = 0
    evolves: Counter[int] = Counter()
    attacks: Counter[int] = Counter()
    for log in logs:
        kind = log.get("type")
        player = log.get("playerIndex")
        if kind == TURN_START:
            if player == seat:
                our_turns += 1
        elif kind == EVOLVE and player == seat:
            evolves[log.get("cardId")] += 1
        elif kind == ATTACK and player == seat:
            attacks[log.get("cardId")] += 1

    return {
        "episode_id": episode_id,
        "seat": seat,
        "result": result,
        "our_turns": our_turns,
        "dudunsparce": int(evolves[DUDUNSPARCE]),
        "kadabra": int(evolves[KADABRA]),
        "alakazam": int(evolves[ALAKAZAM]),
        "powerful_hand": int(attacks[ALAKAZAM]),
        "total_attacks": int(sum(attacks.values())),
        "opponent_archetype": archetype(decks[1 - seat]),
    }


def summarise(games: list[dict[str, Any]]) -> dict[str, Any]:
    decided = [g for g in games if g["result"] in ("win", "loss")]
    wins = sum(1 for g in decided if g["result"] == "win")
    turns = sum(g["our_turns"] for g in games) or 1

    def bucket_rate(field: str, edges: list[int]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i, lo in enumerate(edges):
            hi = edges[i + 1] if i + 1 < len(edges) else None
            sel = [
                g for g in decided
                if g[field] >= lo and (hi is None or g[field] < hi)
            ]
            if not sel:
                continue
            w = sum(1 for g in sel if g["result"] == "win")
            label = f"{lo}" if hi == lo + 1 else (
                f"{lo}+" if hi is None else f"{lo}-{hi - 1}"
            )
            out[label] = f"{w}/{len(sel)} ({w / len(sel):.0%})"
        return out

    return {
        "games": len(games),
        "decided": len(decided),
        "record": f"{wins}-{len(decided) - wins}",
        "win_rate": wins / len(decided) if decided else 0.0,
        "our_turns_per_game": sum(g["our_turns"] for g in games) / len(games),
        "dudunsparce_per_game": sum(g["dudunsparce"] for g in games) / len(games),
        "dudunsparce_per_turn": sum(g["dudunsparce"] for g in games) / turns,
        "zero_dudunsparce_games": sum(
            1 for g in games if g["dudunsparce"] == 0
        ),
        "zero_dudunsparce_share": sum(
            1 for g in games if g["dudunsparce"] == 0
        ) / len(games),
        "powerful_hand_per_game": sum(
            g["powerful_hand"] for g in games
        ) / len(games),
        "games_with_4plus_powerful_hand": sum(
            1 for g in games if g["powerful_hand"] >= 4
        ),
        "share_4plus_powerful_hand": sum(
            1 for g in games if g["powerful_hand"] >= 4
        ) / len(games),
        "alakazam_per_game": sum(g["alakazam"] for g in games) / len(games),
        "win_rate_by_dudunsparce": bucket_rate("dudunsparce", [0, 1, 2, 3]),
        "win_rate_by_powerful_hand": bucket_rate("powerful_hand", [0, 4]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+",
        help="label=run_dir:submission_id",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {"runs": {}}
    for spec in args.runs:
        label, rest = spec.split("=", 1)
        run_dir, submission = rest.rsplit(":", 1)
        run = Path(run_dir)
        seats = seat_map(run, int(submission))
        games = []
        for episode_id, seat in sorted(seats.items()):
            if not (run / "episodes" / episode_id / "replay").exists():
                continue
            games.append(analyse_episode(run, episode_id, seat))
        report["runs"][label] = {
            "run": str(run),
            "submission_id": int(submission),
            "summary": summarise(games),
            "games": games,
        }
        summary = report["runs"][label]["summary"]
        print(json.dumps({label: {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in summary.items()
            if not k.startswith("win_rate_by")
        }}, ensure_ascii=False), flush=True)
        for key in ("win_rate_by_dudunsparce", "win_rate_by_powerful_hand"):
            print(f"  {label} {key}: {summary[key]}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
