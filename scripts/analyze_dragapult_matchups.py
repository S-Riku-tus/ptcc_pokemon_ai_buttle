"""Per-archetype win rate for the exact Dragapult list: teachers vs our run.

A matchup that the teachers also lose is a deck property and not worth a
policy cycle; a matchup only we lose is ours to fix.  Separating the two needs
the same archetype labelling applied to both sets of replays.

Usage:
  python scripts/analyze_dragapult_matchups.py \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --run data/submissions/submission_55545828_dragapult_v1 \
      --report experiments/dragapult_ml_v2/matchups.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}


def archetype(deck: list[int]) -> str:
    pokemon = Counter(
        card_id for card_id in deck if CARDS.get(card_id, {}).get("cardType") == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple[int, int, int, int, int]:
        card_id, count = item
        card = CARDS[card_id]
        return (
            int(bool(card.get("stage2"))),
            int(bool(card.get("megaEx") or card.get("ex"))),
            int(bool(card.get("stage1"))),
            count,
            int(card.get("hp") or 0),
        )

    best = max(pokemon.items(), key=key)[0]
    return str(CARDS.get(best, {}).get("name") or best)


def game(path: Path, seat: int) -> dict[str, Any] | None:
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    if len(steps) < 2:
        return None
    decks: list[list[int]] = [[], []]
    for player in (0, 1):
        action = steps[1][player].get("action")
        if isinstance(action, list) and len(action) == 60:
            decks[player] = [int(value) for value in action]
    rewards = replay.get("rewards") or [0, 0]
    if rewards[seat] == rewards[1 - seat]:
        result = "draw"
    else:
        result = "win" if rewards[seat] > rewards[1 - seat] else "loss"
    return {
        "episode_id": int(
            replay.get("info", {}).get("EpisodeId") or path.stem.split("_")[-1]
        ),
        "result": result,
        "opponent": archetype(decks[1 - seat]),
        "steps": len(steps),
    }


def summarise(games: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in games:
        by_opponent[entry["opponent"]][entry["result"]] += 1
        by_opponent[entry["opponent"]]["games"] += 1
    rows = []
    for opponent, counter in sorted(
        by_opponent.items(), key=lambda item: -item[1]["games"]
    ):
        total = counter["games"]
        rows.append({
            "opponent": opponent,
            "games": total,
            "wins": counter["win"],
            "win_rate": round(counter["win"] / total, 4) if total else 0.0,
        })
    wins = sum(entry["result"] == "win" for entry in games)
    return {
        "label": label,
        "games": len(games),
        "win_rate": round(wins / len(games), 4) if games else 0.0,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--exclude-episode", type=int, nargs="*", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    if args.teacher_index:
        seen: set[tuple[str, int]] = set()
        games = []
        for row in csv.DictReader(
            args.teacher_index.read_text(encoding="utf-8-sig").splitlines()
        ):
            key = (str(row["episode_id"]), int(row["seat_index"]))
            if key in seen:
                continue
            seen.add(key)
            path = Path(row["replay_path"])
            if not path.is_absolute():
                path = args.teacher_index.parent.parent / path
            if not path.exists():
                continue
            entry = game(path, int(row["seat_index"]))
            if entry:
                entry["team_id"] = row.get("team_id")
                games.append(entry)
        report["teachers"] = summarise(games, "teachers")
    if args.run:
        games = []
        for row in csv.DictReader(
            (args.run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
        ):
            episode_id = int(row["episode_id"])
            if episode_id in args.exclude_episode:
                continue
            path = (
                args.run / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            entry = game(path, int(row["detected_submission_agent_index"]))
            if entry:
                games.append(entry)
        report["run"] = summarise(games, "live")

    teacher_rows = {
        row["opponent"]: row for row in report.get("teachers", {}).get("rows", [])
    }
    run_rows = {row["opponent"]: row for row in report.get("run", {}).get("rows", [])}
    print(f"{'opponent':28} {'teach n':>8} {'teach wr':>9} "
          f"{'live n':>7} {'live wr':>8}")
    for opponent, row in sorted(teacher_rows.items(), key=lambda i: -i[1]["games"]):
        live = run_rows.get(opponent)
        live_wr = live["win_rate"] if live else float("nan")
        print(f"{opponent:28} {row['games']:>8} {row['win_rate']:>9.3f} "
              f"{(live or {}).get('games', 0):>7} {live_wr:>8.3f}")
    for opponent, row in run_rows.items():
        if opponent not in teacher_rows:
            print(f"{opponent:28} {'-':>8} {'-':>9} "
                  f"{row['games']:>7} {row['win_rate']:>8.3f}")
    for key in ("teachers", "run"):
        if key in report:
            print(f"{key}: {report[key]['games']} games, "
                  f"win rate {report[key]['win_rate']}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
