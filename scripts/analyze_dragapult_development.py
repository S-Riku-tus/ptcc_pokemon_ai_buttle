"""Compare Dragapult board development turn by turn: our run vs the teachers.

A whole-game agreement number cannot say *when* a game was lost.  This walks
own-turn ordinals - the replay's ``current.turn`` is shared between seats, so
"turn 4" is a different own turn for each seat - and reports the state of the
evolution line and its typed Energy at the end of each of our own turns.

Usage:
  python scripts/analyze_dragapult_development.py \
      --run data/submissions/submission_55545828_dragapult_v1 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --split-report experiments/dragapult_ml_v1/training_report.json \
      --report experiments/dragapult_ml_v1/development_gap.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FIRE, PSYCHIC, DARK = 2, 5, 7
DREEPY, DRAKLOAK, DRAGAPULT, MUNKIDORI, BUDEW = 119, 120, 121, 112, 235
PHANTOM_DIVE = 154
OPT_ATTACK, OPT_END = 13, 14


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def line_state(player: dict[str, Any]) -> dict[str, int]:
    bodies = [
        card for card in
        (list(player.get("active") or []) + list(player.get("bench") or []))
        if isinstance(card, dict)
    ]
    counts = Counter(int(card.get("id", -1)) for card in bodies)
    best_stage = -1
    ready = 0
    typed = 0
    for card in bodies:
        card_id = int(card.get("id", -1))
        stage = {DREEPY: 0, DRAKLOAK: 1, DRAGAPULT: 2}.get(card_id, -1)
        if stage < 0:
            continue
        best_stage = max(best_stage, stage)
        energies = [int(value) for value in card.get("energies") or []]
        typed += int(FIRE in energies) + int(PSYCHIC in energies)
        if card_id == DRAGAPULT and FIRE in energies and PSYCHIC in energies:
            ready += 1
    return {
        "bodies": len(bodies),
        "dreepy": counts[DREEPY],
        "drakloak": counts[DRAKLOAK],
        "dragapult": counts[DRAGAPULT],
        "munkidori": counts[MUNKIDORI],
        "line": counts[DREEPY] + counts[DRAKLOAK] + counts[DRAGAPULT],
        "best_stage": best_stage,
        "route_colors": typed,
        "phantom_ready": ready,
    }


def analyse(path: Path, seat: int) -> dict[str, Any] | None:
    replay = load(path)
    steps = replay.get("steps") or []
    rewards = replay.get("rewards") or [0, 0]
    own_turns: dict[int, dict[str, Any]] = {}
    turn_order: list[int] = []
    attacked: set[int] = set()
    phantom: set[int] = set()
    for step_index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, dict) or not isinstance(select, dict):
            continue
        turn = int(current.get("turn") or 0)
        if int(current.get("yourIndex", seat)) != seat:
            continue
        if turn not in own_turns:
            if int(select.get("context", -1)) != 0:
                # Sub-selections inside another player's effect share the turn
                # counter; only a MAIN offer proves it is our own turn.
                continue
            own_turns[turn] = {}
            turn_order.append(turn)
        players = current.get("players") or [{}, {}]
        own_turns[turn] = line_state(players[seat] if seat in (0, 1) else {})
        own_turns[turn]["prizes_left"] = len(
            (players[seat] if seat in (0, 1) else {}).get("prize") or []
        )
        action = (
            steps[step_index + 1][seat].get("action")
            if step_index + 1 < len(steps) else None
        )
        if isinstance(action, list) and len(action) != 60:
            options = select.get("option") or []
            for index in action:
                if not isinstance(index, int) or not 0 <= index < len(options):
                    continue
                option = options[index]
                if int(option.get("type", -1)) == OPT_ATTACK:
                    attacked.add(turn)
                    if int(option.get("attackId", -1)) == PHANTOM_DIVE:
                        phantom.add(turn)
    if not turn_order:
        return None
    turn_order.sort()
    ordinals: list[dict[str, Any]] = []
    for ordinal, turn in enumerate(turn_order, start=1):
        state = dict(own_turns[turn])
        state["own_turn"] = ordinal
        state["attacked"] = int(turn in attacked)
        state["phantom"] = int(turn in phantom)
        ordinals.append(state)
    first_phantom = next(
        (row["own_turn"] for row in ordinals if row["phantom"]), None
    )
    first_pult = next(
        (row["own_turn"] for row in ordinals if row["dragapult"] > 0), None
    )
    first_ready = next(
        (row["own_turn"] for row in ordinals if row["phantom_ready"] > 0), None
    )
    return {
        "episode_id": int(
            replay.get("info", {}).get("EpisodeId") or path.stem.split("_")[-1]
        ),
        "seat": seat,
        "result": (
            "win" if rewards[seat] > rewards[1 - seat]
            else "loss" if rewards[seat] < rewards[1 - seat] else "draw"
        ),
        "own_turns": len(ordinals),
        "first_dragapult_own_turn": first_pult,
        "first_phantom_ready_own_turn": first_ready,
        "first_phantom_own_turn": first_phantom,
        "rows": ordinals,
    }


def summarise(games: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_ordinal: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        for row in game["rows"]:
            by_ordinal[row["own_turn"]].append(row)

    def mean(values: list[float]) -> float:
        return round(statistics.mean(values), 3) if values else 0.0

    curve = []
    for ordinal in sorted(by_ordinal)[:10]:
        rows = by_ordinal[ordinal]
        curve.append({
            "own_turn": ordinal,
            "games": len(rows),
            "bodies": mean([row["bodies"] for row in rows]),
            "line": mean([row["line"] for row in rows]),
            "best_stage": mean([row["best_stage"] for row in rows]),
            "route_colors": mean([row["route_colors"] for row in rows]),
            "phantom_ready": mean([row["phantom_ready"] for row in rows]),
            "attack_rate": mean([row["attacked"] for row in rows]),
            "phantom_rate": mean([row["phantom"] for row in rows]),
            "lone_body_rate": mean([int(row["bodies"] <= 1) for row in rows]),
        })

    def first_stat(key: str) -> dict[str, Any]:
        values = [game[key] for game in games if game[key] is not None]
        return {
            "reached": len(values),
            "rate": round(len(values) / len(games), 3) if games else 0.0,
            "mean_own_turn": mean(values),
            "median_own_turn": statistics.median(values) if values else None,
        }

    wins = sum(game["result"] == "win" for game in games)
    return {
        "label": label,
        "games": len(games),
        "record": f"{wins}-{len(games) - wins}",
        "win_rate": round(wins / len(games), 4) if games else 0.0,
        "first_dragapult": first_stat("first_dragapult_own_turn"),
        "first_phantom_ready": first_stat("first_phantom_ready_own_turn"),
        "first_phantom": first_stat("first_phantom_own_turn"),
        "curve": curve,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n=== {summary['label']}  {summary['record']} "
          f"({summary['win_rate']:.3f}) over {summary['games']} games")
    for key in ("first_dragapult", "first_phantom_ready", "first_phantom"):
        stat = summary[key]
        print(f"  {key:22} reached {stat['rate']:.3f}  "
              f"mean own-turn {stat['mean_own_turn']}  median {stat['median_own_turn']}")
    print(f"  {'own_turn':>8} {'games':>6} {'bodies':>7} {'line':>6} {'stage':>6} "
          f"{'colors':>7} {'ready':>6} {'atk':>6} {'pd':>6} {'lone':>6}")
    for row in summary["curve"]:
        print(f"  {row['own_turn']:>8} {row['games']:>6} {row['bodies']:>7} "
              f"{row['line']:>6} {row['best_stage']:>6} {row['route_colors']:>7} "
              f"{row['phantom_ready']:>6} {row['attack_rate']:>6} "
              f"{row['phantom_rate']:>6} {row['lone_body_rate']:>6}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--split-report", type=Path)
    parser.add_argument("--exclude-episode", type=int, nargs="*", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    if args.run:
        games = []
        rows = list(csv.DictReader(
            (args.run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
        ))
        for row in rows:
            episode_id = int(row["episode_id"])
            if episode_id in args.exclude_episode:
                continue
            path = (
                args.run / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            game = analyse(path, int(row["detected_submission_agent_index"]))
            if game:
                games.append(game)
        report["run"] = summarise(games, f"live {args.run.name}")
        report["run_games"] = games
        print_summary(report["run"])

    if args.teacher_index:
        boundaries: dict[str, list[int]] = {}
        if args.split_report:
            boundaries = load(args.split_report).get("split_boundaries") or {}
        seen: set[tuple[str, int]] = set()
        games = []
        for row in csv.DictReader(
            args.teacher_index.read_text(encoding="utf-8-sig").splitlines()
        ):
            episode_id = str(row["episode_id"])
            seat = int(row["seat_index"])
            if (episode_id, seat) in seen:
                continue
            seen.add((episode_id, seat))
            boundary = boundaries.get(str(row.get("team_id")))
            if boundary and int(episode_id) <= int(boundary[1]):
                continue
            path = Path(row["replay_path"])
            if not path.is_absolute():
                path = args.teacher_index.parent.parent / path
            game = analyse(path, seat)
            if game:
                game["team_id"] = row.get("team_id")
                games.append(game)
        report["teachers"] = summarise(games, "teachers (held-out split)")
        report["teacher_games"] = games
        print_summary(report["teachers"])

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
