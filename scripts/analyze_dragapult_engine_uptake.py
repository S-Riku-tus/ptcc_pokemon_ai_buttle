"""Per-own-turn uptake of the Dragapult engine's free actions.

The deck's repeatable advantages are Abilities and typed acceleration, not
attacks: Recon Directive draws every turn a Drakloak is in play, Adrena-Brain
converts our own chip damage into their damage, Crispin is the only card that
attaches a second Energy in a turn.  A take rate has to be measured per own
turn against the turns where the action was actually offered - a per-decision
denominator mixes in every unrelated decision of the turn.

Usage:
  python scripts/analyze_dragapult_engine_uptake.py \
      --run data/submissions/submission_55545828_dragapult_v1 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --split-report experiments/dragapult_ml_v1/training_report.json \
      --report experiments/dragapult_ml_v1/engine_uptake.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DREEPY, DRAKLOAK, DRAGAPULT, MUNKIDORI, FEZANDIPITI, MEOWTH = 119, 120, 121, 112, 140, 1071
CRISPIN, BOSS, POFFIN, ULTRA_BALL, POKE_PAD, LILLIE, DAWN, JUDGE = (
    1198, 1182, 1086, 1121, 1152, 1227, 1231, 1213
)
STRETCHER, HAMMER, STAMP, TOWER = 1097, 1120, 1080, 1246
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY, OPT_ATTACK, OPT_END = 7, 8, 9, 10, 13, 14

ABILITY_HOLDERS = {
    DRAKLOAK: "recon_directive",
    MUNKIDORI: "adrena_brain",
    FEZANDIPITI: "flip_the_script",
    MEOWTH: "last_ditch_catch",
}
TRACKED_PLAYS = {
    CRISPIN: "crispin", BOSS: "boss", POFFIN: "poffin", ULTRA_BALL: "ultra_ball",
    POKE_PAD: "poke_pad", LILLIE: "lillie", DAWN: "dawn", JUDGE: "judge",
    STRETCHER: "stretcher", HAMMER: "hammer", STAMP: "stamp", TOWER: "tower",
}


def hand_card_id(observation: dict[str, Any], option: dict[str, Any]) -> int:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    hand = (players[your] if your in (0, 1) else {}).get("hand") or []
    index = int(option.get("index", -1))
    if not 0 <= index < len(hand) or not isinstance(hand[index], dict):
        return -1
    return int(hand[index].get("id", -1))


def holder_id(observation: dict[str, Any], option: dict[str, Any]) -> int:
    """Resolve an ABILITY option's holder.

    OptionType.ABILITY carries ``area``/``index``, not the ``inPlayArea``/
    ``inPlayIndex`` pair that ATTACH and EVOLVE use.  Reading the wrong pair
    silently resolves every ability to "unknown".
    """
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    area = int(option.get("area", -1))
    index = int(option.get("index", -1))
    zone = {
        2: mine.get("hand"), 3: mine.get("discard"), 4: mine.get("active"),
        5: mine.get("bench"), 7: current.get("stadium"),
    }.get(area) or []
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return -1
    card = zone[index]
    return int(card.get("id", -1)) if isinstance(card, dict) else -1


def label(observation: dict[str, Any], option: dict[str, Any]) -> str | None:
    option_type = int(option.get("type", -1))
    if option_type == OPT_ABILITY:
        return ABILITY_HOLDERS.get(holder_id(observation, option))
    if option_type == OPT_PLAY:
        return TRACKED_PLAYS.get(hand_card_id(observation, option))
    if option_type == OPT_ATTACK:
        return "attack"
    if option_type == OPT_EVOLVE:
        card = hand_card_id(observation, option)
        return {DRAKLOAK: "evolve_drakloak", DRAGAPULT: "evolve_dragapult"}.get(card)
    if option_type == OPT_ATTACH:
        return "attach"
    return None


def analyse(path: Path, seat: int) -> dict[str, Any] | None:
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    rewards = replay.get("rewards") or [0, 0]
    offered: dict[int, set[str]] = defaultdict(set)
    taken: dict[int, set[str]] = defaultdict(set)
    own_turns: list[int] = []
    for step_index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, dict) or not isinstance(select, dict):
            continue
        if int(select.get("context", -1)) != 0:
            continue
        turn = int(current.get("turn") or 0)
        if turn not in own_turns:
            own_turns.append(turn)
        options = select.get("option") or []
        for option in options:
            name = label(observation, option)
            if name:
                offered[turn].add(name)
        action = (
            steps[step_index + 1][seat].get("action")
            if step_index + 1 < len(steps) else None
        )
        if not isinstance(action, list) or len(action) == 60:
            continue
        for index in action:
            if not isinstance(index, int) or not 0 <= index < len(options):
                continue
            name = label(observation, options[index])
            if name:
                taken[turn].add(name)
    if not own_turns:
        return None
    return {
        "episode_id": int(
            replay.get("info", {}).get("EpisodeId") or path.stem.split("_")[-1]
        ),
        "result": (
            "win" if rewards[seat] > rewards[1 - seat]
            else "loss" if rewards[seat] < rewards[1 - seat] else "draw"
        ),
        "own_turns": len(own_turns),
        "offered": {str(turn): sorted(values) for turn, values in offered.items()},
        "taken": {str(turn): sorted(values) for turn, values in taken.items()},
    }


def summarise(games: list[dict[str, Any]], name: str) -> dict[str, Any]:
    offer_turns: Counter[str] = Counter()
    take_turns: Counter[str] = Counter()
    total_turns = 0
    for game in games:
        total_turns += game["own_turns"]
        for turn, values in game["offered"].items():
            for value in values:
                offer_turns[value] += 1
        for turn, values in game["taken"].items():
            for value in values:
                take_turns[value] += 1
    rows = []
    for key in sorted(set(offer_turns) | set(take_turns)):
        offers = offer_turns[key]
        takes = take_turns[key]
        rows.append({
            "action": key,
            "offered_turns": offers,
            "taken_turns": takes,
            "take_rate": round(takes / offers, 4) if offers else 0.0,
            "offers_per_game": round(offers / len(games), 3) if games else 0.0,
            "takes_per_game": round(takes / len(games), 3) if games else 0.0,
        })
    return {
        "label": name,
        "games": len(games),
        "own_turns": total_turns,
        "own_turns_per_game": round(total_turns / len(games), 2) if games else 0.0,
        "rows": rows,
    }


def load_games(args: argparse.Namespace) -> tuple[list, list]:
    live: list[dict[str, Any]] = []
    teachers: list[dict[str, Any]] = []
    if args.run:
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
            game = analyse(path, int(row["detected_submission_agent_index"]))
            if game:
                live.append(game)
    if args.teacher_index:
        boundaries: dict[str, list[int]] = {}
        if args.split_report:
            boundaries = json.loads(
                args.split_report.read_text(encoding="utf-8")
            ).get("split_boundaries") or {}
        seen: set[tuple[str, int]] = set()
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
                teachers.append(game)
    return live, teachers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--split-report", type=Path)
    parser.add_argument("--exclude-episode", type=int, nargs="*", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    live, teachers = load_games(args)
    report = {}
    if live:
        report["run"] = summarise(live, "live")
    if teachers:
        report["teachers"] = summarise(teachers, "teachers")

    if "run" in report and "teachers" in report:
        rows = {row["action"]: row for row in report["run"]["rows"]}
        other = {row["action"]: row for row in report["teachers"]["rows"]}
        print(f"{'action':20} {'live take':>10} {'teach take':>11} {'live/g':>8} "
              f"{'teach/g':>8} {'ratio':>7}")
        for key in sorted(set(rows) | set(other)):
            a = rows.get(key, {"take_rate": 0.0, "takes_per_game": 0.0})
            b = other.get(key, {"take_rate": 0.0, "takes_per_game": 0.0})
            ratio = (
                a["takes_per_game"] / b["takes_per_game"]
                if b["takes_per_game"] else float("inf")
            )
            print(f"{key:20} {a['take_rate']:>10.3f} {b['take_rate']:>11.3f} "
                  f"{a['takes_per_game']:>8.3f} {b['takes_per_game']:>8.3f} "
                  f"{ratio:>7.2f}")
        print(f"\nown turns per game: live {report['run']['own_turns_per_game']} "
              f"teachers {report['teachers']['own_turns_per_game']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
