"""Where the typed Energy actually lands: our run vs the teachers.

Uptake analysis showed the action *rates* already match the teachers, so the
gap has to be inside the argument of the action.  Phantom Dive needs one Fire
and one Psychic on the *same* body, so spreading the two colours across two
bodies is a full attachment behind while looking identical to any count of
"route attachments".

Usage:
  python scripts/analyze_dragapult_attach_targets.py \
      --run data/submissions/submission_55545828_dragapult_v1 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --split-report experiments/dragapult_ml_v1/training_report.json \
      --report experiments/dragapult_ml_v1/attach_targets.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

FIRE, PSYCHIC, DARK = 2, 5, 7
DREEPY, DRAKLOAK, DRAGAPULT, MUNKIDORI = 119, 120, 121, 112
LINE_STAGE = {DREEPY: 0, DRAKLOAK: 1, DRAGAPULT: 2}
OPT_ATTACH, OPT_EVOLVE = 8, 9


def own_cards(current: dict[str, Any], seat: int) -> dict[str, Any]:
    players = current.get("players") or [{}, {}]
    return players[seat] if seat in (0, 1) else {}


def target_of(current: dict[str, Any], option: dict[str, Any], seat: int):
    mine = own_cards(current, seat)
    area = int(option.get("inPlayArea", -1))
    index = int(option.get("inPlayIndex", -1))
    zone = mine.get("active") if area == 4 else mine.get("bench") if area == 5 else []
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return None
    card = zone[index]
    return card if isinstance(card, dict) else None


def hand_card(current: dict[str, Any], option: dict[str, Any], seat: int):
    hand = own_cards(current, seat).get("hand") or []
    index = int(option.get("index", -1))
    if not 0 <= index < len(hand) or not isinstance(hand[index], dict):
        return None
    return hand[index]


def route_eta(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 99
    stage = LINE_STAGE.get(int(card.get("id", -1)), -1)
    if stage < 0:
        return 99
    energies = [int(value) for value in card.get("energies") or []]
    missing = int(FIRE not in energies) + int(PSYCHIC not in energies)
    return (2 - stage) + missing


def classify(current: dict[str, Any], option: dict[str, Any], seat: int) -> str | None:
    """Label one taken ATTACH by what it does for the Phantom Dive route."""
    source = hand_card(current, option, seat)
    target = target_of(current, option, seat)
    if source is None or target is None:
        return None
    source_id = int(source.get("id", -1))
    target_id = int(target.get("id", -1))
    energies = [int(value) for value in target.get("energies") or []]
    if source_id == DARK:
        return "dark_to_munkidori" if target_id == MUNKIDORI else "dark_elsewhere"
    if source_id not in (FIRE, PSYCHIC):
        return "other_source"
    if target_id not in LINE_STAGE:
        return "route_color_off_line"
    if source_id in energies:
        return "duplicate_color"
    other = PSYCHIC if source_id == FIRE else FIRE
    if other in energies:
        return f"completes_stage{LINE_STAGE[target_id]}"
    return f"first_color_stage{LINE_STAGE[target_id]}"


def best_alternative(current: dict[str, Any], select: dict[str, Any], seat: int) -> str:
    """The best route class available among all offered ATTACH options."""
    order = [
        "completes_stage2", "completes_stage1", "completes_stage0",
        "first_color_stage2", "first_color_stage1", "first_color_stage0",
        "dark_to_munkidori", "route_color_off_line", "duplicate_color",
        "dark_elsewhere", "other_source",
    ]
    available = set()
    for option in select.get("option") or []:
        if int(option.get("type", -1)) != OPT_ATTACH:
            continue
        label = classify(current, option, seat)
        if label:
            available.add(label)
    for label in order:
        if label in available:
            return label
    return "none"


def analyse(path: Path, seat: int) -> dict[str, Any]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    taken: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    evolve_energy: Counter[str] = Counter()
    for step_index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, dict) or not isinstance(select, dict):
            continue
        action = (
            steps[step_index + 1][seat].get("action")
            if step_index + 1 < len(steps) else None
        )
        if not isinstance(action, list) or len(action) == 60:
            continue
        options = select.get("option") or []
        for index in action:
            if not isinstance(index, int) or not 0 <= index < len(options):
                continue
            option = options[index]
            option_type = int(option.get("type", -1))
            if option_type == OPT_ATTACH:
                label = classify(current, option, seat)
                if label:
                    taken[label] += 1
                    best = best_alternative(current, select, seat)
                    if best != label:
                        missed[f"{label}<-{best}"] += 1
            elif option_type == OPT_EVOLVE:
                source = hand_card(current, option, seat)
                target = target_of(current, option, seat)
                if source is None or target is None:
                    continue
                if int(source.get("id", -1)) != DRAGAPULT:
                    continue
                energies = [int(v) for v in target.get("energies") or []]
                colors = int(FIRE in energies) + int(PSYCHIC in energies)
                evolve_energy[f"pult_onto_{colors}_colors"] += 1
    rewards = replay.get("rewards") or [0, 0]
    return {
        "episode_id": int(
            replay.get("info", {}).get("EpisodeId") or path.stem.split("_")[-1]
        ),
        "result": (
            "win" if rewards[seat] > rewards[1 - seat]
            else "loss" if rewards[seat] < rewards[1 - seat] else "draw"
        ),
        "taken": dict(taken),
        "missed": dict(missed),
        "evolve_energy": dict(evolve_energy),
    }


def summarise(games: list[dict[str, Any]], label: str) -> dict[str, Any]:
    taken: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    evolve: Counter[str] = Counter()
    for game in games:
        taken.update(game["taken"])
        missed.update(game["missed"])
        evolve.update(game["evolve_energy"])
    total = sum(taken.values()) or 1
    return {
        "label": label,
        "games": len(games),
        "attachments": sum(taken.values()),
        "per_game": {
            key: round(value / len(games), 3) for key, value in taken.most_common()
        } if games else {},
        "share": {key: round(value / total, 4) for key, value in taken.most_common()},
        "downgrades": dict(missed.most_common(12)),
        "evolve_energy_per_game": {
            key: round(value / len(games), 3) for key, value in evolve.most_common()
        } if games else {},
    }


def collect(args: argparse.Namespace) -> tuple[list, list]:
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
            live.append(analyse(path, int(row["detected_submission_agent_index"])))
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
            key = (str(row["episode_id"]), int(row["seat_index"]))
            if key in seen:
                continue
            seen.add(key)
            boundary = boundaries.get(str(row.get("team_id")))
            if boundary and int(row["episode_id"]) <= int(boundary[1]):
                continue
            path = Path(row["replay_path"])
            if not path.is_absolute():
                path = args.teacher_index.parent.parent / path
            game = analyse(path, int(row["seat_index"]))
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

    live, teachers = collect(args)
    report: dict[str, Any] = {}
    if live:
        report["run"] = summarise(live, "live")
    if teachers:
        report["teachers"] = summarise(teachers, "teachers")
    if "run" in report and "teachers" in report:
        keys = sorted(set(report["run"]["per_game"]) | set(report["teachers"]["per_game"]))
        print(f"{'attach class':26} {'live/g':>8} {'teach/g':>8} "
              f"{'live%':>7} {'teach%':>7}")
        for key in keys:
            a = report["run"]["per_game"].get(key, 0.0)
            b = report["teachers"]["per_game"].get(key, 0.0)
            sa = report["run"]["share"].get(key, 0.0)
            sb = report["teachers"]["share"].get(key, 0.0)
            print(f"{key:26} {a:>8.3f} {b:>8.3f} {sa:>7.3f} {sb:>7.3f}")
        print("\nDragapult evolved onto a body already holding N route colors:")
        for name in ("pult_onto_0_colors", "pult_onto_1_colors", "pult_onto_2_colors"):
            a = report["run"]["evolve_energy_per_game"].get(name, 0.0)
            b = report["teachers"]["evolve_energy_per_game"].get(name, 0.0)
            print(f"  {name:24} live {a:>6.3f}/g   teachers {b:>6.3f}/g")
        print("\nlive downgrades (chosen <- best available):")
        for key, value in report["run"]["downgrades"].items():
            print(f"  {value:>4}  {key}")
        print("\nteacher downgrades:")
        for key, value in report["teachers"]["downgrades"].items():
            print(f"  {value:>4}  {key}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
