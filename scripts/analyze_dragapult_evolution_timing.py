"""Does the agent evolve into Dragapult ex before it can use it?

Trace of a lost Mega Lucario game (episode 93603391) showed the agent evolving
its Active Drakloak into Dragapult ex on a turn with no Energy in hand.  The
opponent knocked it out immediately and took two prizes; the agent never
attacked in the whole game.  Dragapult ex is a two-prize Pokemon with a
two-color attack cost, so evolving it into an Active it cannot power hands the
opponent a third of the prize race for nothing.

This counts, for every evolution into Dragapult ex:

* whether it landed in the Active slot or on the bench,
* whether that Dragapult ex ever attacked at all,
* whether it was knocked out before it ever attacked,
* how much Energy it had at the end of the turn it was made.

and compares us against the teachers on the same list.

Usage:
  python scripts/analyze_dragapult_evolution_timing.py \
      --run data/submissions/submission_55550682_dragapult_v2 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --teacher-limit 400 \
      --report experiments/dragapult_ml_v2/evolution_timing.json
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DRAGAPULT = 121
FIRE, PSYCHIC = 2, 5
PHANTOM_DIVE = 154
OPT_ATTACH, OPT_EVOLVE, OPT_ATTACK = 8, 9, 13


def walk(replay: dict[str, Any], seat: int, sinks: list[Counter]) -> None:
    steps = replay.get("steps") or []
    seen_turns: set[int] = set()
    own_turn = 0
    # serial -> what we know about each Dragapult ex we ever made.
    tracked: dict[int, dict[str, Any]] = {}
    alive: set[int] = set()

    def bodies(player: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        active = player.get("active") or []
        if isinstance(active, dict):
            active = [active]
        out = [("active", card) for card in active if isinstance(card, dict)]
        out += [("bench", card) for card in (player.get("bench") or [])
                if isinstance(card, dict)]
        return out

    for index, pair in enumerate(steps):
        payload = pair[seat]
        observation = payload.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) != 2:
            continue
        your = int(current.get("yourIndex", seat))
        mine = players[your]
        if payload.get("status") != "ACTIVE":
            continue
        turn = current.get("turn")
        if isinstance(turn, int) and turn not in seen_turns:
            seen_turns.add(turn)
            own_turn += 1

        present: set[int] = set()
        for slot, card in bodies(mine):
            if int(card.get("id", -1)) != DRAGAPULT:
                continue
            serial = int(card.get("serial", -1))
            present.add(serial)
            colors = [int(value) for value in (card.get("energies") or [])]
            entry = tracked.get(serial)
            if entry is None:
                entry = tracked[serial] = {
                    "born_turn": own_turn, "born_slot": slot,
                    "attacked": False, "dived": False,
                    "colors_when_born": list(colors),
                    "ready_when_born": FIRE in colors and PSYCHIC in colors,
                    "ever_ready": False, "turns_seen": 0,
                }
            entry["turns_seen"] += 1
            if FIRE in colors and PSYCHIC in colors:
                entry["ever_ready"] = True
            if entry["born_turn"] == own_turn:
                # Energy can still be attached later in the same turn.
                entry["colors_when_born"] = list(colors)
                entry["ready_when_born"] = FIRE in colors and PSYCHIC in colors

        for serial in alive - present:
            entry = tracked.get(serial)
            if entry is not None:
                entry.setdefault("gone_turn", own_turn)
        alive = present

        select = observation.get("select")
        if select is None:
            continue
        options = select.get("option") or []
        action = (steps[index + 1][seat].get("action")
                  if index + 1 < len(steps) else None)
        if not isinstance(action, list) or len(action) != 1:
            continue
        picked = int(action[0])
        if not 0 <= picked < len(options):
            continue
        chosen = options[picked]
        if int(chosen.get("type", -1)) != OPT_ATTACK:
            continue
        active = mine.get("active") or []
        if isinstance(active, dict):
            active = [active]
        for card in active:
            if isinstance(card, dict) and int(card.get("id", -1)) == DRAGAPULT:
                entry = tracked.get(int(card.get("serial", -1)))
                if entry is not None:
                    entry["attacked"] = True
                    if int(chosen.get("attackId", -1)) == PHANTOM_DIVE:
                        entry["dived"] = True

    for sink in sinks:
        sink["games"] += 1
    for entry in tracked.values():
        for sink in sinks:
            sink["dragapult_made"] += 1
            sink[f"born_{entry['born_slot']}"] += 1
            if entry["attacked"]:
                sink["ever_attacked"] += 1
                if entry["dived"]:
                    sink["ever_dived"] += 1
            else:
                sink["never_attacked"] += 1
                sink[f"never_attacked_born_{entry['born_slot']}"] += 1
                # Disappearing from the board without ever attacking is a
                # knock-out (or a discard); either way two prizes for nothing.
                if "gone_turn" in entry:
                    sink["never_attacked_and_died"] += 1
                    if entry["born_slot"] == "active":
                        sink["never_attacked_and_died_active"] += 1
            if entry["born_slot"] == "active" and not entry["ready_when_born"]:
                sink["born_active_unpowered"] += 1
                if not entry["attacked"] and "gone_turn" in entry:
                    sink["born_active_unpowered_and_died"] += 1
            if entry["born_slot"] == "active" and entry["ready_when_born"]:
                sink["born_active_powered"] += 1


def report(sink: Counter, label: str) -> dict[str, Any]:
    made = sink["dragapult_made"]
    games = max(1, sink["games"])
    print(f"\n=== {label}: {sink['games']} games, {made} Dragapult ex created "
          f"({made / games:.3f}/game)")
    if not made:
        return {"label": label}

    def line(key: str, text: str) -> None:
        print(f"  {text:44} {sink[key]:>6} ({sink[key] / made:.3f} of made, "
              f"{sink[key] / games:.3f}/game)")

    line("born_active", "created in the Active slot")
    line("born_bench", "created on the bench")
    line("born_active_unpowered", "created Active WITHOUT Fire+Psychic")
    line("born_active_powered", "created Active WITH Fire+Psychic")
    line("ever_attacked", "attacked at least once")
    line("ever_dived", "used Phantom Dive at least once")
    line("never_attacked", "never attacked")
    line("never_attacked_and_died", "never attacked and left the board")
    line("never_attacked_and_died_active", "  of those, born Active")
    line("born_active_unpowered_and_died",
         "born Active unpowered AND died unused")
    return {"label": label, **dict(sink)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--teacher-limit", type=int, default=400)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    reports = []
    # Losing players are in bad spots by definition, so a rate measured over
    # all our games against all the teachers' games confounds cause with
    # effect.  Splitting both cohorts by result separates them.
    for run in args.run:
        overall, won, lost = Counter(), Counter(), Counter()
        for row in csv.DictReader(
            (run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
        ):
            seat = row.get("detected_submission_agent_index", "")
            if seat not in ("0", "1"):
                continue
            path = (run / "episodes" / str(row["episode_id"]) / "replay"
                    / f"episode_{row['episode_id']}.json")
            if not path.exists():
                continue
            replay = json.loads(path.read_text(encoding="utf-8"))
            rewards = replay.get("rewards") or [0, 0]
            bucket = won if rewards[int(seat)] > rewards[1 - int(seat)] else lost
            walk(replay, int(seat), [overall, bucket])
        reports.append(report(overall, run.name))
        reports.append(report(won, f"{run.name} [wins]"))
        reports.append(report(lost, f"{run.name} [losses]"))

    if args.teacher_index:
        overall, won, lost = Counter(), Counter(), Counter()
        seen: set[tuple[str, int]] = set()
        used = 0
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
            replay = json.loads(path.read_text(encoding="utf-8"))
            seat = int(row["seat_index"])
            rewards = replay.get("rewards") or [0, 0]
            bucket = won if rewards[seat] > rewards[1 - seat] else lost
            walk(replay, seat, [overall, bucket])
            used += 1
            if args.teacher_limit and used >= args.teacher_limit:
                break
        reports.append(report(overall, f"teachers (n={used})"))
        reports.append(report(won, "teachers [wins]"))
        reports.append(report(lost, "teachers [losses]"))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
