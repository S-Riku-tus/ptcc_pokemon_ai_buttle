"""Why was the Active Dragapult ex not able to use Phantom Dive this turn?

The tail of the Phantom Dive distribution - the games that yield <=1 dive and
are lost 85% of the time - is not an arrival problem.  Dragapult ex reaches the
board at almost the same own turn in tail and non-tail games (6.6 vs 5.6).  What
differs is how many own turns it is *powered*: 0.6 against 4.6.

Phantom Dive costs one Fire and one Psychic.  So for every own turn where our
Active is a Dragapult ex that cannot dive, exactly one of these is true, and
they need different fixes:

* the missing color was in hand and we did not attach it - a policy defect
* the missing color was not in hand and no search for it was in hand - variance
* a search that finds Energy was in hand and we played something else - a
  policy defect one step upstream
* the energy was there last turn and is gone - the opponent removed it, or the
  Pokemon was swapped

Also reports the same tally for the teachers, so "we are unlucky" can be
distinguished from "we play it differently".

Usage:
  python scripts/analyze_dragapult_energy_readiness.py \
      --run data/submissions/submission_55550682_dragapult_v2 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --teacher-limit 400 \
      --report experiments/dragapult_ml_v2/energy_readiness.json
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

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}

DRAGAPULT = 121
FIRE, PSYCHIC, DARK = 2, 5, 7
COLOR_NAME = {FIRE: "Fire", PSYCHIC: "Psychic", DARK: "Dark"}
# Basic Energy in this list, by the color they provide.
ENERGY_COLOR = {
    card_id: int(card.get("energyColor", -1))
    for card_id, card in CARDS.items()
    if str(card.get("kind", "")).upper().startswith("ENERGY")
}
# Cards that can put the missing color onto a Pokemon this turn, or find it.
CRISPIN = 1198
ENERGY_FINDERS = {CRISPIN: "Crispin", 1152: "Poke Pad", 1227: "Lillie's Determination"}

MAIN = 0
OPT_PLAY, OPT_ATTACH, OPT_ATTACK = 7, 8, 13


def energy_color(card: dict[str, Any]) -> int:
    """The color a hand card would provide if attached, or -1.

    Basic Energy card ids equal the color they provide (2 = {R}, 5 = {P},
    7 = {D}), and this deck plays no other Energy, so the id is the color.
    """
    card_id = int(card.get("id", -1))
    return card_id if card_id in COLOR_NAME else -1


def walk(replay: dict[str, Any], seat: int, sink: Counter) -> None:
    steps = replay.get("steps") or []
    seen_turns: set[int] = set()
    own_turn = 0
    # Per own turn: what the Active Dragapult ex had, and what we did about it.
    state: dict[str, Any] | None = None
    previous_colors: list[int] = []

    def close() -> None:
        if state is None or state["colors"] is None:
            return
        colors = state["colors"]
        if FIRE in colors and PSYCHIC in colors:
            sink["turns_dive_ready"] += 1
            if state["attacked_phantom"]:
                sink["ready_and_dived"] += 1
            else:
                sink["ready_but_did_not_dive"] += 1
            return
        sink["turns_pult_active_not_ready"] += 1
        missing = [
            color for color in (FIRE, PSYCHIC) if color not in colors
        ]
        sink[f"missing_{len(missing)}_colors"] += 1
        if state["lost_color"]:
            sink["cause_energy_was_removed"] += 1
            return
        if state["missing_in_hand"]:
            if state["attached_elsewhere"]:
                sink["cause_had_it_attached_elsewhere"] += 1
            elif state["attached_nothing"]:
                sink["cause_had_it_attached_nothing"] += 1
            else:
                sink["cause_had_it_attached_wrong_color"] += 1
            return
        if state["finder_in_hand"]:
            sink["cause_finder_in_hand_unused"] += 1
            return
        sink["cause_no_access"] += 1

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
        turn = current.get("turn")

        if payload.get("status") != "ACTIVE":
            continue
        if isinstance(turn, int) and turn not in seen_turns:
            seen_turns.add(turn)
            close()
            own_turn += 1
            state = {
                "colors": None, "attacked_phantom": False,
                "missing_in_hand": False, "finder_in_hand": False,
                "attached_elsewhere": False, "attached_nothing": True,
                "lost_color": False,
            }

        active = mine.get("active") or []
        if isinstance(active, dict):
            active = [active]
        pult = next(
            (card for card in active
             if isinstance(card, dict) and int(card.get("id", -1)) == DRAGAPULT),
            None,
        )
        if pult is None or state is None:
            continue
        colors = [int(value) for value in (pult.get("energies") or [])]
        # Keep the best state reached during the turn: attaching improves it.
        if state["colors"] is None or len(colors) > len(state["colors"]):
            state["colors"] = colors
        if (set(previous_colors) - set(colors)) and previous_colors:
            state["lost_color"] = True
        previous_colors = colors

        hand = [card for card in (mine.get("hand") or []) if isinstance(card, dict)]
        missing = [color for color in (FIRE, PSYCHIC) if color not in colors]
        if any(energy_color(card) in missing for card in hand):
            state["missing_in_hand"] = True
        if any(int(card.get("id", -1)) in ENERGY_FINDERS for card in hand):
            state["finder_in_hand"] = True

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
        chosen_type = int(chosen.get("type", -1))
        if chosen_type == OPT_ATTACH:
            state["attached_nothing"] = False
            hand_index = int(chosen.get("index", -1))
            if 0 <= hand_index < len(hand):
                if energy_color(hand[hand_index]) in missing:
                    # It went somewhere; whether to the Active is decided by
                    # whether the Active's colors improve, checked next step.
                    state["attached_elsewhere"] = True
        if chosen_type == OPT_ATTACK and int(chosen.get("attackId", -1)) == 154:
            state["attacked_phantom"] = True

    close()


def report(sink: Counter, label: str) -> dict[str, Any]:
    active = sink["turns_dive_ready"] + sink["turns_pult_active_not_ready"]
    print(f"\n=== {label}: {active} own turns with Dragapult ex Active")
    if not active:
        return {"label": label}
    print(f"  dive-ready                 {sink['turns_dive_ready']:>6} "
          f"({sink['turns_dive_ready'] / active:.3f})")
    print(f"    dived                    {sink['ready_and_dived']:>6}")
    print(f"    ready but did not dive   {sink['ready_but_did_not_dive']:>6}")
    unready = sink["turns_pult_active_not_ready"]
    print(f"  NOT dive-ready             {unready:>6} "
          f"({unready / active:.3f})")
    for key, name in (
        ("cause_energy_was_removed", "energy was removed / swapped"),
        ("cause_had_it_attached_elsewhere", "had the color, attached elsewhere"),
        ("cause_had_it_attached_nothing", "had the color, attached nothing"),
        ("cause_had_it_attached_wrong_color", "had the color, attached another"),
        ("cause_finder_in_hand_unused", "no color, but a finder was in hand"),
        ("cause_no_access", "no color and no finder - variance"),
    ):
        print(f"    {name:34} {sink[key]:>6} "
              f"({sink[key] / max(1, unready):.3f})")
    return {"label": label, "turns_pult_active": active, **dict(sink)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--teacher-limit", type=int, default=400)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    reports = []
    for run in args.run:
        sink: Counter = Counter()
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
            walk(json.loads(path.read_text(encoding="utf-8")), int(seat), sink)
        reports.append(report(sink, run.name))

    if args.teacher_index:
        sink = Counter()
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
            walk(json.loads(path.read_text(encoding="utf-8")),
                 int(row["seat_index"]), sink)
            used += 1
            if args.teacher_limit and used >= args.teacher_limit:
                break
        reports.append(report(sink, f"teachers (n={used})"))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
