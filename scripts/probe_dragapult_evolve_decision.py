"""The decision behind the defect: evolve the Active into Dragapult ex, or not?

Outcome counting shows we create an unpowered Active Dragapult ex 22% of the
time against the teachers' 5.8%.  A fix has to move a *decision*, so this
isolates the decision: every main-phase option list that offers an evolution
into Dragapult ex, split by where it would land and whether that body already
carries Fire and Psychic.

The agent is forced onto the trajectory the teacher really played so its
intra-turn history stays aligned, exactly as in analyze_dragapult_play_gap.py.

Usage:
  python scripts/probe_dragapult_evolve_decision.py \
      --agent-dir agents/dragapult/dragapult_ml_v2 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --split-report experiments/dragapult_ml_v2/train_full.json \
      --report experiments/dragapult_ml_v2/evolve_decision.json
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
for path in (ROOT, ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_loader import load_dir_agent_module  # noqa: E402

DRAGAPULT, DRAKLOAK = 121, 120
FIRE, PSYCHIC = 2, 5
MAIN = 0
OPT_EVOLVE = 9


def choice_of(module: Any, observation: dict[str, Any]) -> int | None:
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        picked = list(module._fallback_agent(observation))
        return picked[0] if len(picked) == 1 else None
    index = ranker.choose(observation)
    if index is None:
        picked = list(module._fallback_agent(observation))
        return picked[0] if len(picked) == 1 else None
    guarded = getattr(module, "_guarded_index", None)
    if guarded is not None:
        # The guard returns None when it declines, not the unchanged index.
        replacement = guarded(observation, index)
        if replacement is not None:
            index = replacement
    return index


def target_of(observation: dict[str, Any], option: dict[str, Any]
              ) -> tuple[str, bool] | None:
    """Where an EVOLVE would land and whether that body is already powered.

    An evolution option carries the hand index of the evolving card; the target
    is identified by the option's ``inPlayArea``/``inPlayIndex`` pair when
    present, and otherwise by the only legal Drakloak.
    """
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    hand = mine.get("hand") or []
    hand_index = int(option.get("index", -1))
    card = hand[hand_index] if 0 <= hand_index < len(hand) else {}
    if not isinstance(card, dict) or int(card.get("id", -1)) != DRAGAPULT:
        return None

    active = mine.get("active") or []
    if isinstance(active, dict):
        active = [active]
    bench = list(mine.get("bench") or [])

    area = option.get("inPlayArea", option.get("area"))
    slot = option.get("inPlayIndex", option.get("targetIndex"))
    body: dict[str, Any] | None = None
    where = "unknown"
    if area is not None and slot is not None:
        zone = active if int(area) == 4 else bench
        position = int(slot)
        if 0 <= position < len(zone):
            body = zone[position]
            where = "active" if int(area) == 4 else "bench"
    if body is None:
        candidates = [("active", card) for card in active
                      if isinstance(card, dict)
                      and int(card.get("id", -1)) == DRAKLOAK]
        candidates += [("bench", card) for card in bench
                       if isinstance(card, dict)
                       and int(card.get("id", -1)) == DRAKLOAK]
        if len(candidates) != 1:
            return None
        where, body = candidates[0]
    colors = [int(value) for value in ((body or {}).get("energies") or [])]
    return where, (FIRE in colors and PSYCHIC in colors)


def walk(module: Any, replay: dict[str, Any], seat: int, sink: Counter) -> None:
    steps = replay.get("steps") or []
    module.diag_reset()
    for index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        select = observation.get("select")
        action = (steps[index + 1][seat].get("action")
                  if index + 1 < len(steps) else None)
        if select is None:
            ranker = getattr(module, "_RANKER", None)
            if ranker is not None:
                ranker.reset()
            module._fallback_agent(observation)
            continue
        if not isinstance(action, list) or len(action) != 1:
            continue
        options = select.get("option") or []
        played = int(action[0])
        if not 0 <= played < len(options):
            continue

        ours = choice_of(module, observation)
        ranker = getattr(module, "_RANKER", None)
        if ranker is not None:
            ranker.observe_external(observation, played)

        if int(select.get("context", -1)) != MAIN:
            continue
        # Group the offer by the best-case landing: an option list can offer
        # both an Active and a bench evolution, and they are different choices.
        offers: dict[tuple[str, bool], set[int]] = {}
        for position, option in enumerate(options):
            if int(option.get("type", -1)) != OPT_EVOLVE:
                continue
            target = target_of(observation, option)
            if target is None:
                continue
            offers.setdefault(target, set()).add(position)
        for (where, powered), positions in offers.items():
            key = f"{where}_{'powered' if powered else 'unpowered'}"
            sink[f"offer_{key}"] += 1
            if played in positions:
                sink[f"teacher_take_{key}"] += 1
            if ours in positions:
                sink[f"our_take_{key}"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--split-report", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    module = load_dir_agent_module(args.agent_dir.resolve())
    boundaries: dict[str, list[int]] = {}
    if args.split_report:
        boundaries = json.loads(
            args.split_report.read_text(encoding="utf-8")
        ).get("split_boundaries") or {}

    sink: Counter = Counter()
    seen: set[tuple[str, int]] = set()
    episodes = 0
    for row in csv.DictReader(
        args.teacher_index.read_text(encoding="utf-8-sig").splitlines()
    ):
        episode_id = str(row["episode_id"])
        seat = int(row["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        boundary = boundaries.get(str(row.get("team_id")))
        if boundary:
            low, high = int(boundary[0]), int(boundary[1])
            in_split = (
                int(episode_id) > high if args.split == "test"
                else low < int(episode_id) <= high if args.split == "validation"
                else int(episode_id) <= low
            )
            if not in_split:
                continue
        path = Path(row["replay_path"])
        if not path.is_absolute():
            path = args.teacher_index.parent.parent / path
        if not path.exists():
            continue
        walk(module, json.loads(path.read_text(encoding="utf-8")), seat, sink)
        episodes += 1
        if args.limit and episodes >= args.limit:
            break

    print(f"episodes {episodes}")
    print(f"\n{'evolve target':24} {'offers':>8} {'teacher':>9} {'ours':>9} "
          f"{'delta':>8}")
    report = {"episodes": episodes, "rows": []}
    for key in ("active_powered", "active_unpowered",
                "bench_powered", "bench_unpowered"):
        offers = sink[f"offer_{key}"]
        if not offers:
            continue
        teacher = sink[f"teacher_take_{key}"] / offers
        ours = sink[f"our_take_{key}"] / offers
        print(f"{key:24} {offers:>8} {teacher:>9.4f} {ours:>9.4f} "
              f"{ours - teacher:>+8.4f}")
        report["rows"].append({
            "target": key, "offers": offers,
            "teacher_rate": round(teacher, 4), "our_rate": round(ours, 4),
            "delta": round(ours - teacher, 4),
        })

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
