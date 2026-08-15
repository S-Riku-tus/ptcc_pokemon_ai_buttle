"""How the wall matchups are actually played, per own turn.

Crustle and Cornerstone Mask Ogerpon ex only prevent damage from *attacks*.
Froslass's Freezing Shroud and Munkidori's Adrena-Brain are Abilities, so they
are the only sources in the 60 that can put damage on either wall.  Counting
them per game confounds usage with game length - a 40-turn game has more of
everything - so every rate here is per own turn.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts",
             ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v25"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402

from ml.core.replay_io import deck_hash  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
IMMUNE = {345, 117}          # Crustle, Cornerstone Mask Ogerpon ex

RUNS = {
    "v22": [
        "data/runs/grimmsnarl/20260813_grimmsnarl_ml_v22_sub55479857",
        "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v22_b_sub55483874",
        "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v22_c_sub55486680",
        "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v22_d_sub55486691",
    ],
    "v25": ["data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_sub55507909"],
    "alphatcg": ["data/runs/grimmsnarl/20260814_peer_alphatcg_sub55350342"],
}


def decks(steps: list[Any]) -> list[list[int] | None]:
    out: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for side in (0, 1):
            action = (steps[1][side] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                out[side] = [int(value) for value in action]
    return out


def walk(replay: dict[str, Any], seat: int) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    lists = decks(steps)
    if lists[seat] is None or deck_hash(lists[seat]) != OUR_DECK_HASH:
        return None
    rewards = replay.get("rewards") or [None, None]
    won = rewards[seat] is not None and rewards[seat] > (rewards[1 - seat] or 0)

    own_turns: set[int] = set()
    wall_turns: set[int] = set()
    adrena = 0
    adrena_wall = 0
    froslass_in_play: list[int] = []
    attacks_wall = 0
    ends_on_wall_turn = 0
    retreats_on_wall_turn = 0
    ever_wall = False

    for index, step in enumerate(steps[:-1]):
        record = step[seat] if seat < len(step) else None
        if not record or record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        current = observation.get("current") or {}
        if not options or not current.get("players"):
            continue
        turn = int(current.get("turn", -1))
        own_turns.add(turn)
        us = current["players"][seat]
        them = current["players"][1 - seat]
        active = (mf._cards(them, "active") or [{}])[0]
        walled = int(active.get("id", -1)) in IMMUNE
        ever_wall = ever_wall or walled
        if walled:
            wall_turns.add(turn)
            froslass_in_play.append(sum(
                1 for card in mf._cards(us, "active") + mf._cards(us, "bench")
                if int(card.get("id", -1)) == mf.FROSLASS_ID
            ))
        action = (steps[index + 1][seat] or {}).get("action")
        picked = [int(v) for v in action
                  if isinstance(v, int) and 0 <= int(v) < len(options)] \
            if isinstance(action, list) else []
        for choice in picked:
            option = options[choice]
            try:
                kind = mf.action_type(current, option, select)
            except Exception:  # noqa: BLE001
                kind = "?"
            card = mf.candidate_card(current, option, select) or {}
            if kind == "ability" and int(card.get("id", -1)) == mf.MUNKIDORI_ID:
                adrena += 1
                if walled:
                    adrena_wall += 1
            if walled and kind == "attack":
                attacks_wall += 1
            if walled and kind == "end":
                ends_on_wall_turn += 1
            if walled and kind == "retreat":
                retreats_on_wall_turn += 1

    if not ever_wall:
        return None
    return {
        "won": won,
        "family": family(lists[1 - seat]),
        "own_turns": len(own_turns),
        "wall_turns": len(wall_turns),
        "adrena": adrena,
        "adrena_wall": adrena_wall,
        "attacks_wall": attacks_wall,
        "ends_wall": ends_on_wall_turn,
        "retreats_wall": retreats_on_wall_turn,
        "froslass_mean": (sum(froslass_in_play) / len(froslass_in_play)
                          if froslass_in_play else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v25/wall_engine.json",
    )
    args = parser.parse_args()

    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label, dirs in RUNS.items():
        for directory in dirs:
            run = ROOT / directory
            for entry in csv.DictReader(
                (run / "manifest.csv").open(encoding="utf-8-sig")
            ):
                seat_text = entry.get("detected_submission_agent_index", "")
                if seat_text not in {"0", "1"}:
                    continue
                path = (run / "episodes" / entry["episode_id"] / "replay"
                        / f"episode_{entry['episode_id']}.json")
                if not path.exists():
                    continue
                row = walk(json.loads(path.read_text(encoding="utf-8")),
                           int(seat_text))
                if row:
                    row["episode_id"] = int(entry["episode_id"])
                    games[label].append(row)

    print("games where a damage-immune wall was ever Active "
          "(rates are per own turn spent facing the wall)")
    header = (f"{'ver':9}{'games':>6}{'wr':>7}{'ownTurns':>9}{'wallTurns':>10}"
              f"{'adrena/wallTurn':>16}{'attacks/wallTurn':>17}"
              f"{'froslassInPlay':>15}{'end/wallTurn':>13}")
    print(header)
    for label in ("v22", "v25", "alphatcg"):
        rows = games[label]
        if not rows:
            continue
        wall_turns = sum(r["wall_turns"] for r in rows) or 1
        print(
            f"{label:9}{len(rows):6d}"
            f"{sum(r['won'] for r in rows) / len(rows):7.3f}"
            f"{sum(r['own_turns'] for r in rows) / len(rows):9.1f}"
            f"{sum(r['wall_turns'] for r in rows) / len(rows):10.1f}"
            f"{sum(r['adrena_wall'] for r in rows) / wall_turns:16.2f}"
            f"{sum(r['attacks_wall'] for r in rows) / wall_turns:17.2f}"
            f"{sum(r['froslass_mean'] for r in rows) / len(rows):15.2f}"
            f"{sum(r['ends_wall'] for r in rows) / wall_turns:13.2f}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({k: v for k, v in games.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
