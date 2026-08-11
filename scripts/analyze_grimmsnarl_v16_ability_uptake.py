"""Was Adrena-Brain on the table, and did v15 press it?

Marnie's Grimmsnarl ex has 320 HP and Shadow Bullet deals 180, so a mirror
knock-out is a two-swing plan and 360 damage against 320 leaves exactly 40
points of slack.  Munkidori's Adrena-Brain moves up to 3 damage counters off
one of your Pokemon and onto one of theirs, which is 30 in both directions at
once: two of them between our two swings put the defending Grimmsnarl ex out of
range entirely.  So in the mirror the counter engine, not the attack, decides
whether a Shadow Bullet becomes a prize.

The mirror behaviour split says the winner of a v15 mirror uses Adrena-Brain
7.61 times a game and the loser 3.81, and in our own 15 mirror losses the
opponent used it 7.47 times against our 3.93.  That is only actionable if the
ability was *available* and declined.  The engine answers that exactly: an
Ability activation is a type-10 option in MAIN, so a turn where one was offered
on a Munkidori and no activation was taken is a decision, not a board state.

    python scripts/analyze_grimmsnarl_v16_ability_uptake.py \
        --run data/runs/grimmsnarl/20260810_grimmsnarl_ml_v15_sub55404196 \
        --submission 55404196
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v15"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_v16_prize_conversion import (  # noqa: E402
    deck_label,
    matchup_of,
    own_turn,
)

MAIN_CONTEXT = 0
OPTION_ABILITY = 10
OPTION_ATTACK = 13
AREA_ACTIVE = 4
AREA_BENCH = 5


def ability_body(
    me: dict[str, Any], option: dict[str, Any]
) -> dict[str, Any] | None:
    area = mf._int(option.get("area"))
    slot = mf._int(option.get("index"))
    if area == AREA_ACTIVE:
        pool = mf._cards(me, "active")
    elif area == AREA_BENCH:
        pool = mf._cards(me, "bench")
    else:
        return None
    return pool[slot] if 0 <= slot < len(pool) else None


def walk(
    replay: dict[str, Any], seat: int, first_player: int
) -> dict[str, Any]:
    steps = replay.get("steps") or []
    turns: dict[int, dict[str, Any]] = {}

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(select, dict) or not isinstance(current, dict):
            continue
        if int(select.get("context", -1)) != MAIN_CONTEXT:
            continue
        players = current.get("players") or []
        if len(players) < 2:
            continue
        me = players[seat]
        turn = own_turn(current, seat, first_player)
        state = turns.setdefault(turn, {
            "offered_munkidori": False,
            "offered_froslass": False,
            "used_munkidori": False,
            "used_froslass": False,
            "attacked": False,
            "own_damage_available": False,
        })

        options = select.get("option") or []
        for option in options:
            if mf._int(option.get("type")) != OPTION_ABILITY:
                continue
            body = ability_body(me, option)
            card_id = int((body or {}).get("id", -1))
            if card_id == mf.MUNKIDORI_ID:
                state["offered_munkidori"] = True
            elif card_id == mf.FROSLASS_ID:
                state["offered_froslass"] = True
        if any(
            float(c.get("maxHp", 0) or 0) - float(c.get("hp", 0) or 0) >= 10.0
            for c in mf._cards(me, "active") + mf._cards(me, "bench")
        ):
            state["own_damage_available"] = True

        action = (steps[index + 1][seat] or {}).get("action")
        if not (isinstance(action, list) and len(action) == 1
                and isinstance(action[0], int)):
            continue
        chosen = int(action[0])
        if not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        option_type = mf._int(option.get("type"))
        if option_type == OPTION_ATTACK:
            state["attacked"] = True
        elif option_type == OPTION_ABILITY:
            body = ability_body(me, option)
            card_id = int((body or {}).get("id", -1))
            if card_id == mf.MUNKIDORI_ID:
                state["used_munkidori"] = True
            elif card_id == mf.FROSLASS_ID:
                state["used_froslass"] = True

    ordered = [turns[t] for t in sorted(turns)]
    offered = [s for s in ordered if s["offered_munkidori"]]
    live = [s for s in offered if s["own_damage_available"]]
    return {
        "own_turns": len(ordered),
        "munkidori_offered_turns": len(offered),
        "munkidori_used_turns": sum(1 for s in offered if s["used_munkidori"]),
        "munkidori_live_turns": len(live),
        "munkidori_live_used": sum(1 for s in live if s["used_munkidori"]),
        "munkidori_declined_live": sum(
            1 for s in live if not s["used_munkidori"]
        ),
        "froslass_offered_turns": sum(
            1 for s in ordered if s["offered_froslass"]
        ),
        "froslass_used_turns": sum(1 for s in ordered if s["used_froslass"]),
    }


def load(run_dir: Path, submission: str) -> list[dict[str, Any]]:
    games = []
    for raw in csv.DictReader(
        (run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if raw["state"] != "COMPLETED":
            continue
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC":
            continue
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if a0 == a1:
            continue
        seat = 0 if a0 == submission else 1
        episode_id = int(raw["episode_id"])
        path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        if len(steps) < 3:
            continue
        decks: list[list[int] | None] = [None, None]
        for side in (0, 1):
            action = (steps[1][side] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[side] = [int(v) for v in action]
        label = deck_label(decks[1 - seat])
        first_player = -1
        for step in reversed(steps):
            if seat >= len(step):
                continue
            current = (
                (step[seat] or {}).get("observation") or {}
            ).get("current")
            if isinstance(current, dict) and int(
                current.get("firstPlayer", -1)
            ) >= 0:
                first_player = int(current.get("firstPlayer", -1))
                break
        rewards = replay.get("rewards") or [None, None]
        won = None
        if rewards[seat] is not None:
            other = rewards[1 - seat]
            won = bool(rewards[seat] > (other if other is not None else 0))
        mirror = matchup_of(label) == "mirror"
        games.append({
            "episode_id": episode_id,
            "matchup": matchup_of(label),
            "won": won,
            "us": walk(replay, seat, first_player),
            "them": walk(replay, 1 - seat, first_player) if mirror else None,
        })
    return games


def block(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    total = {
        key: sum(row[key] for row in rows)
        for key in (
            "own_turns", "munkidori_offered_turns", "munkidori_used_turns",
            "munkidori_live_turns", "munkidori_live_used",
            "munkidori_declined_live", "froslass_offered_turns",
            "froslass_used_turns",
        )
    } if rows else {}
    if not rows:
        return {"name": name, "games": 0}
    return {
        "name": name,
        "games": len(rows),
        "own_turns": total["own_turns"],
        "munkidori_offered_turns": total["munkidori_offered_turns"],
        "munkidori_uptake": round(
            total["munkidori_used_turns"]
            / max(1, total["munkidori_offered_turns"]), 3
        ),
        "munkidori_live_uptake": round(
            total["munkidori_live_used"]
            / max(1, total["munkidori_live_turns"]), 3
        ),
        "munkidori_declined_live_per_game": round(
            total["munkidori_declined_live"] / len(rows), 3
        ),
        "munkidori_offered_share_of_turns": round(
            total["munkidori_offered_turns"] / max(1, total["own_turns"]), 3
        ),
        "froslass_uptake": round(
            total["froslass_used_turns"]
            / max(1, total["froslass_offered_turns"]), 3
        ),
        "froslass_offered_share_of_turns": round(
            total["froslass_offered_turns"] / max(1, total["own_turns"]), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--submission", action="append", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    games: list[dict[str, Any]] = []
    for run_dir, submission in zip(args.run, args.submission):
        games.extend(load(run_dir, submission))

    mirrors = [g for g in games if g["matchup"] == "mirror"]
    by_matchup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_matchup[game["matchup"]].append(game["us"])

    out = {
        "us_overall": block([g["us"] for g in games], "us_all"),
        "us_wins": block([g["us"] for g in games if g["won"]], "us_wins"),
        "us_losses": block(
            [g["us"] for g in games if g["won"] is False], "us_losses"
        ),
        "mirror_us": block([g["us"] for g in mirrors], "mirror_us"),
        "mirror_them": block([g["them"] for g in mirrors], "mirror_them"),
        "mirror_us_in_losses": block(
            [g["us"] for g in mirrors if g["won"] is False], "mirror_us_loss"
        ),
        "mirror_them_in_our_losses": block(
            [g["them"] for g in mirrors if g["won"] is False], "mirror_them_w"
        ),
        "by_matchup": {
            key: block(value, key)
            for key, value in sorted(
                by_matchup.items(), key=lambda kv: -len(kv[1])
            )
        },
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
