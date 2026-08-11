"""Where does the once-a-turn manual attachment go, and what does it turn on?

Adrena-Brain uptake is 98.6% of the turns it is offered, in wins and in losses
alike, so the mirror gap in counter-engine use is entirely offer-side: the
ability is not declined, it is unavailable.  It becomes available when a
Munkidori has a Darkness attached, and Punk Up cannot supply one - it searches
Basic Darkness for "your Marnie's Pokemon", and Munkidori is not one.  The only
source is the once-a-turn manual attachment, which is also the resource v15's
own attack-access route spends to escape a non-attacker Active.

So this measures the attachment ledger directly - every manual attachment by
target - and the early-turn availability it produces, at a fixed own-turn index
where the outcome has not yet had time to shape the board.

    python scripts/analyze_grimmsnarl_v16_energy_allocation.py \
        --run data/runs/grimmsnarl/20260810_grimmsnarl_ml_v15_sub55404196 \
        --submission 55404196
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
OPTION_ATTACH = 8
NAMES = {
    mf.GRIMMSNARL_EX_ID: "grimmsnarl_ex",
    mf.MORGREM_ID: "morgrem",
    mf.IMPIDIMP_ID: "impidimp",
    mf.MUNKIDORI_ID: "munkidori",
    mf.FROSLASS_ID: "froslass",
    mf.SNORUNT_ID: "snorunt",
}
PROBE_TURNS = (2, 3, 4)


def walk(
    replay: dict[str, Any], seat: int, first_player: int
) -> dict[str, Any]:
    steps = replay.get("steps") or []
    attach_targets: Counter = Counter()
    attach_to_active = 0
    energised_munkidori: dict[int, int] = {}
    munkidori_in_play: dict[int, int] = {}
    energised_grimmsnarl: dict[int, int] = {}

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(current, dict):
            continue
        players = current.get("players") or []
        if len(players) < 2:
            continue
        me = players[seat]
        turn = own_turn(current, seat, first_player)
        in_play = mf._cards(me, "active") + mf._cards(me, "bench")
        for card in in_play:
            card_id = int(card.get("id", -1))
            energy = mf._energy_count(card)
            if card_id == mf.MUNKIDORI_ID:
                munkidori_in_play[turn] = max(
                    munkidori_in_play.get(turn, 0), 1
                )
                if energy >= 1:
                    energised_munkidori[turn] = max(
                        energised_munkidori.get(turn, 0), 1
                    )
            elif card_id == mf.GRIMMSNARL_EX_ID and energy >= 2:
                energised_grimmsnarl[turn] = max(
                    energised_grimmsnarl.get(turn, 0), 1
                )

        if not isinstance(select, dict):
            continue
        if int(select.get("context", -1)) != MAIN_CONTEXT:
            continue
        action = (steps[index + 1][seat] or {}).get("action")
        if not (isinstance(action, list) and len(action) == 1
                and isinstance(action[0], int)):
            continue
        options = select.get("option") or []
        chosen = int(action[0])
        if not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        if mf._int(option.get("type")) != OPTION_ATTACH:
            continue
        area = mf._int(option.get("inPlayArea"))
        slot = mf._int(option.get("inPlayIndex"))
        pool = (
            mf._cards(me, "active") if area == mf.AREA_ACTIVE
            else mf._cards(me, "bench")
        )
        body = pool[slot] if 0 <= slot < len(pool) else None
        body_id = int((body or {}).get("id", -1))
        attach_targets[NAMES.get(body_id, "other")] += 1
        if area == mf.AREA_ACTIVE:
            attach_to_active += 1

    return {
        "attach_targets": dict(attach_targets),
        "attach_total": sum(attach_targets.values()),
        "attach_to_active": attach_to_active,
        "munkidori_energised_by_turn": {
            str(t): energised_munkidori.get(t, 0) for t in PROBE_TURNS
        },
        "munkidori_in_play_by_turn": {
            str(t): munkidori_in_play.get(t, 0) for t in PROBE_TURNS
        },
        "grimmsnarl_ready_by_turn": {
            str(t): energised_grimmsnarl.get(t, 0) for t in PROBE_TURNS
        },
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
    if not rows:
        return {"name": name, "games": 0}
    targets: Counter = Counter()
    for row in rows:
        targets.update(row["attach_targets"])
    total = sum(targets.values())
    return {
        "name": name,
        "games": len(rows),
        "manual_attachments_per_game": round(total / len(rows), 3),
        "attach_share": {
            key: round(value / max(1, total), 3)
            for key, value in targets.most_common()
        },
        "munkidori_attachments_per_game": round(
            targets.get("munkidori", 0) / len(rows), 3
        ),
        "munkidori_energised_by_turn": {
            t: round(
                sum(r["munkidori_energised_by_turn"][t] for r in rows)
                / len(rows), 3
            )
            for t in map(str, PROBE_TURNS)
        },
        "munkidori_in_play_by_turn": {
            t: round(
                sum(r["munkidori_in_play_by_turn"][t] for r in rows)
                / len(rows), 3
            )
            for t in map(str, PROBE_TURNS)
        },
        "grimmsnarl_ready_by_turn": {
            t: round(
                sum(r["grimmsnarl_ready_by_turn"][t] for r in rows)
                / len(rows), 3
            )
            for t in map(str, PROBE_TURNS)
        },
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
        "mirror_us_in_wins": block(
            [g["us"] for g in mirrors if g["won"]], "mirror_us_win"
        ),
        "mirror_them_in_our_wins": block(
            [g["them"] for g in mirrors if g["won"]], "mirror_them_l"
        ),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
