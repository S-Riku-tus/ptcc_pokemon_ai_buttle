"""Footprint of replacing wall_break's breaker *count* with route viability.

``WallBreak._spends_last_breaker`` refuses a Grimmsnarl ex evolution only when
``len(breakers) != 1`` - a raw count of Morgrem/Impidimp bodies in play.  A
second Morgrem with no Energy therefore disarms the guard even though it cannot
reach the wall, which is the shape of episode 92168220: the only fuelled
breaker was evolved away, the survivor could not attack, and the game ended
0 prizes / deck-out.

The proposed change counts only breakers whose route is actually viable, using
the same damage and ETA arithmetic ``WallBreakGuard._breaker`` already applies.
This measures how many stored decisions the two conditions differ on, so the
change is sized before it costs a ladder slot - the v18 lesson, where both
shipped guards bound zero times.

Reported per version over every stored ladder run:

* ``wall_up``            - our own MAIN decisions taken while Shadow Bullet is
  provably worth zero against their Active;
* ``current_binds``      - decisions the shipped count-based guard refuses;
* ``proposed_binds``     - decisions the viability-based guard refuses;
* ``new_binds``          - proposed and not current, i.e. what the fix buys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v20"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT))

import ml_features as mf  # noqa: E402
import wall_break as wb  # noqa: E402

OPTION_EVOLVE = 9
MAX_ROUTE_TURNS = getattr(wb, "MAX_ROUTE_TURNS", 6)


def _stadium_id(current: dict[str, Any]) -> int:
    stadium = current.get("stadium") or []
    return int(stadium[0].get("id", -1)) if stadium else -1


def _eta(
    body: dict[str, Any],
    their_active: dict[str, Any],
    stadium_id: int,
    deck_left: int,
) -> float:
    """Own turns for this breaker to knock the wall out, or ``inf``.

    Same arithmetic as ``WallBreakGuard._breaker``: real damage after the
    blocker rules, plus one turn per missing Darkness because only one manual
    attachment is available per turn and Punk Up cannot fire without consuming
    a breaker.
    """
    damage, need, _attack = wb.WallBreakGuard._damage(
        body, their_active, stadium_id
    )
    hp = float(their_active.get("hp", 0) or 0)
    if damage <= 0.0 or hp <= 0.0:
        return math.inf
    remaining = hp - float(body.get("damage") or 0.0)
    fuel = max(0, need - mf._dark_energy_count(body))
    eta = fuel + int(math.ceil(max(remaining, 1.0) / damage))
    if eta > MAX_ROUTE_TURNS or eta >= max(1, deck_left):
        return math.inf
    return float(eta)


def scan(replay: dict[str, Any], seat: int) -> Counter:
    counts: Counter = Counter()
    steps = replay.get("steps") or []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
            continue
        options = list(select.get("option") or [])
        current = observation.get("current") or {}
        players = current.get("players") or []
        if not options or len(players) < 2:
            continue
        me, opponent = players[seat], players[1 - seat]
        their_active = (mf._cards(opponent, "active") or [None])[0]
        if their_active is None:
            continue
        stadium_id = _stadium_id(current)

        # "Wall up" is the shipped dead-swing test: Shadow Bullet deals zero.
        if mf.shadow_damage_to(their_active, stadium_id) > 0.0:
            continue
        counts["wall_up_decisions"] += 1

        raw = (steps[index + 1][seat] or {}).get("action")
        picked = {
            int(value) for value in raw
            if isinstance(value, int) and 0 <= int(value) < len(options)
        } if isinstance(raw, list) else set()

        in_play = mf._in_play(me)
        ready = any(
            int(body.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf._dark_energy_count(body) >= 2
            for body in in_play
        )
        breakers = [
            body for body in in_play
            if int(body.get("id", -1)) in wb.BREAKER_IDS
        ]
        deck_left = int(me.get("deckCount", 0) or 0)
        etas = {
            body.get("serial"): _eta(
                body, their_active, stadium_id, deck_left
            )
            for body in breakers
        }
        viable = [
            body for body in breakers
            if etas[body.get("serial")] < math.inf
        ]

        for position in picked:
            option = options[position]
            if mf._int(option.get("type")) != OPTION_EVOLVE:
                continue
            card = mf.candidate_card(current, option, select) or {}
            if int(card.get("id", -1)) != mf.GRIMMSNARL_EX_ID:
                continue
            area = mf._int(option.get("inPlayArea"))
            slot = mf._int(option.get("inPlayIndex"))
            pool = mf._cards(
                me, "active" if area == mf.AREA_ACTIVE else "bench"
            )
            if not 0 <= slot < len(pool):
                continue
            target = pool[slot]
            counts["evolutions_under_wall"] += 1
            if not ready:
                counts["no_ready_grimmsnarl"] += 1
                continue
            on_breaker = any(
                target.get("serial") == body.get("serial")
                for body in breakers
            )
            if not on_breaker:
                continue
            current_binds = (
                len(breakers) == 1
                and target.get("serial") == breakers[0].get("serial")
            )
            on_viable = any(
                target.get("serial") == body.get("serial")
                for body in viable
            )
            proposed_binds = on_viable and len(viable) == 1
            # Variant B: never consume the *fastest* route to the wall while a
            # ready Grimmsnarl ex already exists.  92168220 turn 13 is exactly
            # this - two Morgrem in play, only one of them fuelled - so the
            # count-based and the viability-count rules both miss it.
            own_eta = etas.get(target.get("serial"), math.inf)
            others = [
                etas[body.get("serial")] for body in breakers
                if body.get("serial") != target.get("serial")
            ]
            best_binds = own_eta < math.inf and own_eta < min(
                others, default=math.inf
            )
            counts["current_binds"] += int(current_binds)
            counts["proposed_binds"] += int(proposed_binds)
            counts["best_route_binds"] += int(best_binds)
            if proposed_binds and not current_binds:
                counts["new_binds"] += 1
                counts[f"new_binds_breakers_{len(breakers)}"] += 1
            if best_binds and not current_binds:
                counts["best_route_new_binds"] += 1
            if current_binds and not proposed_binds:
                counts["lost_binds"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20"
        / "ladder_history_games.csv",
    )
    parser.add_argument(
        "--runs", type=Path, default=ROOT / "data" / "runs" / "grimmsnarl"
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20"
        / "breaker_viability.json",
    )
    args = parser.parse_args()

    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(args.runs.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (
                    run_dir, int(row["detected_submission_agent_index"])
                )

    per_version: dict[str, Counter] = defaultdict(Counter)
    affected: dict[str, list[str]] = defaultdict(list)
    for row in csv.DictReader(args.games.open(encoding="utf-8-sig")):
        entry = index.get(row["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (
            run_dir / "episodes" / row["episode_id"] / "replay"
            / f"episode_{row['episode_id']}.json"
        )
        if not path.exists():
            continue
        counts = scan(json.loads(path.read_text(encoding="utf-8")), seat)
        if not counts:
            continue
        per_version[row["version"]] += counts
        per_version["ALL"] += counts
        if counts["best_route_new_binds"]:
            affected[row["version"]].append(
                f"{row['episode_id']} {row['opponent_family']} "
                f"won={row['won']}"
            )

    payload = {
        "per_version": {
            version: dict(counts.most_common())
            for version, counts in sorted(per_version.items())
        },
        "episodes_with_new_binds": dict(affected),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    header = (
        f"{'version':9s} {'wallDecisions':>13s} {'evolvesUnderWall':>16s} "
        f"{'current':>7s} {'proposed':>8s} {'new':>4s} {'lost':>4s} "
        f"{'bestRoute':>9s} {'brNew':>5s}"
    )
    print(header)
    for version, counts in sorted(per_version.items()):
        print(
            f"{version:9s} {counts['wall_up_decisions']:13d} "
            f"{counts['evolutions_under_wall']:16d} "
            f"{counts['current_binds']:7d} {counts['proposed_binds']:8d} "
            f"{counts['new_binds']:4d} {counts['lost_binds']:4d} "
            f"{counts['best_route_binds']:9d} "
            f"{counts['best_route_new_binds']:5d}"
        )
    print()
    for version, episodes in sorted(affected.items()):
        for line in episodes:
            print(f"  {version}: {line}")
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
