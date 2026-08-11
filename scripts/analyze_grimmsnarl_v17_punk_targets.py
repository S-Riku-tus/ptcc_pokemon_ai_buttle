"""Measure whether Punk Up can accelerate v16's wall-break route.

v16 can preserve and manually fuel Marnie's Morgrem against damage-immune
Active Pokemon, but the target selection inside Punk Up is a separate select.
This probe walks the 110 v15 ladder games and asks v16 on every stored board.
For wall games it records each Punk Up attachment after the triggering
Grimmsnarl ex is already able to use Shadow Bullet, then asks whether a
Morgrem which can finish the wall route was offered and passed over.

The replay is teacher-forced: v16 answers the stored position, while its
stateful layers advance with the action actually taken in the game.  The
result is a decision-footprint measurement, not an outcome estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_grimmsnarl_v16_footprint import (  # noqa: E402
    CHALLENGER,
    episodes,
    load,
    single,
)

CTX_ATTACH_FROM = 21
GRIMMSNARL_EX_ID = 648
MORGREM_ID = 647
SHADOW_BULLET_COST = 2


def _sides(current: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    players = current.get("players") or [{}, {}]
    seat = int(current.get("yourIndex", 0) or 0)
    return players[seat], players[1 - seat]


def _is_punk_target(select: dict[str, Any]) -> bool:
    effect = select.get("effect")
    return bool(
        int(select.get("context", -1)) == CTX_ATTACH_FROM
        and isinstance(effect, dict)
        and int(effect.get("id", -1)) == GRIMMSNARL_EX_ID
    )


def _option_body(mf: Any, current: dict[str, Any], select: dict[str, Any],
                 slot: int) -> dict[str, Any] | None:
    options = list(select.get("option") or [])
    if not 0 <= slot < len(options):
        return None
    body, is_self, _area = mf.resolve_option(
        current, select, options[slot]
    )
    return body if body is not None and is_self else None


def _trigger_ready(mf: Any, current: dict[str, Any],
                   select: dict[str, Any]) -> bool:
    effect = select.get("effect") or {}
    wanted = effect.get("serial")
    me, _opponent = _sides(current)
    for body in mf._in_play(me):
        if body.get("serial") != wanted:
            continue
        return (
            mf._dark_energy_count(body)
            >= SHADOW_BULLET_COST
        )
    return False


def _viable_morgrem_slots(module: Any, mf: Any, current: dict[str, Any],
                          select: dict[str, Any]) -> list[int]:
    guard = module._WALL_BREAK
    me, opponent = _sides(current)
    their_active = (mf._cards(opponent, "active") or [{}])[0]
    stadium_id = mf._stadium_id(current)
    target = guard._breaker(me, opponent, their_active, stadium_id)
    if target is None or int(target.get("id", -1)) != MORGREM_ID:
        return []
    if mf._dark_energy_count(target) >= 2:
        return []  # already attacks; another Punk Up energy buys no ETA
    serial = target.get("serial")
    return [
        slot for slot in range(len(select.get("option") or []))
        if (_option_body(mf, current, select, slot) or {}).get("serial")
        == serial
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    module = load(
        CHALLENGER,
        {
            "GRIMMSNARL_WALL_BREAK_DISABLE": "0",
            "GRIMMSNARL_ESCALATION_MIRROR": "off",
        },
    )
    mf = sys.modules["ml_features"]
    totals: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for episode_id, replay, seat, matchup in episodes():
        if matchup != "wall":
            continue
        totals["wall_games"] += 1
        module.diag_reset()
        steps = replay.get("steps") or []
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single((steps[index + 1][seat] or {}).get("action"))
            if played is None:
                continue

            if _is_punk_target(select):
                # Scoring all ~1,700 wall decisions through the 2,000-tree
                # model is unnecessary.  The external observer below keeps
                # the turn history exact; ask the model only on this select.
                answer = single(module.agent(observation))
                totals["punk_target_decisions"] += 1
                me, opponent = _sides(current)
                their_active = (
                    mf._cards(opponent, "active") or [{}]
                )[0]
                dead = module._WALL_BREAK._dead_swing(
                    current, me, opponent, their_active
                )
                if dead:
                    totals["under_dead_wall"] += 1
                trigger_ready = _trigger_ready(mf, current, select)
                if dead and trigger_ready:
                    totals["dead_wall_trigger_ready"] += 1
                viable = (
                    _viable_morgrem_slots(module, mf, current, select)
                    if dead and trigger_ready else []
                )
                if viable:
                    totals["viable_morgrem_offered"] += 1
                    if answer in viable:
                        totals["v16_feeds_morgrem"] += 1
                    else:
                        totals["v16_passes_morgrem"] += 1
                    if played in viable:
                        totals["replay_feeds_morgrem"] += 1
                    rows.append({
                        "episode_id": episode_id,
                        "turn": int(current.get("turn", -1)),
                        "step": index,
                        "v16": answer,
                        "played": played,
                        "morgrem_slots": viable,
                        "options": [
                            {
                                "slot": slot,
                                "id": int((body or {}).get("id", -1)),
                                "serial": (body or {}).get("serial"),
                                "dark": (
                                    mf._dark_energy_count(body)
                                    if body is not None else -1
                                ),
                            }
                            for slot in range(len(select.get("option") or []))
                            for body in [
                                _option_body(mf, current, select, slot)
                            ]
                        ],
                    })
            module.observe_external(observation, played)

    output = {"summary": dict(totals), "opportunities": rows}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
