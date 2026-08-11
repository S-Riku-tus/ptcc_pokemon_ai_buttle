"""Audit Adrena-Brain source choice after first Shadow in exact mirrors.

The aggregate gap is not ability uptake: v15 uses every offered Adrena-Brain.
This probe asks a narrower question.  When the Active Grimmsnarl cannot be
saved from the opponent's next attack even by all available heals, does the
policy still remove counters from it while a damaged Benched Munkidori could
be healed instead?
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
AGENT = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v17"
for path in (ROOT, ROOT / "scripts", AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_v18_mirror_endgame import (  # noqa: E402
    DEFAULT_RUNS,
    OUR_DECK,
    RunSpec,
    deck_at,
    nested_id,
    selected_indices,
)

MAIN = 0
CTX_SOURCE = mf.CTX_REMOVE_DAMAGE_COUNTER
OPTION_ATTACK = 13


def load_rows(spec: RunSpec) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    path = spec.directory / "episodes.csv"
    if not path.exists():
        return output
    for raw in csv.DictReader(path.open(encoding="utf-8-sig")):
        if raw.get("state") != "COMPLETED":
            continue
        a0 = raw.get("agent_0_submission_id", "")
        a1 = raw.get("agent_1_submission_id", "")
        if spec.submission not in (a0, a1) or a0 == a1:
            continue
        seat = 0 if a0 == spec.submission else 1
        episode_id = int(raw["episode_id"])
        replay_path = (
            spec.directory / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not replay_path.exists():
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        if deck_at(steps, seat) != OUR_DECK or deck_at(steps, 1 - seat) != OUR_DECK:
            continue
        rewards = replay.get("rewards") or [0, 0]
        won = rewards[seat] > rewards[1 - seat]
        saw_shadow = False
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            options = list(select.get("option") or [])
            chosen = selected_indices(steps, index, seat)
            if not current.get("players") or not chosen:
                continue
            if any(not 0 <= slot < len(options) for slot in chosen):
                continue
            context = int(select.get("context", -1))
            if context == MAIN:
                if any(
                    mf._int(options[slot].get("type")) == OPTION_ATTACK
                    and mf._int(options[slot].get("attackId"))
                    == mf.SHADOW_BULLET_ID
                    for slot in chosen
                ):
                    saw_shadow = True
                continue
            if (
                not saw_shadow
                or context != CTX_SOURCE
                or nested_id(select.get("effect")) != mf.MUNKIDORI_ID
            ):
                continue
            resolved = [
                mf.resolve_option(current, select, option)
                for option in options
            ]
            cards = [card or {} for card, _, _ in resolved]
            players = current.get("players") or [{}, {}]
            me = players[seat]
            active = (mf._cards(me, "active") or [{}])[0]
            active_slots = [
                slot for slot, (_, owner, area) in enumerate(resolved)
                if owner
                and area == mf.AREA_ACTIVE
                and int(cards[slot].get("id", -1)) == mf.GRIMMSNARL_EX_ID
            ]
            munk_slots = [
                slot for slot, (_, owner, area) in enumerate(resolved)
                if owner
                and area == mf.AREA_BENCH
                and int(cards[slot].get("id", -1)) == mf.MUNKIDORI_ID
                and mf.movable_counters(cards[slot]) > 0
            ]
            if not active_slots or not munk_slots:
                continue
            state = mf.state_features(current)
            chosen_active = any(slot in active_slots for slot in chosen)
            chosen_munk = any(slot in munk_slots for slot in chosen)
            output.append({
                "run": spec.name,
                "group": spec.group,
                "submission": spec.submission,
                "episode_id": episode_id,
                "won": won,
                "turn": int(current.get("turn", -1)),
                "self_prizes_taken": 6 - len(me.get("prize") or []),
                "active_hp": float(active.get("hp", 0) or 0),
                "active_damage": max(
                    0.0,
                    float(active.get("maxHp", 0) or 0)
                    - float(active.get("hp", 0) or 0),
                ),
                "active_threat": state["opp_board_threat_damage"],
                "heals_needed": state["self_active_heals_needed"],
                "heals_available": state["self_active_heals_available"],
                "active_savable": bool(state["self_active_savable_by_heals"]),
                "active_unsavable": bool(state["self_active_unsavable"]),
                "active_movable": state["self_active_movable_counters"],
                "munk_movable": [
                    mf.movable_counters(cards[slot]) for slot in munk_slots
                ],
                "chosen_active": chosen_active,
                "chosen_munk": chosen_munk,
            })
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unsavable = [row for row in rows if row["active_unsavable"]]
    savable = [row for row in rows if row["active_savable"]]

    def choice(block: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "offers": len(block),
            "active_taken": sum(row["chosen_active"] for row in block),
            "munk_taken": sum(row["chosen_munk"] for row in block),
            "active_rate": (
                round(sum(row["chosen_active"] for row in block) / len(block), 4)
                if block else None
            ),
        }

    return {
        "both_offered": len(rows),
        "all": choice(rows),
        "active_unsavable": choice(unsavable),
        "active_savable": choice(savable),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for name, group, submission, directory in DEFAULT_RUNS:
        rows.extend(load_rows(RunSpec(name, group, submission, ROOT / directory)))

    blocks: dict[str, list[dict[str, Any]]] = {
        "deployed_all": [row for row in rows if row["group"] == "deployed"],
        "deployed_wins": [
            row for row in rows if row["group"] == "deployed" and row["won"]
        ],
        "deployed_losses": [
            row for row in rows if row["group"] == "deployed" and not row["won"]
        ],
        "teachers_all": [row for row in rows if row["group"] == "teacher"],
        "teachers_wins": [
            row for row in rows if row["group"] == "teacher" and row["won"]
        ],
        "teachers_losses": [
            row for row in rows if row["group"] == "teacher" and not row["won"]
        ],
    }
    for run in sorted({row["run"] for row in rows}):
        blocks[run] = [row for row in rows if row["run"] == run]
    misses = [
        row for row in blocks["deployed_all"]
        if row["active_unsavable"] and row["chosen_active"]
    ]
    output = {
        "definition": (
            "both damaged Active Grimmsnarl and damaged Benched Munkidori "
            "are legal Adrena-Brain sources after first Shadow"
        ),
        "blocks": {name: summarize(value) for name, value in blocks.items()},
        "deployed_unsavable_active_choices": misses,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
