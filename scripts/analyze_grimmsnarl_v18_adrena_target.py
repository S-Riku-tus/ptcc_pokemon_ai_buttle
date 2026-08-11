"""Audit two-prize Adrena-Brain targets after first Shadow in mirrors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
CTX_TARGET = mf.CTX_DAMAGE_COUNTER
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
                or context != CTX_TARGET
                or nested_id(select.get("effect")) != mf.MUNKIDORI_ID
            ):
                continue
            targets: list[dict[str, Any]] = []
            pending = int(select.get("remainDamageCounter") or 0)
            swing = 10 * pending if pending else 30
            for option in options:
                card, owner_is_self, area = mf.resolve_option(
                    current, select, option
                )
                card = card or {}
                hp = float(card.get("hp", 0) or 0)
                targets.append({
                    "id": int(card.get("id", -1)),
                    "hp": hp,
                    "owner_is_self": owner_is_self,
                    "area": area,
                    "prizes": mf.prize_value(int(card.get("id", -1))),
                    "dies_to_remaining": 0 < hp <= swing,
                })
            grim_slots = [
                slot for slot, target in enumerate(targets)
                if not target["owner_is_self"]
                and target["area"] == mf.AREA_BENCH
                and target["id"] == mf.GRIMMSNARL_EX_ID
            ]
            one_prize_slots = [
                slot for slot, target in enumerate(targets)
                if not target["owner_is_self"]
                and target["area"] == mf.AREA_BENCH
                and target["prizes"] == 1
            ]
            if not grim_slots or not one_prize_slots:
                continue
            grim_lethal = [
                slot for slot in grim_slots if targets[slot]["dies_to_remaining"]
            ]
            any_lethal = [
                slot for slot, target in enumerate(targets)
                if target["dies_to_remaining"]
            ]
            players = current.get("players") or [{}, {}]
            me = players[seat]
            state = mf.state_features(current)
            output.append({
                "run": spec.name,
                "group": spec.group,
                "episode_id": episode_id,
                "won": won,
                "turn": int(current.get("turn", -1)),
                "pending_counters": pending,
                "self_prizes_taken": 6 - len(me.get("prize") or []),
                "opp_prizes_taken": 6 - len(
                    players[1 - seat].get("prize") or []
                ),
                "munkidori_ready_count": state["munkidori_ready_count"],
                "self_board_movable_counters": state[
                    "self_board_movable_counters"
                ],
                "froslass_count": state["froslass_engine_count"],
                "grim_hps": [targets[slot]["hp"] for slot in grim_slots],
                "one_prize_hps": [
                    targets[slot]["hp"] for slot in one_prize_slots
                ],
                "grim_taken": any(slot in grim_slots for slot in chosen),
                "one_prize_taken": any(
                    slot in one_prize_slots for slot in chosen
                ),
                "grim_lethal_offered": bool(grim_lethal),
                "grim_lethal_taken": any(
                    slot in grim_lethal for slot in chosen
                ),
                "any_lethal_offered": bool(any_lethal),
                "chosen": [targets[slot] for slot in chosen],
            })
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lethal = [row for row in rows if row["grim_lethal_offered"]]
    no_other_lethal = [
        row for row in lethal
        if not any(
            target["dies_to_remaining"] and target["prizes"] == 1
            for target in row["chosen"]
        )
    ]
    return {
        "grim_and_one_prize_offered": len(rows),
        "grim_taken": sum(row["grim_taken"] for row in rows),
        "grim_rate": (
            round(sum(row["grim_taken"] for row in rows) / len(rows), 4)
            if rows else None
        ),
        "lethal_grim_offered": len(lethal),
        "lethal_grim_taken": sum(row["grim_lethal_taken"] for row in lethal),
        "lethal_grim_rate": (
            round(
                sum(row["grim_lethal_taken"] for row in lethal) / len(lethal),
                4,
            ) if lethal else None
        ),
        "diagnostic_no_other_lethal_rows": len(no_other_lethal),
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
        if row["grim_lethal_offered"] and not row["grim_lethal_taken"]
    ]
    output = {
        "definition": (
            "Adrena-Brain can finish a two-prize Benched Grimmsnarl with its "
            "remaining counters while a one-prize target is also offered"
        ),
        "blocks": {name: summarize(value) for name, value in blocks.items()},
        "deployed_lethal_grim_misses": misses,
        "deployed_prompts": blocks["deployed_all"],
        "teacher_prompts": blocks["teachers_all"],
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
