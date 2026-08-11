"""Measure whether Bench-30 turns a benched Grimmsnarl into a one-shot.

Shadow Bullet deals 180 to the Active and 30 to one Benched Pokemon.  A
benched Grimmsnarl ex with 181-210 HP remaining is therefore a special target:
the 30 changes the next Shadow Bullet from a two-hit route into a one-hit,
two-prize route.  This script measures that exact offer/choice rather than a
generic target preference.
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
CTX_DAMAGE = mf.CTX_DAMAGE
OPTION_ATTACK = 13


def in_play_counts(player: dict[str, Any]) -> Counter[int]:
    return Counter(
        int(card.get("id", -1))
        for card in mf._cards(player, "active") + mf._cards(player, "bench")
    )


def target_row(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
) -> dict[str, Any]:
    card, owner_is_self, area = mf.resolve_option(current, select, option)
    card = card or {}
    hp = float(card.get("hp", 0) or 0)
    return {
        "id": int(card.get("id", -1)),
        "hp": hp,
        "max_hp": float(card.get("maxHp", 0) or 0),
        "damage": max(
            0.0,
            float(card.get("maxHp", 0) or 0) - hp,
        ),
        "owner_is_self": owner_is_self,
        "area": area,
        "grim_one_shot_setup": (
            int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf.SHADOW_BULLET_DAMAGE < hp
            <= mf.SHADOW_BULLET_DAMAGE + mf.SHADOW_BULLET_BENCH_DAMAGE
        ),
        "bench_ko_now": 0 < hp <= mf.SHADOW_BULLET_BENCH_DAMAGE,
    }


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
        shadow_count = 0
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
                    shadow_count += 1
                continue
            if (
                context != CTX_DAMAGE
                or nested_id(select.get("effect")) != mf.GRIMMSNARL_EX_ID
                or shadow_count <= 0
            ):
                continue
            targets = [target_row(current, select, option) for option in options]
            chosen_targets = [targets[slot] for slot in chosen]
            setup = [target for target in targets if target["grim_one_shot_setup"]]
            lethal = [target for target in targets if target["bench_ko_now"]]
            players = current.get("players") or [{}, {}]
            me, opponent = players[seat], players[1 - seat]
            ready_munkidori = sum(
                int(
                    int(card.get("id", -1)) == mf.MUNKIDORI_ID
                    and mf._dark_energy_count(card) > 0
                )
                for card in mf._cards(me, "active") + mf._cards(me, "bench")
            )
            movable_counters = sum(
                mf.movable_counters(card)
                for card in mf._cards(me, "active") + mf._cards(me, "bench")
            )
            output.append({
                "run": spec.name,
                "group": spec.group,
                "submission": spec.submission,
                "episode_id": episode_id,
                "won": won,
                "turn": int(current.get("turn", -1)),
                "shadow_number": shadow_count,
                "self_prizes_taken": 6 - len(me.get("prize") or []),
                "opp_prizes_taken": 6 - len(opponent.get("prize") or []),
                "self_deck": int(me.get("deckCount", 0) or 0),
                "self_field": dict(in_play_counts(me)),
                "opp_field": dict(in_play_counts(opponent)),
                "froslass_count": in_play_counts(me)[mf.FROSLASS_ID],
                "munkidori_count": in_play_counts(me)[mf.MUNKIDORI_ID],
                "ready_munkidori_count": ready_munkidori,
                "self_movable_counters": movable_counters,
                "adrena_damage_capacity": 10 * min(
                    movable_counters, 3 * ready_munkidori
                ),
                "targets": targets,
                "chosen_targets": chosen_targets,
                "setup_offered": bool(setup),
                "setup_taken": any(
                    target["grim_one_shot_setup"] for target in chosen_targets
                ),
                "lethal_offered": bool(lethal),
                "lethal_taken": any(
                    target["bench_ko_now"] for target in chosen_targets
                ),
                "setup_hps": [target["hp"] for target in setup],
                "chosen_ids": [target["id"] for target in chosen_targets],
                "chosen_hps": [target["hp"] for target in chosen_targets],
            })
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    setup = [row for row in rows if row["setup_offered"]]
    clean_setup = [row for row in setup if not row["lethal_offered"]]
    later_setup = [row for row in clean_setup if row["shadow_number"] >= 2]
    return {
        "prompts": len(rows),
        "games": len({(row["run"], row["episode_id"]) for row in rows}),
        "record": (
            f"{len({(r['run'], r['episode_id']) for r in rows if r['won']})}-"
            f"{len({(r['run'], r['episode_id']) for r in rows if not r['won']})}"
        ),
        "setup_offered": len(setup),
        "setup_taken": sum(row["setup_taken"] for row in setup),
        "setup_rate": (
            round(sum(row["setup_taken"] for row in setup) / len(setup), 4)
            if setup else None
        ),
        "clean_setup_offered": len(clean_setup),
        "clean_setup_taken": sum(row["setup_taken"] for row in clean_setup),
        "clean_setup_rate": (
            round(
                sum(row["setup_taken"] for row in clean_setup)
                / len(clean_setup),
                4,
            ) if clean_setup else None
        ),
        "later_clean_setup_offered": len(later_setup),
        "later_clean_setup_taken": sum(
            row["setup_taken"] for row in later_setup
        ),
        "lethal_offered": sum(row["lethal_offered"] for row in rows),
        "lethal_taken": sum(row["lethal_taken"] for row in rows),
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
        if row["setup_offered"]
        and not row["setup_taken"]
        and not row["lethal_offered"]
    ]
    output = {
        "definition": (
            "a benched Grimmsnarl ex at 181-210 HP; Bench-30 makes the next "
            "180-damage Shadow Bullet a KO"
        ),
        "blocks": {name: summarize(value) for name, value in blocks.items()},
        "deployed_clean_setup_misses": misses,
        "deployed_prompts_with_grim": [
            row for row in blocks["deployed_all"]
            if any(
                target["id"] == mf.GRIMMSNARL_EX_ID
                for target in row["targets"]
            )
        ],
        "teacher_prompts_with_grim": [
            row for row in blocks["teachers_all"]
            if any(
                target["id"] == mf.GRIMMSNARL_EX_ID
                for target in row["targets"]
            )
        ],
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
