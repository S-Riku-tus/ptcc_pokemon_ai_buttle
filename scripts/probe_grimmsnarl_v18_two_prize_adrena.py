"""Replay high-rated mirror prompts through v17's ranker.

The target audit found five teacher decisions where the remaining
Adrena-Brain counters could immediately knock out a two-prize Benched
Grimmsnarl ex while a one-prize target was also legal.  This probe answers the
missing causal question: would v17's deployed ranker already choose the
Grimmsnarl on those exact public boards?
"""

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
from ml_runtime import Ranker  # noqa: E402
from analyze_grimmsnarl_v18_mirror_endgame import (  # noqa: E402
    DEFAULT_RUNS,
    OUR_DECK,
    RunSpec,
    deck_at,
    nested_id,
    selected_indices,
)


def target_rows(
    current: dict[str, Any], select: dict[str, Any]
) -> list[dict[str, Any]]:
    pending = int(select.get("remainDamageCounter") or 0)
    damage = 10 * pending if pending else 30
    rows: list[dict[str, Any]] = []
    for slot, option in enumerate(select.get("option") or []):
        card, owner_is_self, area = mf.resolve_option(current, select, option)
        card = card or {}
        card_id = int(card.get("id", -1))
        hp = int(card.get("hp", 0) or 0)
        rows.append({
            "slot": slot,
            "card_id": card_id,
            "hp": hp,
            "prizes": mf.prize_value(card_id),
            "opponent_bench": not owner_is_self and area == mf.AREA_BENCH,
            "lethal": 0 < hp <= damage,
        })
    return rows


def is_probe(select: dict[str, Any], targets: list[dict[str, Any]]) -> bool:
    if (
        int(select.get("context", -1)) != mf.CTX_DAMAGE_COUNTER
        or nested_id(select.get("effect")) != mf.MUNKIDORI_ID
    ):
        return False
    return any(
        row["opponent_bench"]
        and row["card_id"] == mf.GRIMMSNARL_EX_ID
        and row["lethal"]
        for row in targets
    ) and any(
        row["opponent_bench"] and row["prizes"] == 1
        for row in targets
    )


def probe_run(spec: RunSpec, ranker: Ranker) -> list[dict[str, Any]]:
    if spec.group != "teacher":
        return []
    rows: list[dict[str, Any]] = []
    with (spec.directory / "episodes.csv").open(encoding="utf-8-sig") as handle:
        episodes = list(csv.DictReader(handle))
    for raw in episodes:
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

        saw_shadow = False
        probes: list[tuple[int, dict[str, Any], list[dict[str, Any]], int]] = []
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            options = list(select.get("option") or [])
            chosen = selected_indices(steps, index, seat)
            if not chosen or any(not 0 <= value < len(options) for value in chosen):
                continue

            if int(select.get("context", -1)) == mf.MAIN_CONTEXT:
                if any(
                    mf._int(options[value].get("type")) == 13
                    and mf._int(options[value].get("attackId"))
                    == mf.SHADOW_BULLET_ID
                    for value in chosen
                ):
                    saw_shadow = True

            if saw_shadow and current.get("players"):
                targets = target_rows(current, select)
                if is_probe(select, targets):
                    probes.append((index, observation, targets, chosen[0]))

        # The ranker's history columns reset whenever ``current.turn``
        # changes.  Replaying only the target turn is therefore exact and is
        # roughly two orders of magnitude faster than scoring all 600 games.
        for probe_index, observation, targets, selected in probes:
            probe_turn = int((observation.get("current") or {}).get("turn", -1))
            ranker.reset()
            ranker.teacher_forced = True
            ranker.suspend_escalation = True
            for prior_index in range(probe_index):
                prior_observation = (
                    (steps[prior_index][seat] or {}).get("observation") or {}
                )
                prior_current = prior_observation.get("current") or {}
                if int(prior_current.get("turn", -2)) != probe_turn:
                    continue
                prior_chosen = selected_indices(steps, prior_index, seat)
                prior_options = list(
                    (prior_observation.get("select") or {}).get("option") or []
                )
                if (
                    len(prior_chosen) == 1
                    and 0 <= prior_chosen[0] < len(prior_options)
                    and ranker.is_corpus_scorable(
                        prior_observation.get("select") or {}
                    )
                ):
                    ranker.observe_external(prior_observation, prior_chosen[0])
            predicted = ranker.choose(observation)
            rows.append({
                        "run": spec.name,
                        "episode_id": episode_id,
                        "turn": probe_turn,
                        "actual_slot": selected,
                        "predicted_slot": predicted,
                        "actual": targets[selected],
                        "predicted": (
                            targets[predicted]
                            if predicted is not None and 0 <= predicted < len(targets)
                            else None
                        ),
                        "v17_matches_teacher": predicted == selected,
                        "v17_takes_lethal_grim": bool(
                            predicted is not None
                            and targets[predicted]["card_id"]
                            == mf.GRIMMSNARL_EX_ID
                            and targets[predicted]["lethal"]
                        ),
                        "targets": targets,
                    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    ranker = Ranker()
    rows: list[dict[str, Any]] = []
    for name, group, submission, directory in DEFAULT_RUNS:
        rows.extend(probe_run(
            RunSpec(name, group, submission, ROOT / directory), ranker
        ))
    output = {
        "definition": (
            "post-first-Shadow Adrena-Brain target where a two-prize Benched "
            "Grimmsnarl ex is immediately lethal and a one-prize target exists"
        ),
        "prompts": len(rows),
        "teacher_takes_lethal_grim": sum(
            row["actual"]["card_id"] == mf.GRIMMSNARL_EX_ID
            and row["actual"]["lethal"]
            for row in rows
        ),
        "v17_takes_lethal_grim": sum(
            row["v17_takes_lethal_grim"] for row in rows
        ),
        "v17_matches_teacher": sum(row["v17_matches_teacher"] for row in rows),
        "rows": rows,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
