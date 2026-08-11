"""What v16 does that v15 does not, on v15's own 110 rated boards.

Two questions, both answered against stored ladder games rather than a fresh
arena, because the arena cannot be paired: ``--seed`` does not seed the native
shuffle, so two runs of the same agent are two different samples.

1. **Counterfactual.** With ``GRIMMSNARL_WALL_BREAK_DISABLE=1`` and
   ``GRIMMSNARL_ESCALATION_MIRROR=on``, v16 must answer every stored board
   exactly as v15 does.  Any difference there is an unintended change.
2. **Footprint.** With both live, every difference is the version, listed by
   the shape of the action it replaces and by matchup - so the claim "wall
   games and mirrors only" is a measurement rather than an intention.

Both replays are teacher-forced: the agent answers the stored board, but its
intra-turn history advances with the action the game actually played, so one
divergence cannot cascade through every later feature column.

    python scripts/probe_grimmsnarl_v16_footprint.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402
from analyze_grimmsnarl_v16_prize_conversion import (  # noqa: E402
    deck_label,
    matchup_of,
)

CHAMPION = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v15"
CHALLENGER = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v16"
RUNS = (
    ("20260810_grimmsnarl_ml_v15_sub55404196", "55404196"),
    ("20260811_grimmsnarl_ml_v15_b_sub55409394", "55409394"),
)
AGENT_MODULES = (
    "main", "ml_runtime", "ml_features", "fallback_policy", "ml_planner",
    "ml_residual", "policy_router", "matchup_guard", "attack_access",
    "policy_base", "wall_break",
)
SHADOW_BULLET_ID = 937
ATTACK_OPTION = 13


def load(agent_dir: Path, env: dict[str, str]) -> Any:
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    for name in AGENT_MODULES:
        sys.modules.pop(name, None)
    for name in list(sys.path):
        if name.endswith("grimmsnarl_ml_v15") or name.endswith(
            "grimmsnarl_ml_v16"
        ):
            sys.path.remove(name)
    try:
        module = load_dir_agent_module(agent_dir)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if getattr(module, "_RANKER", None) is not None:
        module._RANKER.teacher_forced = True
    return module


def episodes() -> list[tuple[int, dict[str, Any], int, str]]:
    out = []
    for run, submission in RUNS:
        run_dir = ROOT / "data/runs/grimmsnarl" / run
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
            out.append(
                (episode_id, replay, seat,
                 matchup_of(deck_label(decks[1 - seat])))
            )
    return out


def shape(observation: dict[str, Any], index: int) -> str:
    select = observation.get("select") or {}
    options = select.get("option") or []
    context = int(select.get("context", -1))
    if not 0 <= index < len(options):
        return f"ctx{context}:invalid"
    option = options[index]
    kind = option.get("type")
    if kind == ATTACK_OPTION:
        attack_id = int(option.get("attackId", -1))
        name = "shadow" if attack_id == SHADOW_BULLET_ID else str(attack_id)
        return f"ctx{context}:attack:{name}"
    return f"ctx{context}:type{kind}"


def single(action: Any) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
    ):
        return action[0]
    return None


def compare(games, champion, challenger) -> dict[str, Any]:
    totals = {
        "games": 0,
        "decisions": 0,
        "differences": 0,
        "by_shape": Counter(),
        "by_matchup": defaultdict(lambda: {"decisions": 0, "differences": 0}),
        "episodes_touched": [],
    }
    guard: Counter = Counter()
    for episode_id, replay, seat, matchup in games:
        champion.diag_reset()
        challenger.diag_reset()
        steps = replay["steps"]
        touched = 0
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            if not isinstance(observation, dict):
                continue
            if observation.get("select") is None:
                continue
            current = observation.get("current")
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single((steps[index + 1][seat] or {}).get("action"))
            if played is None:
                continue
            left = single(champion.agent(observation))
            right = single(challenger.agent(observation))
            totals["decisions"] += 1
            totals["by_matchup"][matchup]["decisions"] += 1
            if left is not None and right is not None and left != right:
                totals["differences"] += 1
                totals["by_matchup"][matchup]["differences"] += 1
                totals["by_shape"][
                    f"{shape(observation, left)}"
                    f" -> {shape(observation, right)}"
                ] += 1
                touched += 1
            champion.observe_external(observation, played)
            challenger.observe_external(observation, played)
        totals["games"] += 1
        guard.update(challenger.diag_snapshot().get("wall_break", {}))
        if touched:
            totals["episodes_touched"].append(
                {"episode_id": episode_id, "matchup": matchup,
                 "differences": touched}
            )
    totals["by_shape"] = dict(totals["by_shape"].most_common())
    totals["by_matchup"] = {
        key: value for key, value in sorted(totals["by_matchup"].items())
    }
    totals["episodes_touched"].sort(key=lambda row: -row["differences"])
    # Accumulated over every episode: diag_reset runs per game, so the last
    # snapshot alone would describe one game and read as an inert guard.
    totals["guard"] = dict(guard)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--matchups",
        help="comma separated subset, e.g. wall,mirror",
    )
    parser.add_argument(
        "--skip-counterfactual",
        action="store_true",
        help="only measure the footprint; the switches-off pass is slow",
    )
    args = parser.parse_args()

    games = episodes()
    if args.matchups:
        wanted = {name.strip() for name in args.matchups.split(",")}
        games = [row for row in games if row[3] in wanted]
    if args.limit:
        games = games[: args.limit]

    counterfactual = {"skipped": True}
    if not args.skip_counterfactual:
        champion = load(CHAMPION, {})
        off = load(
            CHALLENGER,
            {
                "GRIMMSNARL_WALL_BREAK_DISABLE": "1",
                "GRIMMSNARL_ESCALATION_MIRROR": "on",
            },
        )
        counterfactual = compare(games, champion, off)

    champion = load(CHAMPION, {})
    on = load(
        CHALLENGER,
        {
            "GRIMMSNARL_WALL_BREAK_DISABLE": "0",
            "GRIMMSNARL_ESCALATION_MIRROR": "off",
        },
    )
    footprint = compare(games, champion, on)

    out = {
        "counterfactual_v16_disabled": counterfactual,
        "footprint_v16_live": footprint,
        "verdict": (
            "not run"
            if counterfactual.get("skipped")
            else "clean" if counterfactual["differences"] == 0
            else "UNINTENDED CHANGE with both switches off"
        ),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if counterfactual.get("skipped"):
        return 0
    return 0 if counterfactual["differences"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
