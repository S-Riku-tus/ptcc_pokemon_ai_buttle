"""Teacher-forced v19/v20 footprint on the submitted v16-v19 boards.

The v20-on/v20-off comparison isolates the two-turn horizon tie-breaker.  The
v19/v20-off comparison measures the retrained ranker and new attack-continuity
features.  Stored actions advance each runtime's history, so one changed answer
does not create a synthetic future trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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

RUNS = (
    ("20260811_grimmsnarl_ml_v16_sub55422280", "55422280"),
    ("20260811_grimmsnarl_ml_v17_sub55423572", "55423572"),
    ("20260811_grimmsnarl_ml_v18_sub55428191", "55428191"),
    ("20260811_grimmsnarl_ml_v19_sub55428196", "55428196"),
)
MODULES = (
    "main", "ml_runtime", "ml_features", "fallback_policy", "ml_planner",
    "ml_residual", "policy_router", "matchup_guard", "attack_access",
    "policy_base", "wall_break", "mirror_prize", "horizon_prize",
)


def load(agent_dir: Path, env: dict[str, str]) -> Any:
    tracked = set(env) | {
        "GRIMMSNARL_HORIZON_PRIZE_DISABLE",
        "GRIMMSNARL_HORIZON_SCORE_TOLERANCE",
    }
    previous = {key: os.environ.get(key) for key in tracked}
    for key in tracked:
        os.environ.pop(key, None)
    os.environ.update(env)
    for name in MODULES:
        sys.modules.pop(name, None)
    for name in list(sys.path):
        if "grimmsnarl_ml_v19" in name or "grimmsnarl_ml_v20" in name:
            sys.path.remove(name)
    try:
        module = load_dir_agent_module(agent_dir)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def single(action: Any) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
    ):
        return action[0]
    return None


def shape(observation: dict[str, Any], index: int | None) -> str:
    select = observation.get("select") or {}
    options = list(select.get("option") or [])
    context = int(select.get("context", -1))
    if index is None or not 0 <= index < len(options):
        return f"ctx{context}:multi"
    option = options[index]
    return (
        f"ctx{context}:type{option.get('type')}:"
        f"card{option.get('index', -1)}:attack{option.get('attackId', -1)}"
    )


def games(
    wanted: set[str], wanted_runs: set[str] | None = None
) -> list[tuple[int, dict[str, Any], int, str]]:
    found = []
    for run, submission in RUNS:
        if wanted_runs and run not in wanted_runs:
            continue
        run_dir = ROOT / "data/runs/grimmsnarl" / run
        with (run_dir / "episodes.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("state") != "COMPLETED":
                continue
            a0 = row.get("agent_0_submission_id")
            a1 = row.get("agent_1_submission_id")
            if a0 == a1 or submission not in (a0, a1):
                continue
            seat = 0 if a0 == submission else 1
            episode_id = int(row["episode_id"])
            replay_path = (
                run_dir / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not replay_path.exists():
                continue
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            steps = replay.get("steps") or []
            decks: list[list[int] | None] = [None, None]
            if len(steps) > 1:
                for side in (0, 1):
                    action = (steps[1][side] or {}).get("action")
                    if isinstance(action, list) and len(action) == 60:
                        decks[side] = [int(value) for value in action]
            matchup = matchup_of(deck_label(decks[1 - seat]))
            if wanted and matchup not in wanted:
                continue
            found.append((episode_id, replay, seat, matchup))
    return found


def compare_fast(
    selected: list[tuple[int, dict[str, Any], int, str]],
    v20: Any,
) -> dict[str, Any]:
    """One-model audit against the submitted action stored on each board."""
    totals: Counter = Counter()
    by_matchup: dict[str, Counter] = defaultdict(Counter)
    changes: list[dict[str, Any]] = []
    diag = Counter()
    started = time.perf_counter()
    for episode_id, replay, seat, matchup in selected:
        _reset(v20)
        touched = False
        totals["games"] += 1
        by_matchup[matchup]["games"] += 1
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            if not isinstance(observation, dict) or observation.get("select") is None:
                continue
            current = observation.get("current")
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single((steps[step_index + 1][seat] or {}).get("action"))
            if played is None:
                continue
            answer = single(v20.agent(observation))
            v20.observe_external(observation, played)
            totals["decisions"] += 1
            by_matchup[matchup]["decisions"] += 1
            if answer != played:
                totals["differences_from_submitted"] += 1
                by_matchup[matchup]["differences_from_submitted"] += 1
                touched = True
                select = observation.get("select") or {}
                changes.append({
                    "episode_id": episode_id,
                    "matchup": matchup,
                    "step": step_index,
                    "turn": int(current.get("turn", -1)),
                    "context": int(select.get("context", -1)),
                    "played": played,
                    "v20": answer,
                    "played_shape": shape(observation, played),
                    "v20_shape": shape(observation, answer),
                })
        totals["games_touched"] += int(touched)
        by_matchup[matchup]["games_touched"] += int(touched)
        _sum_diag(diag, v20)
    return {
        "scope": {
            "mode": "fast-v20-vs-submitted",
            "matchups": sorted({matchup for *_, matchup in selected}),
            "games": totals["games"],
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "teacher_forced": True,
        },
        "totals": dict(totals),
        "by_matchup": {
            key: dict(value) for key, value in sorted(by_matchup.items())
        },
        "diagnostics": {"v20": dict(diag)},
        "changes": changes,
    }


def _reset(module: Any) -> None:
    module.diag_reset()
    ranker = getattr(module, "_RANKER", None)
    if ranker is not None:
        ranker.teacher_forced = True


def _sum_diag(total: Counter, module: Any) -> None:
    snapshot = module.diag_snapshot()
    for section, values in snapshot.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, (int, float)):
                total[f"{section}.{key}"] += value


def compare(
    selected: list[tuple[int, dict[str, Any], int, str]],
    v19: Any,
    v20_off: Any,
    v20_on: Any,
) -> dict[str, Any]:
    totals: Counter = Counter()
    by_matchup: dict[str, Counter] = defaultdict(Counter)
    changes: list[dict[str, Any]] = []
    diag = {"v19": Counter(), "v20_off": Counter(), "v20_on": Counter()}
    started = time.perf_counter()
    for episode_id, replay, seat, matchup in selected:
        for module in (v19, v20_off, v20_on):
            _reset(module)
        totals["games"] += 1
        by_matchup[matchup]["games"] += 1
        touched_model = touched_horizon = False
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            if not isinstance(observation, dict) or observation.get("select") is None:
                continue
            current = observation.get("current")
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single((steps[step_index + 1][seat] or {}).get("action"))
            if played is None:
                continue
            answers = [
                single(module.agent(observation))
                for module in (v19, v20_off, v20_on)
            ]
            for module in (v19, v20_off, v20_on):
                module.observe_external(observation, played)
            totals["decisions"] += 1
            by_matchup[matchup]["decisions"] += 1
            if answers[0] != answers[1]:
                totals["model_differences"] += 1
                by_matchup[matchup]["model_differences"] += 1
                touched_model = True
            if answers[1] != answers[2]:
                totals["horizon_differences"] += 1
                by_matchup[matchup]["horizon_differences"] += 1
                touched_horizon = True
            if answers[0] != answers[1] or answers[1] != answers[2]:
                select = observation.get("select") or {}
                changes.append({
                    "episode_id": episode_id,
                    "matchup": matchup,
                    "step": step_index,
                    "turn": int(current.get("turn", -1)),
                    "context": int(select.get("context", -1)),
                    "played": played,
                    "v19": answers[0],
                    "v20_horizon_off": answers[1],
                    "v20_horizon_on": answers[2],
                    "played_shape": shape(observation, played),
                    "v19_shape": shape(observation, answers[0]),
                    "v20_shape": shape(observation, answers[2]),
                })
        totals["model_games_touched"] += int(touched_model)
        totals["horizon_games_touched"] += int(touched_horizon)
        by_matchup[matchup]["model_games_touched"] += int(touched_model)
        by_matchup[matchup]["horizon_games_touched"] += int(touched_horizon)
        for name, module in zip(
            ("v19", "v20_off", "v20_on"), (v19, v20_off, v20_on)
        ):
            _sum_diag(diag[name], module)
    return {
        "scope": {
            "matchups": sorted({matchup for *_, matchup in selected}),
            "games": totals["games"],
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "teacher_forced": True,
        },
        "totals": dict(totals),
        "by_matchup": {
            key: dict(value) for key, value in sorted(by_matchup.items())
        },
        "diagnostics": {
            name: dict(values) for name, values in diag.items()
        },
        "changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matchups", default="mirror,alakazam",
        help="Comma-separated public router labels; empty means all.",
    )
    parser.add_argument(
        "--runs", default="",
        help="Comma-separated run directory names; empty uses v16-v19.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Score v20 once and compare it with each stored submitted action.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    wanted = {value.strip() for value in args.matchups.split(",") if value.strip()}
    wanted_runs = {value.strip() for value in args.runs.split(",") if value.strip()}
    selected = games(wanted, wanted_runs or None)
    if args.fast:
        v20 = load(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v20", {})
        report = compare_fast(selected, v20)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "scope": report["scope"],
            "totals": report["totals"],
            "by_matchup": report["by_matchup"],
            "horizon": {
                key: value for key, value in
                report["diagnostics"]["v20"].items()
                if key.startswith("horizon_prize.")
            },
        }, ensure_ascii=False, indent=2))
        return 0
    v19 = load(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v19", {})
    v20_off = load(
        ROOT / "agents/grimmsnarl/grimmsnarl_ml_v20",
        {"GRIMMSNARL_HORIZON_PRIZE_DISABLE": "1"},
    )
    v20_on = load(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v20", {})
    report = compare(selected, v19, v20_off, v20_on)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "scope": report["scope"],
        "totals": report["totals"],
        "by_matchup": report["by_matchup"],
        "horizon": {
            key: value for key, value in
            report["diagnostics"]["v20_on"].items()
            if key.startswith("horizon_prize.")
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
