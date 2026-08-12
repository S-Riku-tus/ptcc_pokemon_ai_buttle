"""Legal-input footprint of the v21 wall-breaker guard against v20's.

Replaying only the stored index cannot separate a binding guard from a
saturated invariant, so every legal single-pick index is fed to both versions
of ``WallBreakGuard`` on the same public board, with persistent state advanced
by the action actually played.

Two numbers matter, and the v18 lesson is why both are reported: ``moved``
counts counterfactual legal inputs the guard would correct, and
``actual_overrides`` counts the stored decisions it would really have changed.
A guard with a large sweep and zero actual overrides is inert.

    python scripts/sweep_grimmsnarl_v21_wall_break.py --report out.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V21 = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21"
V20 = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v20"
for path in (V21, ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402


def load_guard(agent_dir: Path, alias: str):
    """Load one agent's ``wall_break`` under its own module name.

    Both versions import the identical ``ml_features``, which is already in
    ``sys.modules``, so only the guard itself is duplicated.
    """
    spec = importlib.util.spec_from_file_location(
        alias, agent_dir / "wall_break.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


WB20 = load_guard(V20, "wall_break_v20")
WB21 = load_guard(V21, "wall_break_v21")


def single(action: Any) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
    ):
        return action[0]
    return None


def sweep(guard: Any, observation: dict, select: dict, played: int) -> list[int]:
    options = list(select.get("option") or [])
    outputs = []
    for incoming in range(len(options)):
        probe = copy.deepcopy(guard)
        outputs.append(int(probe.adjust(observation, select, incoming, [played])))
    return outputs


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
        "--versions", default="",
        help="Comma-separated version labels; default is every stored run.",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v21"
        / "wall_break_legal_sweep.json",
    )
    args = parser.parse_args()
    wanted = {v for v in args.versions.split(",") if v}

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

    totals = {name: Counter() for name in ("v20", "v21")}
    by_matchup = defaultdict(lambda: {"v20": Counter(), "v21": Counter()})
    examples: list[dict[str, Any]] = []
    stats_total = {"v20": Counter(), "v21": Counter()}
    games = 0

    for row in csv.DictReader(args.games.open(encoding="utf-8-sig")):
        if wanted and row["version"] not in wanted:
            continue
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
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        guards = {"v20": WB20.WallBreakGuard(), "v21": WB21.WallBreakGuard()}
        games += 1

        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current")
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single(
                (steps[step_index + 1][seat] or {}).get("action")
            )
            options = list(select.get("option") or [])
            if played is None or not options:
                continue
            for guard in guards.values():
                guard.note(observation)
            if int(select.get("maxCount") or 0) != 1:
                continue

            outputs = {
                name: sweep(guard, observation, select, played)
                for name, guard in guards.items()
            }
            for name, values in outputs.items():
                moved = sum(
                    int(out != incoming)
                    for incoming, out in enumerate(values)
                )
                actual = int(
                    0 <= played < len(values) and values[played] != played
                )
                block = totals[name]
                block["decisions"] += 1
                block["legal_inputs"] += len(values)
                block["moved_inputs"] += moved
                block["binding_prompts"] += int(moved > 0)
                block["actual_overrides"] += actual
                match = by_matchup[row["opponent_family"]][name]
                match["moved_inputs"] += moved
                match["binding_prompts"] += int(moved > 0)
                match["actual_overrides"] += actual

            if (
                outputs["v21"][played] != outputs["v20"][played]
                and len(examples) < 200
            ):
                examples.append({
                    "version": row["version"],
                    "episode_id": row["episode_id"],
                    "opponent": row["opponent_family"],
                    "won": row["won"],
                    "turn": int(current.get("turn", -1)),
                    "context": int(select.get("context", -1)),
                    "played": played,
                    "v20_output": outputs["v20"][played],
                    "v21_output": outputs["v21"][played],
                })

            # Advance persistent state with the action actually played.
            for guard in guards.values():
                guard.adjust(observation, select, played, [played])

        for name, guard in guards.items():
            stats_total[name].update(guard.stats)

    payload = {
        "games": games,
        "totals": {name: dict(block) for name, block in totals.items()},
        "guard_stats": {
            name: dict(sorted(block.items())) for name, block in stats_total.items()
        },
        "by_matchup": {
            family: {name: dict(block) for name, block in blocks.items()}
            for family, blocks in sorted(
                by_matchup.items(),
                key=lambda item: -item[1]["v21"]["moved_inputs"],
            )
        },
        "changed_stored_decisions": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"games: {games}")
    for name in ("v20", "v21"):
        block = totals[name]
        print(
            f"  {name}: decisions {block['decisions']} "
            f"legal inputs {block['legal_inputs']} "
            f"moved {block['moved_inputs']} "
            f"binding prompts {block['binding_prompts']} "
            f"actual overrides {block['actual_overrides']}"
        )
    print(f"\nstored decisions where v21 differs from v20: {len(examples)}")
    for item in examples:
        print(
            f"  {item['version']:8s} ep{item['episode_id']} "
            f"turn {item['turn']:3d} {item['opponent'][:24]:24s} "
            f"won={item['won']:5s} {item['played']} -> {item['v21_output']}"
        )
    for key in ("last_breaker_evolve_refused", "last_breaker_only_body",
                "last_breaker_fastest_route", "dead_swing_turns", "errors"):
        print(
            f"  stat {key:32s} v20 {stats_total['v20'].get(key, 0):5d} "
            f"v21 {stats_total['v21'].get(key, 0):5d}"
        )
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
