"""Is v21 a different policy from v20, or another v18?

v18 shipped two guards that bound zero times and a model that answered the same
way almost everywhere; the ladder could not have told it apart from v17. The
test that would have caught it is this one: teacher-force both agents through
the same stored boards and count how many decisions and how many games they
actually answer differently. Stored actions advance both runtimes, so one
changed answer never invents a divergent future.
"""

from __future__ import annotations

import argparse
import csv
import json
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

MODULES = (
    "main", "ml_runtime", "ml_features", "fallback_policy", "ml_planner",
    "ml_residual", "policy_router", "matchup_guard", "attack_access",
    "policy_base", "wall_break", "mirror_prize", "horizon_prize",
)


def load(agent_dir: Path) -> Any:
    for name in MODULES:
        sys.modules.pop(name, None)
    for entry in list(sys.path):
        if "grimmsnarl_ml_v" in entry or "v21_candidates" in entry:
            sys.path.remove(entry)
    return load_dir_agent_module(agent_dir)


def single(action: Any) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
    ):
        return action[0]
    return None


def answers(module: Any, replay: dict, seat: int) -> list[tuple[int, int, int, int | None]]:
    """(step, turn, context, answer) for every own single-pick decision."""
    out = []
    steps = replay.get("steps") or []
    for hook in ("diag_reset", "reset_state"):
        fn = getattr(module, hook, None)
        if callable(fn):
            fn()
            break
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(select, dict) or not isinstance(current, dict):
            continue
        if not current.get("players") or not (select.get("option") or []):
            continue
        played = single((steps[index + 1][seat] or {}).get("action"))
        if played is None:
            continue
        out.append((
            index,
            int(current.get("turn", -1)),
            int(select.get("context", -1)),
            single(module.agent(observation)),
        ))
        module.observe_external(observation, played)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="agents/grimmsnarl/grimmsnarl_ml_v20")
    parser.add_argument("--candidate", default="agents/grimmsnarl/grimmsnarl_ml_v21")
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20"
        / "ladder_history_games.csv",
    )
    parser.add_argument(
        "--runs", type=Path, default=ROOT / "data" / "runs" / "grimmsnarl"
    )
    parser.add_argument("--versions", default="v19,v19_old,v20")
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v21"
        / "footprint_v20_vs_v21.json",
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

    selected = []
    for meta in csv.DictReader(args.games.open(encoding="utf-8-sig")):
        if wanted and meta["version"] not in wanted:
            continue
        entry = index.get(meta["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (
            run_dir / "episodes" / meta["episode_id"] / "replay"
            / f"episode_{meta['episode_id']}.json"
        )
        if path.exists():
            selected.append(
                (meta["episode_id"], path, seat, meta["opponent_family"])
            )

    started = time.perf_counter()
    collected: dict[str, list] = {}
    for label, directory in (("base", args.base), ("candidate", args.candidate)):
        module = load(ROOT / directory)
        rows = []
        for episode_id, path, seat, family in selected:
            replay = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                (episode_id, family, answers(module, replay, seat))
            )
        collected[label] = rows
        print(f"  {label} done ({directory})")

    totals: Counter = Counter()
    by_matchup: dict[str, Counter] = defaultdict(Counter)
    by_context: Counter = Counter()
    examples: list[dict[str, Any]] = []
    for (episode_id, family, base), (_, _, cand) in zip(
        collected["base"], collected["candidate"]
    ):
        touched = False
        totals["games"] += 1
        by_matchup[family]["games"] += 1
        for (step, turn, context, a), (_, _, _, b) in zip(base, cand):
            totals["decisions"] += 1
            by_matchup[family]["decisions"] += 1
            if a != b:
                totals["changed"] += 1
                by_matchup[family]["changed"] += 1
                by_context[context] += 1
                touched = True
                if len(examples) < 100:
                    examples.append({
                        "episode_id": episode_id,
                        "opponent": family,
                        "step": step,
                        "turn": turn,
                        "context": context,
                        "v20": a,
                        "v21": b,
                    })
        totals["games_touched"] += int(touched)
        by_matchup[family]["games_touched"] += int(touched)

    payload = {
        "base": args.base,
        "candidate": args.candidate,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "totals": dict(totals),
        "changed_share": round(
            totals["changed"] / max(1, totals["decisions"]), 4
        ),
        "by_context": dict(sorted(by_context.items())),
        "by_matchup": {
            family: dict(counts)
            for family, counts in sorted(
                by_matchup.items(), key=lambda item: -item[1]["changed"]
            )
        },
        "examples": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: v for k, v in payload.items() if k not in {"examples", "by_matchup"}},
        ensure_ascii=False, indent=2,
    ))
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
