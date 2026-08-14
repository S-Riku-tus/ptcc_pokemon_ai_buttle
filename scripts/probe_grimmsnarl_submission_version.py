"""Which agent version actually played a fetched Kaggle run?

Kaggle replays carry no build marker, so a mislabelled run directory is
unfalsifiable from its own metadata.  Teacher-forced agreement is the missing
evidence: walk the stored boards, ask each candidate agent for its answer, and
count how often that answer equals the action the submission really played.
The version that ran matches near-perfectly; any other version diverges by its
own footprint.

Usage:

    python scripts/probe_grimmsnarl_submission_version.py \
      --run data/runs/grimmsnarl/20260814_grimmsnarl_ml_v23_sub55486691 \
      --agent agents/grimmsnarl/grimmsnarl_ml_v22 \
      --agent agents/grimmsnarl/grimmsnarl_ml_v23 \
      --report experiments/grimmsnarl_ml_v23/version_probe_55486691.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_grimmsnarl_v21_footprint import load, single  # noqa: E402


def agreement(module: Any, replay: dict, seat: int) -> Counter:
    """Count own single-pick decisions where the agent reproduces the play."""
    counts: Counter = Counter()
    steps = replay.get("steps") or []
    for hook in ("diag_reset", "reset_state"):
        fn = getattr(module, hook, None)
        if callable(fn):
            fn()
            break
    # Same invariant as the footprint walker: the proposal is diagnostic, and
    # intra-turn history advances exactly once, with the stored action.
    ranker = getattr(module, "_RANKER", None)
    if ranker is not None and hasattr(ranker, "teacher_forced"):
        ranker.teacher_forced = True
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
        proposed = single(module.agent(observation))
        counts["decisions"] += 1
        counts["matched"] += int(proposed == played)
        module.observe_external(observation, played)
    return counts


def episodes_of(run_dir: Path, limit: int) -> list[tuple[str, Path, int]]:
    manifest = run_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"No manifest.csv under {run_dir}")
    out: list[tuple[str, Path, int]] = []
    for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
        seat = row.get("detected_submission_agent_index")
        if seat not in {"0", "1"}:
            continue
        path = (
            run_dir / "episodes" / row["episode_id"] / "replay"
            / f"episode_{row['episode_id']}.json"
        )
        if path.exists():
            out.append((row["episode_id"], path, int(seat)))
    out.sort(key=lambda item: int(item[0]))
    if limit > 0:
        out = out[:limit]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--agent", action="append", required=True, type=Path,
        help="Candidate agent directory; repeat for each version to test.",
    )
    parser.add_argument(
        "--max-games", type=int, default=0,
        help="Probe only the first N games by episode ID; 0 means all.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    selected = episodes_of(args.run, args.max_games)
    if not selected:
        print("No replays with a detected seat were found.", file=sys.stderr)
        return 1

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for directory in args.agent:
        module = load(ROOT / directory if not directory.is_absolute() else directory)
        totals: Counter = Counter()
        per_game: list[dict[str, Any]] = []
        for episode_id, path, seat in selected:
            replay = json.loads(path.read_text(encoding="utf-8"))
            counts = agreement(module, replay, seat)
            totals.update(counts)
            per_game.append({
                "episode_id": episode_id,
                "decisions": counts["decisions"],
                "matched": counts["matched"],
            })
        rate = totals["matched"] / max(1, totals["decisions"])
        results.append({
            "agent": str(directory),
            "games": len(selected),
            "decisions": totals["decisions"],
            "matched": totals["matched"],
            "agreement": round(rate, 4),
            "per_game": per_game,
        })
        print(
            f"  {directory}: {totals['matched']}/{totals['decisions']} "
            f"= {rate:.4f}"
        )

    ranked = sorted(results, key=lambda row: -row["agreement"])
    payload = {
        "run": str(args.run),
        "games": len(selected),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "best_match": ranked[0]["agent"],
        "margin": round(
            ranked[0]["agreement"] - ranked[1]["agreement"], 4
        ) if len(ranked) > 1 else None,
        "results": [
            {k: v for k, v in row.items() if k != "per_game"} for row in ranked
        ],
        "per_game": {row["agent"]: row["per_game"] for row in results},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report: {args.report}")
    print(json.dumps(
        {k: v for k, v in payload.items() if k != "per_game"},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
