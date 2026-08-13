"""Parallel teacher-forced action footprint for two directory agents."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "vendor", ROOT / "scripts", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402
from probe_grimmsnarl_v21_footprint import answers  # noqa: E402

_MODULES: dict[str, Any] = {}
_INIT_ERROR: str | None = None


def _init(base: str, candidate: str) -> None:
    global _INIT_ERROR
    try:
        _MODULES["base"] = load_dir_agent_module(Path(base))
        _MODULES["candidate"] = load_dir_agent_module(Path(candidate))
    except Exception as exc:  # noqa: BLE001
        _INIT_ERROR = f"{type(exc).__name__}: {exc}"


def _job(payload: tuple[str, str, int, str]) -> dict[str, Any]:
    episode_id, path, seat, family = payload
    if _INIT_ERROR:
        return {"episode_id": episode_id, "error": _INIT_ERROR}
    try:
        replay = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            "episode_id": episode_id,
            "family": family,
            "base": answers(_MODULES["base"], replay, seat),
            "candidate": answers(_MODULES["candidate"], replay, seat),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"}


def _selected(args: argparse.Namespace) -> list[tuple[str, str, int, str]]:
    wanted = {value for value in args.versions.split(",") if value}
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
                (meta["episode_id"], str(path), seat, meta["opponent_family"])
            )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20" / "ladder_history_games.csv",
    )
    parser.add_argument(
        "--runs", type=Path, default=ROOT / "data" / "runs" / "grimmsnarl"
    )
    parser.add_argument("--versions", default="v19,v19_old,v20")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    selected = _selected(args)
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        max(1, args.workers),
        initializer=_init,
        initargs=(str(args.base.resolve()), str(args.candidate.resolve())),
    ) as pool:
        rows = []
        for completed, row in enumerate(
            pool.imap_unordered(_job, selected, chunksize=1), start=1
        ):
            rows.append(row)
            if completed % max(1, len(selected) // 20) == 0 or completed == len(selected):
                print(f"progress {completed}/{len(selected)}", file=sys.stderr, flush=True)
    rows.sort(key=lambda row: int(row["episode_id"]))
    errors = [row for row in rows if row.get("error")]

    totals: Counter[str] = Counter()
    by_matchup: dict[str, Counter[str]] = defaultdict(Counter)
    by_context: Counter[int] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row.get("error"):
            continue
        family = row["family"]
        base, candidate = row["base"], row["candidate"]
        totals["games"] += 1
        by_matchup[family]["games"] += 1
        touched = False
        if len(base) != len(candidate):
            errors.append({"episode_id": row["episode_id"], "error": "decision_count_mismatch"})
            continue
        for (step, turn, context_id, left), (_, _, _, right) in zip(base, candidate):
            totals["decisions"] += 1
            by_matchup[family]["decisions"] += 1
            if left == right:
                continue
            touched = True
            totals["changed"] += 1
            by_matchup[family]["changed"] += 1
            by_context[context_id] += 1
            if len(examples) < 100:
                examples.append({
                    "episode_id": row["episode_id"], "opponent": family,
                    "step": step, "turn": turn, "context": context_id,
                    "base": left, "candidate": right,
                })
        totals["games_touched"] += int(touched)
        by_matchup[family]["games_touched"] += int(touched)

    report = {
        "valid": not errors and totals["games"] == len(selected),
        "base": str(args.base),
        "candidate": str(args.candidate),
        "workers": args.workers,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "selected_games": len(selected),
        "errors": errors[:100],
        "totals": dict(totals),
        "changed_actions_per_game": totals["changed"] / max(1, totals["games"]),
        "changed_share": totals["changed"] / max(1, totals["decisions"]),
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k not in {"examples", "by_matchup"}}, ensure_ascii=False, indent=2))
    print(f"report: {args.report.resolve()}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
