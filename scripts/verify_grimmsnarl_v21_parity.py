"""Prove the v21 columns mean the same thing at fit time and at inference.

``upgrade_grimmsnarl_v21_corpus.py`` reconstructs six columns from primitives
the frozen corpus already stores.  ``ml_features.py`` computes them from the
live observation.  If the two definitions drift, training rewards a precision
the deployed agent does not have - the failure v20's audit caught on
``candidate_punk_targets_trigger``.

This walks stored ladder replays, runs the v21 feature extractor on every own
decision, applies the corpus formulas to the same extracted row, and reports
per-feature mismatches.  Both sides therefore see identical inputs and only the
definitions are under test.
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
AGENT = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21"
for path in (AGENT, ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402

IMPIDIMP = mf.IMPIDIMP_ID
MORGREM = mf.MORGREM_ID
GRIMMSNARL = mf.GRIMMSNARL_EX_ID
MARNIE_LINE = (IMPIDIMP, MORGREM, GRIMMSNARL)
BENCH_SLOTS = 5

CHECKED = (
    "active_retreat_locked",
    "bench_marnie_body_count",
    "bench_locked_ready_attacker",
    "retreat_lock_risk",
    "candidate_funds_active_retreat",
    "candidate_leaves_retreat_locked",
)


def corpus_formula(row: dict[str, Any]) -> dict[str, float]:
    """The upgrade script's arithmetic, expressed on one extracted row."""
    def value(name: str, default: float = 0.0) -> float:
        return float(row.get(name, default))

    active_id = value("self_active_id", -1)
    locked = (
        active_id > 0
        and int(active_id) not in MARNIE_LINE
        and value("self_active_dark_energy") == 0
    )
    bench_marnie = 0.0
    for slot in range(BENCH_SLOTS):
        slot_id = int(value(f"self_bench_slot_{slot}_id", -1))
        if slot_id in MARNIE_LINE:
            bench_marnie += 1.0
    risk = locked and bench_marnie > 0
    return {
        "active_retreat_locked": float(locked),
        "bench_marnie_body_count": bench_marnie,
        "bench_locked_ready_attacker": float(
            locked and value("backup_attacker_ready") > 0
        ),
        "retreat_lock_risk": float(risk),
        "candidate_funds_active_retreat": float(
            value("is_energy") > 0
            and value("candidate_target_is_active") > 0
            and risk
        ),
        "candidate_leaves_retreat_locked": float(
            value("is_energy") > 0
            and value("candidate_target_is_bench") > 0
            and risk
        ),
    }


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
    parser.add_argument("--versions", default="v19,v19_old,v20")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v21"
        / "feature_parity.json",
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

    mismatches: Counter = Counter()
    support: Counter = Counter()
    games = decisions = candidate_rows = 0
    examples: list[dict[str, Any]] = []

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
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        games += 1
        if args.max_games and games > args.max_games:
            games -= 1
            break

        for step in steps[:-1]:
            if seat >= len(step):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            current = observation.get("current") or {}
            if not options or not (current.get("players") or []):
                continue
            try:
                state = mf.state_features(current)
            except Exception:  # noqa: BLE001
                continue
            decisions += 1
            for option in options:
                try:
                    row = mf.option_features(
                        current, select, option, base_state=state,
                        observation=observation,
                    )
                except Exception:  # noqa: BLE001
                    continue
                candidate_rows += 1
                merged = dict(state)
                merged.update(row)
                expected = corpus_formula(merged)
                for name in CHECKED:
                    runtime = float(merged.get(name, 0.0))
                    if runtime:
                        support[name] += 1
                    if abs(runtime - expected[name]) > 1e-6:
                        mismatches[name] += 1
                        if len(examples) < 40:
                            examples.append({
                                "episode_id": meta["episode_id"],
                                "turn": int(current.get("turn", -1)),
                                "context": int(select.get("context", -1)),
                                "feature": name,
                                "runtime": runtime,
                                "corpus_formula": expected[name],
                            })

    payload = {
        "scope": {
            "versions": sorted(wanted),
            "games": games,
            "decisions": decisions,
            "candidate_rows": candidate_rows,
            "features_checked": len(CHECKED),
        },
        "comparison": (
            "runtime ml_features.py values versus the deterministic "
            "corpus_v21_current.npz reconstruction formulas"
        ),
        "total_mismatches": int(sum(mismatches.values())),
        "mismatches_by_feature": dict(mismatches),
        "runtime_support_rows": dict(support),
        "examples": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: v for k, v in payload.items() if k != "examples"},
        ensure_ascii=False, indent=2,
    ))
    for item in examples[:10]:
        print("  mismatch:", item)
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
