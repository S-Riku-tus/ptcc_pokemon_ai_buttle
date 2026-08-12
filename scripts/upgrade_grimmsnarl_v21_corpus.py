"""Append v21's retreat-lock columns to the frozen v20 Grimmsnarl corpus.

The six new columns are functions of primitives the v20 corpus already stores -
the Active's id and Darkness count, each Bench slot's id, the benched
ready-attacker count, the action type and the candidate's target area - so they
can be reconstructed losslessly instead of re-extracting from replays that may
have been cleaned. ``scripts/verify_grimmsnarl_v21_parity.py`` proves the
reconstruction equals what the runtime computes, decision by decision.

Runtime models address features by name, so appending is equivalent to a fresh
extraction for both training and deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL = 648
MARNIE_LINE = (IMPIDIMP, MORGREM, GRIMMSNARL)
BENCH_SLOTS = 5

NEW_COLUMNS = (
    "active_retreat_locked",
    "bench_marnie_body_count",
    "bench_locked_ready_attacker",
    "retreat_lock_risk",
    "candidate_funds_active_retreat",
    "candidate_leaves_retreat_locked",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=False)
    try:
        names = [str(value) for value in source["feature_names"]]
        index = {name: slot for slot, name in enumerate(names)}
        matrix = source["features"]
        rows = len(matrix)

        missing = [
            name for name in (
                "self_active_id", "self_active_dark_energy",
                "backup_attacker_ready", "is_energy",
                "candidate_target_is_active", "candidate_target_is_bench",
            ) if name not in index
        ]
        if missing:
            raise SystemExit(f"corpus lacks required primitives: {missing}")
        already = [name for name in NEW_COLUMNS if name in index]
        if already:
            raise SystemExit(f"corpus already has: {already}")

        def col(name: str, default: float = 0.0) -> np.ndarray:
            if name not in index:
                return np.full(rows, default, dtype=np.float32)
            return matrix[:, index[name]].astype(np.float32, copy=False)

        active_id = col("self_active_id", -1)
        active_dark = col("self_active_dark_energy")
        bench_ready = col("backup_attacker_ready")

        # An empty Active never happens inside a MAIN prompt, but the corpus
        # also stores forced-promotion contexts where it briefly does; -1 is
        # the extractor's "no body" sentinel and must not read as locked.
        has_active = active_id > 0
        active_is_marnie = np.zeros(rows, dtype=bool)
        for card_id in MARNIE_LINE:
            active_is_marnie |= active_id == card_id
        locked = has_active & (~active_is_marnie) & (active_dark == 0)

        bench_marnie = np.zeros(rows, dtype=np.float32)
        for slot in range(BENCH_SLOTS):
            slot_id = col(f"self_bench_slot_{slot}_id", -1)
            hit = np.zeros(rows, dtype=bool)
            for card_id in MARNIE_LINE:
                hit |= slot_id == card_id
            bench_marnie += hit.astype(np.float32)

        risk = locked & (bench_marnie > 0)
        addition = {
            "active_retreat_locked": locked.astype(np.float32),
            "bench_marnie_body_count": bench_marnie,
            "bench_locked_ready_attacker": (
                locked & (bench_ready > 0)
            ).astype(np.float32),
            "retreat_lock_risk": risk.astype(np.float32),
            "candidate_funds_active_retreat": (
                (col("is_energy") > 0)
                & (col("candidate_target_is_active") > 0)
                & risk
            ).astype(np.float32),
            "candidate_leaves_retreat_locked": (
                (col("is_energy") > 0)
                & (col("candidate_target_is_bench") > 0)
                & risk
            ).astype(np.float32),
        }

        extended = np.concatenate(
            [matrix] + [addition[name].reshape(-1, 1) for name in NEW_COLUMNS],
            axis=1,
        ).astype(np.float32, copy=False)
        payload = {
            key: source[key] for key in source.files
            if key not in {"features", "feature_names"}
        }
        payload["features"] = extended
        payload["feature_names"] = np.asarray(
            names + list(NEW_COLUMNS), dtype=object
        ).astype("U")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **payload)

        report = {
            "input": str(args.input),
            "output": str(args.output),
            "rows": int(rows),
            "features_before": len(names),
            "features_after": extended.shape[1],
            "new_columns": list(NEW_COLUMNS),
            "support": {
                name: {
                    "nonzero_rows": int(np.count_nonzero(addition[name])),
                    "share": round(
                        float(np.count_nonzero(addition[name])) / rows, 6
                    ),
                    "max": float(addition[name].max()),
                }
                for name in NEW_COLUMNS
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
