"""Add v20 attack-continuity columns to an existing Grimmsnarl corpus.

The v19 corpus already stores every public observation primitive needed by the
new ETA and candidate-route features: ordered board ids/energy/age, hand card
counts, select context/effect, and candidate target identity.  Reconstructing
the derived columns from those primitives is lossless for the board ETA
features and deterministic for candidate features.  It also avoids depending
on replay files that may have been cleaned after the frozen v19 corpus was
built.

The output keeps all v19 arrays and appends feature columns by name.  Runtime
models address features by name, not physical position, so this representation
is equivalent to a fresh extraction for training and deployment.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL = 648
RARE_CANDY = 1079
DARKNESS = 7
MARNIE_LINE = (IMPIDIMP, MORGREM, GRIMMSNARL)

ACTION_ENERGY = 5
ACTION_EVOLVE = 6
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_ATTACH_FROM = 21
AREA_ACTIVE = 4
AREA_BENCH = 5


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

        def col(name: str, default: float = 0.0) -> np.ndarray:
            if name not in index:
                return np.full(rows, default, dtype=np.float32)
            return matrix[:, index[name]].astype(np.float32, copy=False)

        hand_grim = col("hand_648")
        hand_morgrem = col("hand_647")
        hand_candy = col("hand_1079")

        def eta(card_id: np.ndarray, dark: np.ndarray,
                appeared: np.ndarray) -> np.ndarray:
            out = np.full(rows, 9.0, dtype=np.float32)
            grim = card_id == GRIMMSNARL
            out[grim] = np.maximum(0.0, 2.0 - dark[grim])
            morgrem = card_id == MORGREM
            out[morgrem] = appeared[morgrem] + np.where(
                hand_grim[morgrem] > 0, 1.0, 2.0
            )
            imp = card_id == IMPIDIMP
            candy = (hand_candy > 0) & (hand_grim > 0)
            out[imp] = appeared[imp] + np.where(
                candy[imp], 1.0, np.where(hand_morgrem[imp] > 0, 2.0, 3.0)
            )
            return np.minimum(out, 9.0).astype(np.float32)

        active_id = col("self_active_id", -1)
        active_dark = col("self_active_dark_energy")
        active_appeared = col("self_active_appear_this_turn")
        active_eta = eta(active_id, active_dark, active_appeared)

        bench_etas = []
        for slot in range(5):
            prefix = f"self_bench_slot_{slot}"
            bench_etas.append(eta(
                col(f"{prefix}_id", -1),
                col(f"{prefix}_dark_energy"),
                col(f"{prefix}_appear_this_turn"),
            ))
        bench_matrix = np.column_stack(bench_etas)
        backup_eta = np.min(bench_matrix, axis=1)
        all_etas = np.column_stack([active_eta, bench_matrix])
        sorted_etas = np.sort(all_etas, axis=1)
        second_eta = sorted_etas[:, 1]

        active_ready = col("active_attacker_ready")
        backup_ready = col("backup_attacker_ready")
        total_ready = col("ready_attacker_count")

        additions: dict[str, np.ndarray] = {
            "active_grim_line_eta": active_eta,
            "backup_grim_line_eta": backup_eta,
            "second_grim_line_eta": second_eta,
            "backup_grim_eta_le1": (backup_eta <= 1).astype(np.float32),
            "backup_grim_eta_le2": (backup_eta <= 2).astype(np.float32),
            "attack_chain_ready": (
                (active_ready > 0) & (backup_ready > 0)
            ).astype(np.float32),
            "future_attacker_line_count": np.sum(
                all_etas <= 2, axis=1
            ).astype(np.float32),
            "single_attacker_risk": (
                (total_ready == 1) & (second_eta > 1)
            ).astype(np.float32),
        }

        target_id = col("candidate_target_id", -1)
        target_dark = col("candidate_target_dark_energy")
        target_appeared = col("candidate_target_appear_this_turn")
        target_eta_before = eta(target_id, target_dark, target_appeared)
        action = col("action_type_id", -1)
        card_id = col("candidate_card_id", -1)
        context = col("select_context", -1)
        effect_id = col("select_effect_card_id", -1)
        target_area = np.where(
            col("candidate_inplay_area", -1) >= 0,
            col("candidate_inplay_area", -1),
            col("ctx_area", -1),
        )
        target_own = (
            (col("candidate_raw_player_relative", 0) == 0)
            & (col("ctx_owner_is_self", 1) != 0)
        )
        target_line = np.isin(target_id, MARNIE_LINE) & target_own
        punk = (
            (context == CTX_ATTACH_FROM)
            & (effect_id == GRIMMSNARL)
            & target_line
        )
        manual_dark = (
            (action == ACTION_ENERGY) & (card_id == DARKNESS) & target_line
        )
        adds_dark = punk | manual_dark
        evolve_grim = (action == ACTION_EVOLVE) & (card_id == GRIMMSNARL)

        target_eta_after = target_eta_before.copy()
        funded_grim = (target_id == GRIMMSNARL) & adds_dark
        target_eta_after[funded_grim] = np.maximum(
            0.0, target_eta_before[funded_grim] - 1.0
        )
        target_eta_after[evolve_grim] = 0.0
        ready_target = (target_id == GRIMMSNARL) & (target_dark >= 2)
        promotes_ready = (
            np.isin(context, (CTX_SWITCH, CTX_TO_ACTIVE))
            & target_own & (target_area == AREA_BENCH) & ready_target
        )

        # The v19 corpus did not retain the effect serial.  v20 therefore uses
        # the same fit/runtime definition: a Grimmsnarl that appeared this
        # turn during the Punk Up chain.  Keeping that definition identical is
        # more important than introducing a serial-only inference feature.
        punk_trigger = punk & (target_id == GRIMMSNARL) & (target_appeared > 0)

        # Whether a ready promotion exists is a group-level offer feature.
        promotion_offered = np.zeros(rows, dtype=np.float32)
        starts = np.r_[0, np.cumsum(source["groups"], dtype=np.int64)[:-1]]
        ends = np.cumsum(source["groups"], dtype=np.int64)
        for start, end in zip(starts, ends):
            if np.any(promotes_ready[start:end]):
                promotion_offered[start:end] = float(
                    np.count_nonzero(promotes_ready[start:end])
                )

        additions.update({
            "candidate_target_grim_line_eta_before": target_eta_before,
            "candidate_target_grim_line_eta_after": target_eta_after,
            "candidate_target_grim_line_eta_gain": np.maximum(
                0.0, target_eta_before - target_eta_after
            ),
            "candidate_creates_ready_active": (
                (target_eta_after == 0) & (target_area == AREA_ACTIVE)
                & target_line
            ).astype(np.float32),
            "candidate_creates_ready_backup": (
                (target_eta_after == 0) & (target_area == AREA_BENCH)
                & target_line
            ).astype(np.float32),
            "candidate_punk_targets_trigger": punk_trigger.astype(np.float32),
            "candidate_punk_funds_active": (
                punk & (target_area == AREA_ACTIVE)
            ).astype(np.float32),
            "candidate_punk_overfunds_ready_active": (
                punk & (target_area == AREA_ACTIVE) & ready_target
            ).astype(np.float32),
            "candidate_punk_funds_future_line": (
                punk & (target_area == AREA_BENCH) & ~ready_target
            ).astype(np.float32),
            "candidate_ready_promotion_offered": promotion_offered,
            "candidate_promotes_ready_attacker": promotes_ready.astype(np.float32),
            "candidate_passes_ready_promotion": (
                (promotion_offered > 0) & ~promotes_ready
            ).astype(np.float32),
        })

        duplicates = sorted(set(additions) & set(names))
        if duplicates:
            raise SystemExit(f"v20 columns already exist: {duplicates}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        added_names = sorted(additions)
        # The source matrix is ~1.5 GB.  Building ``column_stack`` beside it
        # briefly doubles resident memory and can kill an otherwise valid
        # corpus upgrade.  Materialise the widened matrix as a bounded-memory
        # memmap, then let NumPy stream it into the compressed archive.
        staging = args.output.with_suffix(args.output.suffix + ".features.npy")
        try:
            upgraded = np.lib.format.open_memmap(
                staging,
                mode="w+",
                dtype=np.float32,
                shape=(rows, len(names) + len(added_names)),
            )
            step = 50_000
            for start in range(0, rows, step):
                end = min(rows, start + step)
                upgraded[start:end, :len(names)] = matrix[start:end]
                for offset, name in enumerate(added_names, start=len(names)):
                    upgraded[start:end, offset] = additions[name][start:end]
            upgraded.flush()

            payload = {key: source[key] for key in source.files}
            payload["features"] = upgraded
            payload["feature_names"] = np.asarray(names + added_names)
            np.savez_compressed(args.output, **payload)
            del payload["features"]
            upgraded._mmap.close()
            del upgraded
            gc.collect()
        finally:
            staging.unlink(missing_ok=True)

        report = {
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "source_features": len(names),
            "added_features": added_names,
            "output_features": len(names) + len(added_names),
            "candidate_rows": rows,
            "decisions": int(len(source["groups"])),
            "nonzero_support": {
                name: int(np.count_nonzero(np.nan_to_num(additions[name])))
                for name in added_names
            },
            "derivation": (
                "deterministic reconstruction from v19 public-observation "
                "primitive columns; no result, opponent deck identity, or "
                "teacher identity added"
            ),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
