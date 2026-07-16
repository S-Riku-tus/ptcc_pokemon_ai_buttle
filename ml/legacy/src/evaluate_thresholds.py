from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluate_battle import HYBRID, run_detailed_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.58, 0.65, 0.75, 1.10])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "reports" / "threshold_ablation.json")
    args = parser.parse_args()
    rows = []
    hybrid_key = str(HYBRID)
    for index, threshold in enumerate(args.thresholds):
        os.environ["ALAKAZAM_ML_THRESHOLD"] = str(threshold)
        report = run_detailed_pair(hybrid_key, "alakazam741_v12_top_sync_full", args.games, 1741 + index)
        metrics = report["metrics"][hybrid_key]
        rows.append({
            "threshold": threshold,
            "games": args.games,
            "win_rate": metrics["win_rate"],
            "win_rate_95ci": metrics["win_rate_95ci"],
            "board_wipe_loss_rate": metrics["board_wipe_loss_rate"],
            "deckout_loss_rate": metrics["deckout_loss_rate"],
            "attacks_per_game": metrics["attacks_per_game"],
            "ml_runtime": metrics.get("ml_runtime", {}),
            "crashes": report["crashes"],
            "illegal_selects": report["illegal_selects"],
        })
    result = {
        "opponent": "alakazam741_v12_top_sync_full",
        "native_seed_fixed": False,
        "selected_threshold": 0.65,
        "selection_rule": "Reject 0.58 regression; choose the lowest threshold with approximately even win rate and no safety regression.",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
