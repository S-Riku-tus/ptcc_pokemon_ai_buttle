"""v5's resource gates, re-measured on a later run of the same planner code.

v5 fixed Punk Up mining the deck dry; v6 did not touch `ml_planner.py`,
`fallback_policy.py` or `ml_features.py` (byte-identical, sha256-checked), so
these numbers are a regression check rather than a new result. Reuses v5's own
scanner so the definitions cannot drift.

Usage:
    python experiments/v6_meta_gap_analysis/check_v6_resource_gates.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "grimmsnarl_ml_v5"))

from measure_fuel_chain import scan  # noqa: E402

RUNS = {
    "v5": ("20260806_grimmsnarl_ml_v5_sub55275642", "55275642"),
    "v6": ("20260806_grimmsnarl_ml_v6_sub55290882", "55290882"),
}


def jobs_for(run: Path, submission: str, label: str) -> list[tuple]:
    out = []
    for row in csv.DictReader(open(run / "episodes.csv", encoding="utf-8-sig")):
        if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
            continue
        seat = 0 if row["agent_0_submission_id"] == submission else 1
        episode = row["episode_id"]
        path = (
            run / "episodes" / episode / "replay" / f"episode_{episode}.json"
        )
        if path.exists():
            out.append((str(path), seat, label))
    return out


def main() -> int:
    header = (
        f"{'run':<5}{'turns':>7}{'attach/turn':>13}{'take|off':>10}"
        f"{'darkHand%':>11}{'darkLeft':>10}{'lateDarkLeft':>14}"
        f"{'lateTake|off':>14}"
    )
    print(header)
    for label, (name, submission) in RUNS.items():
        run = ROOT / "data" / "runs" / "grimmsnarl" / name
        work = jobs_for(run, submission, label)
        counts: Counter = Counter()
        with ProcessPoolExecutor(max_workers=8) as pool:
            for result in pool.map(scan, work, chunksize=8):
                counts.update(result)
        turns = counts["own_turns"] or 1
        late = counts["late_turns"] or 1
        print(
            f"{label:<5}{counts['own_turns']:>7}"
            f"{100 * counts['turns_attach_taken'] / turns:>12.1f}%"
            f"{100 * counts['turns_attach_taken'] / max(1, counts['turns_attach_offered']):>9.1f}%"
            f"{100 * counts['turns_hand_dark'] / turns:>10.1f}%"
            f"{counts['deck_dark_sum'] / turns:>10.2f}"
            f"{counts['late_deck_dark_sum'] / late:>14.2f}"
            f"{100 * counts['late_attach_taken'] / max(1, counts['late_attach_offered']):>13.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
