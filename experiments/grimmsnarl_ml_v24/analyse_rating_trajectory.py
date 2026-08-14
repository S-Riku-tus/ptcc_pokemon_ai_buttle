"""Is 928 a policy verdict or an unfinished rating?

Kaggle starts every new submission near 600 and moves it by a sigma-damped
update, so a submission's *reported* rating is a function of how many games it
has been allowed to play as much as of how it plays.  This project has been
reading each run's final number as if it were converged.  This script prints,
per submission, the ordered per-game rating trajectory and the equilibrium
rating implied by the observed win rate against the observed opponents:

    implied = mean(opponent_rating) + 400 * log10(w / (1 - w))

If the trajectory is still rising at the last game and sits far below the
implied equilibrium, the run was truncated, not beaten.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/runs/grimmsnarl"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/rating_trajectory.json"

SUBMISSIONS = [
    ("v19", 55445763), ("v20", 55445769), ("v21", 55456713),
    ("v22_a", 55479857), ("v22_b", 55483874), ("v22_c", 55486680),
    ("v22_d", 55486691), ("v23", 55485982),
    ("v24_a", 55496021), ("v24_b", 55496665),
]


def run_dir(submission: int) -> Path | None:
    for path in sorted(RUNS.iterdir()):
        if path.is_dir() and path.name.endswith(f"sub{submission}"):
            return path
    return None


def trajectory(label: str, submission: int) -> dict[str, Any] | None:
    directory = run_dir(submission)
    if directory is None:
        return None
    rows = list(csv.DictReader(
        (directory / "episodes.csv").open(encoding="utf-8-sig")))
    points: list[dict[str, Any]] = []
    for row in rows:
        seat = 0 if row["agent_0_submission_id"] == str(submission) else 1
        try:
            updated = float(row[f"agent_{seat}_updated_score"])
            initial = float(row[f"agent_{seat}_initial_score"])
        except ValueError:
            continue
        try:
            opponent = float(row[f"agent_{1 - seat}_initial_score"])
        except ValueError:
            opponent = None
        points.append({
            "end_time": row["end_time"],
            "initial": initial,
            "updated": updated,
            "won": updated > initial,
            "opponent": opponent,
        })
    points.sort(key=lambda p: p["end_time"])
    if not points:
        return None

    wins = sum(1 for p in points if p["won"])
    played = len(points)
    win_rate = wins / played
    opponents = [p["opponent"] for p in points if p["opponent"] is not None]
    opponent_mean = sum(opponents) / len(opponents) if opponents else None
    clipped = min(max(win_rate, 1e-3), 1 - 1e-3)
    implied = (
        opponent_mean + 400 * math.log10(clipped / (1 - clipped))
        if opponent_mean is not None else None
    )
    last10 = points[-10:]
    return {
        "label": label,
        "submission": submission,
        "games": played,
        "record": f"{wins}-{played - wins}",
        "win_rate": round(win_rate, 4),
        "first_rating": round(points[0]["initial"], 1),
        "final_rating": round(points[-1]["updated"], 1),
        "peak_rating": round(max(p["updated"] for p in points), 1),
        "delta_last_10_games": round(
            points[-1]["updated"] - last10[0]["initial"], 1),
        "opponent_mean": round(opponent_mean, 1) if opponent_mean else None,
        "implied_equilibrium": round(implied, 1) if implied else None,
        "gap_to_equilibrium": round(implied - points[-1]["updated"], 1)
        if implied else None,
        "curve": [round(p["updated"], 1) for p in points],
        "opponent_curve": [
            round(p["opponent"], 1) if p["opponent"] is not None else None
            for p in points
        ],
    }


def main() -> int:
    payload = {}
    for label, submission in SUBMISSIONS:
        result = trajectory(label, submission)
        if result is None:
            continue
        payload[label] = result
        print(
            f"{label:<7} sub{submission}  n={result['games']:>3}  "
            f"{result['record']:>7} {result['win_rate']:.3f}  "
            f"start {result['first_rating']:>6.1f} -> final "
            f"{result['final_rating']:>6.1f} (peak {result['peak_rating']:.1f})"
        )
        print(
            f"        opp mean {result['opponent_mean']}  implied equilibrium "
            f"{result['implied_equilibrium']}  gap "
            f"{result['gap_to_equilibrium']}  last-10 delta "
            f"{result['delta_last_10_games']}"
        )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
