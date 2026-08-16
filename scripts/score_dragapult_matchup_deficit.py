"""How many wins does our opponent mix say we should have had?

A raw win-rate difference against the teachers confounds two things: playing
worse, and meeting a harder field.  Applying the teachers' per-archetype win
rate to the archetypes we actually faced separates them, and the per-cell
residual says which matchups carry the whole difference.

Reads the report written by analyze_dragapult_matchups.py.

Usage:
  python scripts/score_dragapult_matchup_deficit.py \
      --report experiments/dragapult_ml_v2/matchups_v2_26g.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-teacher-games", type=int, default=20,
                        help="Below this the teacher rate is itself noise.")
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    # The report is keyed by cohort; join the two on the opponent label.
    teachers = {
        str(row["opponent"]): row
        for row in (data.get("teachers") or {}).get("rows") or []
    }
    live_key = next(
        (key for key in data if key != "teachers" and isinstance(data[key], dict)),
        None,
    )
    live = {
        str(row["opponent"]): row
        for row in (data.get(live_key) or {}).get("rows") or []
    }

    total_games = 0
    total_wins = 0.0
    total_expected = 0.0
    residuals = []
    weak_evidence = []
    for opponent, row in live.items():
        live_n = int(row.get("games") or 0)
        if not live_n:
            continue
        teacher_row = teachers.get(opponent) or {}
        teacher_n = int(teacher_row.get("games") or 0)
        teacher_wr = teacher_row.get("win_rate")
        live_wr = row.get("win_rate")
        if teacher_wr is None or live_wr is None:
            continue
        wins = float(row.get("wins", float(live_wr) * live_n))
        expected = float(teacher_wr) * live_n
        total_games += live_n
        total_wins += wins
        total_expected += expected
        entry = {
            "opponent": opponent, "live_n": live_n,
            "teacher_n": teacher_n, "teacher_wr": round(float(teacher_wr), 3),
            "live_wr": round(float(live_wr), 3),
            "wins": wins, "expected": round(expected, 2),
            "residual": round(wins - expected, 2),
        }
        (residuals if teacher_n >= args.min_teacher_games
         else weak_evidence).append(entry)

    residuals.sort(key=lambda item: item["residual"])
    print(f"{'opponent':30} {'live':>5} {'teach_n':>8} {'teach_wr':>9} "
          f"{'live_wr':>8} {'exp_w':>7} {'act_w':>6} {'resid':>7}")
    for entry in residuals + weak_evidence:
        marker = "" if entry["teacher_n"] >= args.min_teacher_games else "  (thin)"
        print(f"{str(entry['opponent'])[:30]:30} {entry['live_n']:>5} "
              f"{entry['teacher_n']:>8} {entry['teacher_wr']:>9.3f} "
              f"{entry['live_wr']:>8.3f} {entry['expected']:>7.2f} "
              f"{entry['wins']:>6.0f} {entry['residual']:>+7.2f}{marker}")

    print(f"\ngames {total_games}  actual wins {total_wins:.0f}  "
          f"expected on the teachers' rates {total_expected:.2f}  "
          f"residual {total_wins - total_expected:+.2f}")
    if total_games:
        print(f"actual win rate {total_wins / total_games:.4f}  "
              f"field-adjusted expectation {total_expected / total_games:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
