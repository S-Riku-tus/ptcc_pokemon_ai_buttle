"""Is the top of this leaderboard skill or luck?

The rating a submission converges to is ``mean(opponent) + 400 log10(w/(1-w))``,
and a 34-game run of one fixed policy has a 90% spread of roughly 215 points
(``analyze_grimmsnarl_rating_mechanics.py``).  That invites the conclusion
that the standings are noise.  This script tests that against the three full
leaderboard snapshots stored under ``data/kaggle_top100`` - 2026-08-05,
08-07 and 08-14, about 6.3k to 6.8k teams each - plus the per-team list of
public submissions in the newest snapshot.

Three independent cuts:

* **persistence** - rank correlation between snapshots, and what happens to
  the teams that were on top nine days earlier.  Pure noise regresses fully to
  the mean; skill does not.
* **same-team spread** - every team runs two submission slots at once, so the
  gap between a team's two live scores is an upper bound on run-to-run noise
  at that skill level (upper, because the two slots usually hold different
  agents).
* **where we sit** - the percentile of each of our own ratings, and the score
  actually required to reach a given rank.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "data/kaggle_top100"
SNAPSHOTS = (
    ("2026-08-05", "20260805_113507_JST"),
    ("2026-08-07", "20260807_104146_JST"),
    ("2026-08-14", "20260814_215710_JST"),
)
OURS = {
    "v22_c": 1020.4, "v22_b": 1018.6, "v22_a": 1000.6, "v22_d": 952.8,
    "v24_b": 928.1, "v24_a": 911.3, "v25_a": 910.7, "v27": 853.3,
    "v26": 835.5, "v25_b": 808.4,
}


def load_board(folder: str) -> dict[int, dict]:
    path = TOP / folder / "raw/api/leaderboard_full.json"
    board: dict[int, dict] = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["publicLeaderboard"]:
        try:
            score = float(row["displayScore"])
        except (TypeError, ValueError):
            continue
        board[int(row["teamId"])] = {
            "score": score, "rank": int(row["rank"]),
            "submission": row.get("submissionId"),
        }
    return board


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/leaderboard_persistence.json",
    )
    args = parser.parse_args()
    payload: dict = {}

    boards = {day: load_board(folder) for day, folder in SNAPSHOTS}

    print("=== 1. the board itself ===")
    print(f"{'day':<12}{'teams':>7}{'rank1':>9}{'top10':>9}{'top50':>9}"
          f"{'top100':>9}{'median':>9}{'p90':>9}")
    for day, _ in SNAPSHOTS:
        scores = np.array(sorted(
            (row["score"] for row in boards[day].values()), reverse=True
        ))
        print(f"{day:<12}{len(scores):>7}{scores[0]:>9.1f}{scores[9]:>9.1f}"
              f"{scores[49]:>9.1f}{scores[99]:>9.1f}"
              f"{np.median(scores):>9.1f}"
              f"{np.percentile(scores, 90):>9.1f}")
        payload.setdefault("boards", {})[day] = {
            "teams": len(scores), "rank1": float(scores[0]),
            "top10": float(scores[9]), "top50": float(scores[49]),
            "top100": float(scores[99]), "median": float(np.median(scores)),
        }

    print("\nthe #1 team is a different team in all three snapshots:")
    for day, _ in SNAPSHOTS:
        top = min(boards[day].items(), key=lambda item: item[1]["rank"])
        print(f"  {day}  team {top[0]}  score {top[1]['score']:.1f}")

    print("\n=== 2. persistence between snapshots ===")
    pairs = (("2026-08-05", "2026-08-07"), ("2026-08-07", "2026-08-14"),
             ("2026-08-05", "2026-08-14"))
    print(f"{'window':<26}{'both':>7}{'spearman':>10}{'pearson':>9}"
          f"{'sd of change':>14}")
    for first, second in pairs:
        common = sorted(set(boards[first]) & set(boards[second]))
        a = np.array([boards[first][t]["score"] for t in common])
        b = np.array([boards[second][t]["score"] for t in common])
        rho = spearmanr(a, b).statistic
        pear = float(np.corrcoef(a, b)[0, 1])
        print(f"{first[5:]} -> {second[5:]:<18}{len(common):>7}{rho:>10.3f}"
              f"{pear:>9.3f}{float(np.std(b - a)):>14.1f}")
        payload.setdefault("persistence", {})[f"{first}->{second}"] = {
            "n": len(common), "spearman": float(rho), "pearson": pear,
        }

    print("\n=== 3. what happens to the teams that were on top ===")
    first, second = "2026-08-05", "2026-08-14"
    common = sorted(set(boards[first]) & set(boards[second]))
    ordered = sorted(common, key=lambda t: boards[first][t]["rank"])
    overall = float(np.mean([boards[second][t]["score"] for t in common]))
    print(f"field mean on {second}: {overall:.1f} over {len(common)} "
          f"teams present in both snapshots")
    print(f"{'bucket on 08-05':<20}{'n':>5}{'score 08-05':>13}"
          f"{'score 08-14':>13}{'kept':>8}{'still top50':>13}"
          f"{'still top200':>14}")
    buckets = ((0, 10), (10, 50), (50, 100), (100, 250), (250, 500),
               (500, 1000))
    for low, high in buckets:
        group = ordered[low:high]
        if not group:
            continue
        before = float(np.mean([boards[first][t]["score"] for t in group]))
        after = float(np.mean([boards[second][t]["score"] for t in group]))
        kept = (after - overall) / (before - overall) if before != overall else 0
        top50 = sum(boards[second][t]["rank"] <= 50 for t in group) / len(group)
        top200 = sum(boards[second][t]["rank"] <= 200 for t in group) / len(group)
        print(f"{f'rank {low + 1}-{high}':<20}{len(group):>5}{before:>13.1f}"
              f"{after:>13.1f}{kept:>8.2f}{top50:>13.1%}{top200:>14.1%}")
        payload.setdefault("top_decay", {})[f"{low + 1}-{high}"] = {
            "n": len(group), "before": before, "after": after,
            "kept_fraction": kept, "still_top50": top50,
        }
    print("\n'kept' is the fraction of the team's edge over the field that")
    print("survived nine days. 0.00 would be pure luck fully regressing;")
    print("1.00 would be a perfectly stable skill ordering.")

    print("\n=== 4. same team, two live slots, same day ===")
    gaps = []
    rows = []
    for path in sorted(
        (TOP / "latest/raw/api/team_public_submissions").glob("team_*.json")
    ):
        subs = json.loads(path.read_text(encoding="utf-8"))["submissions"]
        scores = []
        for sub in subs:
            try:
                scores.append(float(sub["publicScoreFormatted"]))
            except (TypeError, ValueError, KeyError):
                continue
        if len(scores) < 2:
            continue
        gap = max(scores) - min(scores)
        gaps.append(gap)
        rows.append((int(path.stem.split("_")[1]), max(scores), gap))
    gaps_array = np.array(gaps)
    print(f"teams with two scored slots: {len(gaps)}")
    print(f"  gap between a team's own two slots: "
          f"median {np.median(gaps_array):.1f}, "
          f"mean {gaps_array.mean():.1f}, "
          f"p25 {np.percentile(gaps_array, 25):.1f}, "
          f"p75 {np.percentile(gaps_array, 75):.1f}, "
          f"max {gaps_array.max():.1f}")
    print(f"  share above 100 points: {(gaps_array > 100).mean():.1%}; "
          f"above 150: {(gaps_array > 150).mean():.1%}")
    rows.sort(key=lambda r: -r[1])
    print("\n  the ten highest-scoring teams and their own internal gap:")
    print(f"  {'team':>12}{'best':>9}{'gap to their other slot':>26}")
    for team, best, gap in rows[:10]:
        print(f"  {team:>12}{best:>9.1f}{gap:>26.1f}")
    payload["same_team_gap"] = {
        "n": len(gaps), "median": float(np.median(gaps_array)),
        "mean": float(gaps_array.mean()), "max": float(gaps_array.max()),
    }

    print("\n=== 5. where our runs sit ===")
    latest = boards["2026-08-14"]
    scores = np.array(sorted((r["score"] for r in latest.values()),
                             reverse=True))
    print(f"{'run':<8}{'rating':>9}{'rank':>8}{'percentile':>12}")
    for run, rating in OURS.items():
        rank = int((scores > rating).sum()) + 1
        print(f"{run:<8}{rating:>9.1f}{rank:>8}"
              f"{1 - rank / len(scores):>12.1%}")
        payload.setdefault("our_rank", {})[run] = rank
    print("\nscore needed for a given rank on 2026-08-14:")
    for rank in (1, 3, 10, 25, 50, 100, 200, 400):
        print(f"  rank {rank:>4}: {scores[rank - 1]:.1f}")
    payload["rank_thresholds"] = {
        str(rank): float(scores[rank - 1])
        for rank in (1, 3, 10, 25, 50, 100, 200, 400)
    }

    print("\n=== 6. how big is the skill spread compared with run noise? ===")
    both = np.array([boards[second][t]["score"] for t in common])
    print(f"field sd of team scores on 08-14: {both.std():.1f}")
    print(f"sd of a team's own change over nine days: "
          f"{float(np.std([boards[second][t]['score'] - boards[first][t]['score'] for t in common])):.1f}")
    v22 = [OURS[k] for k in ("v22_a", "v22_b", "v22_c", "v22_d")]
    print(f"sd of our four byte-identical v22 runs: "
          f"{statistics.stdev(v22):.1f} (range {max(v22) - min(v22):.1f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
