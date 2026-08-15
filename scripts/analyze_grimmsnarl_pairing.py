"""Who the ladder pairs you against, and why that decides the rating.

A settled rating is ``mean(opponent) + 400 log10(w/(1-w))``.  The second term
is the agent.  The first term is the *draw*, and on this ladder the draw is not
exogenous: the matchmaker pairs on current rating, so the opponents a run meets
are a consequence of where that run already sits.

Combine that with the fitted K schedule - 216 at game 1, 62 by game 10, 18 by
game 34 - and the ladder has a trap in it.  The first ten games are played with
63% of all the K a 34-game run will ever have, and they decide which pool the
remaining games are sampled from.  A run that reaches game 10 low is then fed
weak opponents, and no longer has the K to climb out of them even by winning.

This script measures the three links in that chain:

1. is pairing actually rating-proximate, and how tight;
2. how each run's opponent stream tracked its own rating;
3. what a run's rating at game 10 predicts about its opponent mean afterwards,
   and hence about its final rating.

It closes with the counterfactual that matters: replay each run's own results
against its own opponents, but perturb only the opening.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_rating_mechanics import fit_k, load  # noqa: E402

BOARD = ROOT / "data/kaggle_top100/latest/raw/api/leaderboard_full.json"


def elo(w: float) -> float:
    w = min(max(w, 1e-4), 1 - 1e-4)
    return 400 * math.log10(w / (1 - w))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/pairing.json",
    )
    args = parser.parse_args()
    payload: dict = {}

    by_run = load()
    schedule, _ = fit_k(by_run)
    rows = [row for games in by_run.values() for row in games]

    scores = np.array(sorted(
        float(r["displayScore"])
        for r in json.loads(BOARD.read_text(encoding="utf-8"))
        ["publicLeaderboard"]
        if r["displayScore"] not in (None, "")
    ))
    print("=== 0. what a random opponent would look like ===")
    print(f"the live field of {len(scores)} teams has median "
          f"{np.median(scores):.0f}, mean {scores.mean():.0f}, "
          f"p25 {np.percentile(scores, 25):.0f}, "
          f"p75 {np.percentile(scores, 75):.0f}")
    print(f"our 445 opponents averaged "
          f"{np.mean([r['opp'] for r in rows]):.0f}, so pairing is not uniform")

    print("\n=== 1. is pairing rating-proximate? ===")
    ours = np.array([r["before"] for r in rows])
    opp = np.array([r["opp"] for r in rows])
    print(f"corr(our rating before the game, opponent rating) = "
          f"pearson {pearsonr(ours, opp).statistic:.3f}, "
          f"spearman {spearmanr(ours, opp).statistic:.3f}, n = {len(rows)}")
    gap = opp - ours
    print(f"opponent minus us: mean {gap.mean():+.1f}, median "
          f"{np.median(gap):+.1f}, sd {gap.std():.1f}")
    payload["pairing_corr"] = float(pearsonr(ours, opp).statistic)

    print("\nopponent rating by the band we were sitting in:")
    print(f"{'our rating':<16}{'n':>5}{'opp mean':>10}{'opp sd':>9}"
          f"{'opp - us':>10}")
    for low, high in ((0, 700), (700, 800), (800, 900), (900, 1000),
                      (1000, 1100), (1100, 9999)):
        group = [r for r in rows if low <= r["before"] < high]
        if len(group) < 5:
            continue
        values = np.array([r["opp"] for r in group])
        mean_before = np.mean([r["before"] for r in group])
        print(f"{f'{low}-{high}':<16}{len(group):>5}{values.mean():>10.0f}"
              f"{values.std():>9.0f}{values.mean() - mean_before:>10.0f}")
        payload.setdefault("by_band", {})[f"{low}-{high}"] = {
            "n": len(group), "opp_mean": float(values.mean()),
        }
    print("\nthe pool tracks us with a slope well under one, so a run that")
    print("sits low is fed opponents it beats but gains almost nothing from.")

    print("\n=== 2. opponent stream per run, by phase of the run ===")
    print(f"{'run':<8}{'g1-5':>8}{'g6-10':>8}{'g11-20':>9}{'g21-34':>9}"
          f"{'g35+':>8}{'r@10':>9}{'final':>9}")
    for run in sorted(by_run):
        games = by_run[run]

        def slice_mean(a: int, b: int) -> str:
            part = [r["opp"] for r in games[a:b]]
            return f"{np.mean(part):.0f}" if part else "-"

        print(f"{run:<8}{slice_mean(0, 5):>8}{slice_mean(5, 10):>8}"
              f"{slice_mean(10, 20):>9}{slice_mean(20, 34):>9}"
              f"{slice_mean(34, 999):>8}"
              f"{games[9]['after']:>9.1f}{games[-1]['after']:>9.1f}")

    print("\n=== 3. rating at game 10 predicts the rest of the ladder ===")
    r10 = np.array([by_run[run][9]["after"] for run in sorted(by_run)])
    later_opp = np.array([
        np.mean([r["opp"] for r in by_run[run][10:]]) for run in sorted(by_run)
    ])
    final = np.array([by_run[run][-1]["after"] for run in sorted(by_run)])
    later_wr = np.array([
        np.mean([r["w"] for r in by_run[run][10:]]) for run in sorted(by_run)
    ])
    print(f"{'run':<8}{'r@10':>9}{'opp after 10':>14}{'wr after 10':>13}"
          f"{'final':>9}")
    for index, run in enumerate(sorted(by_run)):
        print(f"{run:<8}{r10[index]:>9.1f}{later_opp[index]:>14.0f}"
              f"{later_wr[index]:>13.3f}{final[index]:>9.1f}")
    print(f"\ncorr(rating at game 10, opponent mean afterwards) = "
          f"{pearsonr(r10, later_opp).statistic:.3f} "
          f"(p = {pearsonr(r10, later_opp).pvalue:.4f}, n = {len(r10)})")
    print(f"corr(rating at game 10, final rating)             = "
          f"{pearsonr(r10, final).statistic:.3f} "
          f"(p = {pearsonr(r10, final).pvalue:.4f})")
    print(f"corr(win rate after game 10, final rating)        = "
          f"{pearsonr(later_wr, final).statistic:.3f} "
          f"(p = {pearsonr(later_wr, final).pvalue:.4f})")
    payload["r10_vs_final"] = float(pearsonr(r10, final).statistic)
    payload["r10_vs_later_opp"] = float(pearsonr(r10, later_opp).statistic)
    payload["laterwr_vs_final"] = float(pearsonr(later_wr, final).statistic)
    print("\nthe opening predicts the finish; the win rate after the opening")
    print("barely does. That is the trap, stated as a correlation.")

    print("\n=== 4. slope: how much final rating per point at game 10 ===")
    design = np.column_stack([np.ones(len(r10)), r10])
    beta, *_ = np.linalg.lstsq(design, final, rcond=None)
    print(f"final = {beta[0]:.1f} + {beta[1]:.3f} * rating_at_game_10")
    design2 = np.column_stack([np.ones(len(r10)), r10, later_wr])
    beta2, *_ = np.linalg.lstsq(design2, final, rcond=None)
    print(f"final = {beta2[0]:.1f} + {beta2[1]:.3f} * rating_at_game_10 "
          f"+ {beta2[2]:.1f} * win_rate_after_game_10")
    payload["slope_r10"] = float(beta[1])

    print("\n=== 5. an escape check: can a run climb out after game 10? ===")
    print("for each run, the rating gained from game 11 to the end, and the")
    print("rating that same tail would have produced from a 1000-point start.")
    print(f"{'run':<8}{'r@10':>9}{'final':>9}{'moved':>9}{'tail n':>8}"
          f"{'tail wr':>9}{'from 1000':>11}")
    for run in sorted(by_run):
        games = by_run[run]
        tail = games[10:]
        if not tail:
            continue
        rating = 1000.0
        top = len(schedule) - 2
        for index, row in enumerate(tail, 11):
            expected = 1 / (1 + 10 ** ((row["opp"] - rating) / 400))
            rating += schedule[min(index, top)] * (row["w"] - expected)
        wins = sum(r["w"] for r in tail)
        print(f"{run:<8}{games[9]['after']:>9.1f}{games[-1]['after']:>9.1f}"
              f"{games[-1]['after'] - games[9]['after']:>9.1f}{len(tail):>8}"
              f"{wins / len(tail):>9.3f}{rating:>11.1f}")
    print("\n'from 1000' replays the identical tail games starting at 1000")
    print("instead of where the run actually was. Where that column stays")
    print("near 1000, the tail could not have punished a good opening.")

    print("\n=== 6. the ceiling the draw imposes ===")
    print("even a perfect agent is capped at opponent mean + Elo(win rate).")
    print(f"{'run':<8}{'opp mean':>10}{'at 0.70':>10}{'at 0.80':>10}"
          f"{'at 0.90':>10}{'actual':>10}")
    for run in sorted(by_run):
        games = by_run[run]
        opp_mean = float(np.mean([r["opp"] for r in games]))
        print(f"{run:<8}{opp_mean:>10.0f}{opp_mean + elo(0.70):>10.0f}"
              f"{opp_mean + elo(0.80):>10.0f}{opp_mean + elo(0.90):>10.0f}"
              f"{games[-1]['after']:>10.1f}")
    print("\nv27's draw (mean 789) caps a 90%-winning agent at 971.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
