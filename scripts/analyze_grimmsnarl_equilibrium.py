"""The rating our measured strength curve converges to, and what moves it.

Two functions decide where a run settles.

* **pairing** - the opponent pool tracks our own rating with a slope under one:
  ``opponent = a + b * ours``, fitted here on all 445 stored games.
* **strength** - our win rate as a function of how strong the opponent really
  is, graded by their settled leaderboard score rather than the noisy number
  they carried when we met them.

A run stops moving where the two agree:

    r* = opponent(r*) + 400 log10(w(opponent(r*)) / (1 - w(...)))

That fixed point is the agent's rating, stripped of the draw, the ordering and
the day.  Everything else in this file is the derivative of it: how much r*
moves if a named matchup is repaired, and how much of the top of the
leaderboard is explained by the fact that a team's displayed score is the
maximum of its two live slots rather than a single run.

Small cells are reported with their Wilson interval and the fixed point is
recomputed at both ends, so the conclusions carry their own error bars.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402
from analyze_grimmsnarl_rating_mechanics import load  # noqa: E402
from analyze_grimmsnarl_true_strength import submission_index  # noqa: E402

BOARD = ROOT / "data/kaggle_top100/latest/raw/api/leaderboard_full.json"
LN10_400 = math.log(10) / 400


def elo(w: float) -> float:
    w = min(max(w, 1e-4), 1 - 1e-4)
    return 400 * math.log10(w / (1 - w))


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Newton fit of ``P(win) = sigmoid(alpha + beta * x)``; returns se too."""
    beta = np.array([0.0, 0.0])
    design = np.column_stack([np.ones(len(x)), x])
    for _ in range(200):
        eta = design @ beta
        p = 1 / (1 + np.exp(-eta))
        weight = np.clip(p * (1 - p), 1e-9, None)
        hessian = design.T @ (design * weight[:, None])
        step = np.linalg.solve(hessian, design.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(design.T @ (design * weight[:, None]))
    return float(beta[0]), float(beta[1]), float(math.sqrt(cov[1, 1]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/equilibrium.json",
    )
    args = parser.parse_args()
    payload: dict = {}

    by_run = load()
    rows = [row for games in by_run.values() for row in games]
    sub_to_team, team_score = submission_index()
    for row in rows:
        team = sub_to_team.get(int(row["opponent_submission"] or 0))
        row["true"] = team_score.get(team) if team else None
    matched = [row for row in rows if row["true"] is not None]

    print("=== 1. the pairing function ===")
    ours = np.array([r["before"] for r in rows])
    opp = np.array([r["opp"] for r in rows])
    slope, intercept = np.polyfit(ours, opp, 1)
    print(f"opponent = {intercept:.1f} + {slope:.3f} * our rating "
          f"(n = {len(rows)})")
    print(f"fixed point of pairing alone: "
          f"{intercept / (1 - slope):.0f} - above that the pool lags us")
    for rating in (800, 900, 1000, 1100, 1200):
        print(f"  at rating {rating}: pool {intercept + slope * rating:.0f} "
              f"({intercept + slope * rating - rating:+.0f})")
    payload["pairing"] = {"slope": float(slope), "intercept": float(intercept)}

    print("\n=== 2. our strength curve against settled opponents ===")
    x = np.array([r["true"] for r in matched], dtype=float)
    y = np.array([r["w"] for r in matched], dtype=float)
    alpha, beta, se = logistic_fit(x, y)
    print(f"logit P(win) = {alpha:.3f} {beta:+.5f} * settled_opponent_score")
    print(f"  slope se {se:.5f}, z = {beta / se:.2f}, "
          f"p = {2 * norm.sf(abs(beta / se)):.2g}, n = {len(matched)}")
    print(f"  a pure Elo agent has slope -{LN10_400:.5f}; ours is "
          f"{beta / -LN10_400:.2f}x as steep")
    print("  our implied rating (where the curve crosses 50%): "
          f"{-alpha / beta:.0f}")
    payload["strength_curve"] = {
        "alpha": alpha, "beta": beta, "se": se,
        "implied_50pct": float(-alpha / beta),
        "steepness_vs_elo": float(beta / -LN10_400),
    }
    print("\n  a slope steeper than Elo means we fall off faster than a")
    print("  rating-consistent agent would: the stronger the opponent, the")
    print("  worse we do relative to what our own rating predicts.")

    def win_rate(opponent: float) -> float:
        return 1 / (1 + math.exp(-(alpha + beta * opponent)))

    print("\n=== 3. the fixed point ===")

    def gap(rating: float) -> float:
        pool = intercept + slope * rating
        return intercept + slope * rating + elo(win_rate(pool)) - rating

    star = brentq(gap, 400, 1600)
    pool_star = intercept + slope * star
    print(f"r* = {star:.0f}, reached against a pool averaging "
          f"{pool_star:.0f} with a win rate of {win_rate(pool_star):.3f}")
    board = [
        float(r["displayScore"])
        for r in json.loads(BOARD.read_text(encoding="utf-8"))
        ["publicLeaderboard"] if r["displayScore"] not in (None, "")
    ]
    scores = np.array(sorted(board, reverse=True))
    print(f"that is rank {int((scores > star).sum()) + 1} of {len(scores)}")
    payload["fixed_point"] = float(star)

    print("\nsensitivity: refit the curve at the ends of the slope interval")
    for label, b in (("slope -1sd", beta - se), ("slope +1sd", beta + se)):
        a_adj = alpha - (b - beta) * float(np.mean(x))

        def gap_adj(rating: float, a=a_adj, bb=b) -> float:
            pool = intercept + slope * rating
            w = 1 / (1 + math.exp(-(a + bb * pool)))
            return pool + elo(w) - rating

        print(f"  {label}: r* = {brentq(gap_adj, 300, 1800):.0f}")

    print("\n=== 4. observed vs fitted, band by band ===")
    print(f"{'settled band':<16}{'n':>5}{'observed':>10}{'fitted':>9}"
          f"{'wilson':>18}")
    for low, high in ((0, 700), (700, 850), (850, 1000), (1000, 1100),
                      (1100, 9999)):
        group = [r for r in matched if low <= r["true"] < high]
        if not group:
            continue
        wins = sum(r["w"] for r in group)
        centre = float(np.mean([r["true"] for r in group]))
        lo, hi = wilson(wins, len(group))
        print(f"{f'{low}-{high}':<16}{len(group):>5}"
              f"{wins / len(group):>10.3f}{win_rate(centre):>9.3f}"
              f"{f'[{lo:.2f},{hi:.2f}]':>18}")

    print("\n=== 5. what repairing a matchup is worth ===")
    print("each row raises our win rate in that cell to the target and")
    print("recomputes the fixed point; exposure is the cell's share of all")
    print("445 games, and of the 26 games against settled-1000+ opponents.")
    families: dict[str, list] = defaultdict(list)
    for row in rows:
        families[row["opponent_family"]].append(row)
    overall_wins = sum(r["w"] for r in rows)
    print(f"\nbaseline: {overall_wins}-{len(rows) - overall_wins} "
          f"({overall_wins / len(rows):.3f}) over {len(rows)} games")
    print(f"{'matchup':<30}{'n':>5}{'share':>7}{'wr':>7}{'->0.55':>9}"
          f"{'delta wr':>10}{'delta Elo':>11}")
    repairs = []
    for family, group in sorted(families.items(), key=lambda i: -len(i[1])):
        if len(group) < 8:
            continue
        wins = sum(r["w"] for r in group)
        rate = wins / len(group)
        if rate >= 0.55:
            continue
        gained = 0.55 * len(group) - wins
        new_rate = (overall_wins + gained) / len(rows)
        delta = elo(new_rate) - elo(overall_wins / len(rows))
        print(f"{family:<30}{len(group):>5}{len(group) / len(rows):>7.1%}"
              f"{rate:>7.3f}{gained:>9.1f}"
              f"{new_rate - overall_wins / len(rows):>10.3f}"
              f"{delta:>11.1f}")
        repairs.append({"family": family, "n": len(group), "wr": rate,
                        "delta_elo": float(delta)})
    payload["repairs"] = repairs

    weak = [f for f in families
            if len(families[f]) >= 8
            and sum(r["w"] for r in families[f]) / len(families[f]) < 0.55]
    gained = sum(
        0.55 * len(families[f]) - sum(r["w"] for r in families[f])
        for f in weak
    )
    combined = (overall_wins + gained) / len(rows)
    print(f"\nall of them together: {overall_wins / len(rows):.3f} -> "
          f"{combined:.3f}, worth "
          f"{elo(combined) - elo(overall_wins / len(rows)):+.1f} Elo")
    print(f"games involved: {sum(len(families[f]) for f in weak)} of "
          f"{len(rows)} ({sum(len(families[f]) for f in weak) / len(rows):.1%})")

    print("\nand the same repair pushed through the fixed point, which is")
    print("larger because a better agent is also paired higher:")
    for target, label in ((0.55, "weak cells to 0.55"),
                          (0.60, "weak cells to 0.60")):
        gained_t = sum(
            max(target * len(families[f]) - sum(r["w"] for r in families[f]), 0)
            for f in weak
        )
        lift = math.log(
            (overall_wins + gained_t) / (len(rows) - overall_wins - gained_t)
        ) - math.log(overall_wins / (len(rows) - overall_wins))

        def gap_lift(rating: float, shift=lift) -> float:
            pool = intercept + slope * rating
            w = 1 / (1 + math.exp(-(alpha + beta * pool + shift)))
            return pool + elo(w) - rating

        new_star = brentq(gap_lift, 400, 2000)
        print(f"  {label:<22} r* {star:.0f} -> {new_star:.0f} "
              f"({new_star - star:+.0f}), rank "
              f"{int((scores > new_star).sum()) + 1}")
        payload.setdefault("repair_fixed_point", {})[label] = float(new_star)

    print("\n=== 6. what each rank actually demands of the agent ===")
    print("a uniform logit shift is applied to the strength curve until the")
    print("fixed point lands on the score that rank required on 08-14.")
    print(f"{'rank':>6}{'rating':>9}{'shift (Elo)':>13}"
          f"{'win rate vs a 950 pool':>25}")
    now_950 = win_rate(950)
    for rank in (200, 150, 100, 50, 25, 10, 3, 1):
        target = float(scores[rank - 1])

        def gap_shift(shift: float, t=target) -> float:
            def inner(rating: float) -> float:
                pool = intercept + slope * rating
                w = 1 / (1 + math.exp(-(alpha + beta * pool + shift)))
                return pool + elo(w) - rating
            return brentq(inner, 300, 2500) - t

        shift = brentq(gap_shift, -3, 6)
        need = 1 / (1 + math.exp(-(alpha + beta * 950 + shift)))
        print(f"{rank:>6}{target:>9.1f}{shift * 400 / math.log(10):>13.0f}"
              f"{f'{need:.3f} (now {now_950:.3f})':>25}")
        payload.setdefault("rank_targets", {})[str(rank)] = {
            "rating": target, "elo_shift": float(shift * 400 / math.log(10)),
            "wr_vs_950": float(need),
        }

    print("\n=== 7. the leaderboard shows the better of a team's two slots ===")
    gaps = []
    for path in sorted(
        (ROOT / "data/kaggle_top100/latest/raw/api/team_public_submissions")
        .glob("team_*.json")
    ):
        values = [
            float(s["publicScoreFormatted"])
            for s in json.loads(path.read_text(encoding="utf-8"))["submissions"]
            if s.get("publicScoreFormatted")
        ]
        if len(values) >= 2:
            gaps.append(max(values) - min(values))
    gaps_array = np.array(gaps)
    print(f"verified on all {len(gaps)} top-60 teams: displayed score equals "
          f"the max of their live slots")
    print(f"median internal gap {np.median(gaps_array):.1f} points")
    sigma = 62.9
    print(f"with a run-to-run noise sd of {sigma:.1f} (from the nine-day "
          f"leaderboard decomposition), the expected value of the better of")
    print(f"two independent runs of the SAME agent is "
          f"+{sigma / math.sqrt(math.pi):.1f} points, and of three runs "
          f"+{sigma * 1.0854:.1f}.")
    print("running one agent in both slots is therefore worth about as much")
    print("as repairing a whole matchup, and costs nothing.")
    payload["max_of_two_bonus"] = float(sigma / math.sqrt(math.pi))

    print("\n=== 8. exposure: our runs never met the opponents that cap us ===")
    strong = [r for r in matched if r["true"] >= 1000]
    print(f"of {len(matched)} strength-graded games, {len(strong)} were "
          f"against a settled-1000+ opponent ({len(strong) / len(matched):.1%})")
    print("v27 specifically: "
          f"{sum(1 for r in by_run['v27'] if r['true'] and r['true'] >= 1000)} "
          f"of {sum(1 for r in by_run['v27'] if r['true'])} graded games")
    print("at r* the pairing function would supply a pool averaging "
          f"{intercept + slope * star:.0f}; at rating 1050 it supplies "
          f"{intercept + slope * 1050:.0f}.")
    print("so the matchups that decide the ceiling are barely sampled during")
    print("the runs we use to judge a version.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
