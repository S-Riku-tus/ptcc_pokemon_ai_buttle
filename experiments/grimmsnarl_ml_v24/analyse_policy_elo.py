"""What is this policy worth, pooled, and how far is that from the same deck's ceiling?

A single submission's final number scatters 904-1020 across six runs of code
that is byte-identical apart from an 18-decision veto, so no one run is an
estimate of anything.  Pooling all 281 games and inverting the Elo expectation
against the observed opponent field gives one number with a usable interval:

    strength = mean(opponent rating) + 400 * log10(w / (1 - w))

The same inversion, applied to the win rate we would need against the >=950
band, converts "reach 1100" into a measurable target instead of a rating wish.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
PEERS = {"AlphaTCG (rank 22)": 1095.3, "NguyenThanhNhan (rank 27)": 1086.0}


def elo(w: float) -> float:
    w = min(max(w, 1e-4), 1 - 1e-4)
    return 400 * math.log10(w / (1 - w))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [
        r for r in csv.DictReader(GAMES.open(encoding="utf-8-sig"))
        if r["version"].startswith(("v22", "v24")) and r["opponent_rating"]
    ]
    for label, pool in (
        ("all 281 pooled games", rows),
        ("v22 only", [r for r in rows if r["version"].startswith("v22")]),
        ("v24 only", [r for r in rows if r["version"].startswith("v24")]),
        ("opponents >= 950", [r for r in rows if float(r["opponent_rating"]) >= 950]),
    ):
        n = len(pool)
        wins = sum(1 for r in pool if r["won"] == "True")
        opp = sum(float(r["opponent_rating"]) for r in pool) / n
        low, high = wilson(wins, n)
        print(f"{label:<24} n={n:>3}  {wins}-{n - wins}  {wins / n:.4f}  "
              f"opp {opp:6.1f}  ->  strength {opp + elo(wins / n):7.1f} "
              f"[{opp + elo(low):.0f}, {opp + elo(high):.0f}]")
    print()

    strong = [r for r in rows if float(r["opponent_rating"]) >= 950]
    opp = sum(float(r["opponent_rating"]) for r in strong) / len(strong)
    wins = sum(1 for r in strong if r["won"] == "True")
    print(f"the >=950 band is where the rating settles: mean opponent {opp:.1f}, "
          f"our rate {wins / len(strong):.3f}")
    for peer, rating in PEERS.items():
        needed = 1 / (1 + 10 ** ((opp - rating) / 400))
        print(f"  to sit where {peer} sits ({rating:.0f}) we would need "
              f"{needed:.3f} in that band; we are at {wins / len(strong):.3f} "
              f"({(needed - wins / len(strong)) * len(strong):+.1f} games out of "
              f"{len(strong)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
