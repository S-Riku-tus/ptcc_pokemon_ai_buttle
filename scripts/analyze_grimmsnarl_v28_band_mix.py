"""Which archetypes gate the rating, and what fixing them would be worth.

The pooled win rate hides the only thing that moves a Kaggle rating: the mix
of archetypes changes with opponent strength.  Three families - Ogerpon,
Mega Lopunny / Froslass and Hydrapple ex - are 7% of the field below 900 and
32% of it at 950+, which is exactly the band the rating converges on.

Reports the mix and our record per band, the per-family record inside the
950+ band, and the Elo that lifting those three cells to a coin flip would be
worth, pooled and inside the band.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_grimmsnarl_v27_vs_champions as champ  # noqa: E402

champ.GROUPS["v28"] = ("v28",)

HARD = {"Ogerpon", "Mega Lopunny / Froslass", "other: Hydrapple ex"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = champ.load(ROOT / "experiments/grimmsnarl_ml_v28/version_games.csv")

    for label, subset in (
        ("all", rows),
        ("opp < 900", [r for r in rows if (r["opponent_rating"] or 0) < 900]),
        ("opp 900-950",
         [r for r in rows if 900 <= (r["opponent_rating"] or 0) < 950]),
        ("opp >= 950", [r for r in rows if (r["opponent_rating"] or 0) >= 950]),
        ("opp >= 1000",
         [r for r in rows if (r["opponent_rating"] or 0) >= 1000]),
    ):
        counts = Counter(r["opponent_family"] for r in subset)
        hard = [r for r in subset if r["opponent_family"] in HARD]
        rest = [r for r in subset if r["opponent_family"] not in HARD]
        hw = sum(r["won"] for r in hard)
        rw = sum(r["won"] for r in rest)
        total_w = sum(r["won"] for r in subset)
        print(
            f"{label:<14} n={len(subset):<4} "
            f"wr {total_w / max(len(subset), 1):.3f}  "
            f"hard-3 share {len(hard) / max(len(subset), 1):.1%} "
            f"({hw}-{len(hard) - hw}, {hw / max(len(hard), 1):.3f})  "
            f"rest ({rw}-{len(rest) - rw}, {rw / max(len(rest), 1):.3f})"
        )
        print("   top families: " + ", ".join(
            f"{k} {v}" for k, v in counts.most_common(6)
        ))

    print("\nper family inside the >=950 band:")
    band = [r for r in rows if (r["opponent_rating"] or 0) >= 950]
    for family in sorted({r["opponent_family"] for r in band}):
        subset = [r for r in band if r["opponent_family"] == family]
        if len(subset) < 4:
            continue
        wins = sum(r["won"] for r in subset)
        print(f"  {family:<26} {wins:>3}-{len(subset) - wins:<3} "
              f"{wins / len(subset):.3f}  n={len(subset)}")

    print("\nwhat fixing the hard-3 to 0.500 would be worth overall:")
    hard = [r for r in rows if r["opponent_family"] in HARD]
    hw = sum(r["won"] for r in hard)
    total_w = sum(r["won"] for r in rows)
    old_rate = total_w / len(rows)
    new_rate = (total_w - hw + 0.5 * len(hard)) / len(rows)
    print(f"  pooled {old_rate:.4f} -> {new_rate:.4f}  "
          f"= {champ.elo(new_rate) - champ.elo(old_rate):+.1f} Elo")
    band_hard = [r for r in band if r["opponent_family"] in HARD]
    bw = sum(r["won"] for r in band_hard)
    band_w = sum(r["won"] for r in band)
    new_band = (band_w - bw + 0.5 * len(band_hard)) / len(band)
    print(f"  >=950  {band_w / len(band):.4f} -> {new_band:.4f}  "
          f"= {champ.elo(new_band) - champ.elo(band_w / len(band)):+.1f} Elo "
          f"(hard-3 is {len(band_hard)}/{len(band)} of that band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
