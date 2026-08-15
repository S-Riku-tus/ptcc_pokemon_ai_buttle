"""v22, v26 and v27 are one policy: pool them and split by day and seat.

``probe_grimmsnarl_v27_ladder_footprint.py`` reproduces v22's answer on 2747
of v27's 2755 stored decisions and 3043 of v26's 3051, so the three runs
differ by 16 decisions in 264 games.  Anything that separates their results is
therefore the field, the pairing draw or chance - not the policy.

Two splits are reported on that pool:

* by calendar day, with the implied strength ``opponent mean + Elo(win rate)``,
  which is what a Kaggle rating converges to;
* by turn order, which is randomised inside every episode and so is the only
  causally identified contrast available.  It is then cut by opponent band to
  separate "going first is worth more against weak opponents" from "the
  08-15 field punishes the second seat".
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

POOL = {"v22_a", "v22_b", "v22_c", "v22_d", "v26", "v27"}
LATE_FROM = "2026-08-15"


def elo(w):
    w = min(max(w, 1e-4), 1 - 1e-4)
    return 400 * math.log10(w / (1 - w))


rows = [
    r for r in csv.DictReader(
        (ROOT / "experiments/grimmsnarl_ml_v27/version_games.csv").open(
            encoding="utf-8-sig"
        )
    )
    if r["version"] in POOL and r["opponent_rating"]
]
for r in rows:
    r["rating"] = float(r["opponent_rating"])
    r["late"] = r["create_time"][:10] >= LATE_FROM
    r["w"] = int(r["won"])
by_day = defaultdict(list)
for r in rows:
    by_day[r["create_time"][:10]].append(r)

print("pooled v22-equivalent policy, by day")
print(f"{'day':<12}{'n':>5}{'record':>10}{'wr':>8}{'wilson':>18}"
      f"{'opp mean':>10}{'strength':>10}")
for day in sorted(by_day):
    group = by_day[day]
    wins = sum(int(r["won"]) for r in group)
    n = len(group)
    opp = sum(float(r["opponent_rating"]) for r in group) / n
    low, high = wilson(wins, n)
    print(f"{day:<12}{n:>5}{f'{wins}-{n - wins}':>10}{wins / n:>8.3f}"
          f"{f'[{low:.2f},{high:.2f}]':>18}{opp:>10.1f}"
          f"{opp + elo(wins / n):>10.1f}")

print("\nsame, restricted to opponents rated 700-900")
for day in sorted(by_day):
    group = [r for r in by_day[day] if 700 <= float(r["opponent_rating"]) < 900]
    if not group:
        continue
    wins = sum(int(r["won"]) for r in group)
    n = len(group)
    low, high = wilson(wins, n)
    print(f"{day:<12}{n:>5}{f'{wins}-{n - wins}':>10}{wins / n:>8.3f}"
          f"{f'[{low:.2f},{high:.2f}]':>18}")

print("\nsame, split by turn order")
for day in sorted(by_day):
    line = f"{day:<12}"
    for order in ("first", "second"):
        group = [r for r in by_day[day] if r["went_first"] == order]
        wins = sum(int(r["won"]) for r in group)
        line += f"  {order} {wins:>3}-{len(group) - wins:<3} {wins / len(group):.3f}"
    print(line)


def show(label, subset):
    first = [r for r in subset if r["went_first"] == "first"]
    second = [r for r in subset if r["went_first"] == "second"]
    if not first or not second:
        print(f"{label:<34} insufficient")
        return
    table = [
        [sum(r["w"] for r in first), len(first) - sum(r["w"] for r in first)],
        [sum(r["w"] for r in second), len(second) - sum(r["w"] for r in second)],
    ]
    a = table[0][0] / len(first)
    b = table[1][0] / len(second)
    print(
        f"{label:<34} first {table[0][0]:>3}-{table[0][1]:<3}({a:.3f})  "
        f"second {table[1][0]:>3}-{table[1][1]:<3}({b:.3f})  "
        f"diff {a - b:+.3f}  p={float(fisher_exact(table).pvalue):.4f}"
    )


print("--- by date, all opponent ratings ---")
show("08-13 (v22)", [r for r in rows if not r["late"]])
show("08-15 (v26+v27)", [r for r in rows if r["late"]])

print("\n--- restricted to opponents rated < 900 ---")
show("08-13, opp<900", [r for r in rows if not r["late"] and r["rating"] < 900])
show("08-15, opp<900", [r for r in rows if r["late"] and r["rating"] < 900])

print("\n--- restricted to opponents rated 700-900 ---")
show("08-13, 700-900", [r for r in rows if not r["late"] and 700 <= r["rating"] < 900])
show("08-15, 700-900", [r for r in rows if r["late"] and 700 <= r["rating"] < 900])

print("\n--- v22 alone, by opponent band (is the split rating-dependent?) ---")
early = [r for r in rows if not r["late"]]
for low, high in ((0, 700), (700, 800), (800, 900), (900, 1000), (1000, 9999)):
    show(f"08-13, opp {low}-{high}",
         [r for r in early if low <= r["rating"] < high])

print("\n--- excluding the mirror and Ogerpon, which move with the meta ---")
show("08-13, no mirror/Ogerpon", [
    r for r in rows if not r["late"]
    and r["opponent_family"] not in ("Grimmsnarl (mirror)", "Ogerpon")
])
show("08-15, no mirror/Ogerpon", [
    r for r in rows if r["late"]
    and r["opponent_family"] not in ("Grimmsnarl (mirror)", "Ogerpon")
])
