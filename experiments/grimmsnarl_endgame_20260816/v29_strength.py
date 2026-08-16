"""Opponent-adjusted strength per version, plus day-pooled field difficulty."""
import csv
import math
import collections
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "experiments/grimmsnarl_endgame_20260816/version_games.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))

def implied(wins, n, oppmean):
    if n == 0:
        return None
    w = min(max(wins / n, 1e-6), 1 - 1e-6)
    return oppmean + 400 * math.log10(w / (1 - w))

by = collections.defaultdict(list)
for r in rows:
    by[r["version"]].append(r)

print(f"{'ver':7s} {'n':>4s} {'rec':>8s} {'wr':>6s} {'oppmean':>8s} {'strength':>9s} {'final':>8s}")
for ver, rs in by.items():
    n = len(rs)
    wins = sum(int(r["won"]) for r in rs)
    opp = sum(float(r["opponent_rating"]) for r in rs) / n
    s = implied(wins, n, opp)
    final = float(rs[-1]["our_rating_after"])
    print(f"{ver:7s} {n:4d} {wins:3d}-{n-wins:<4d} {wins/n:6.3f} {opp:8.1f} {s:9.1f} {final:8.1f}")

print()
print("=== by calendar day (all versions pooled) ===")
day = collections.defaultdict(list)
for r in rows:
    day[r["create_time"][:10]].append(r)
for d in sorted(day):
    rs = day[d]
    n = len(rs)
    wins = sum(int(r["won"]) for r in rs)
    opp = sum(float(r["opponent_rating"]) for r in rs) / n
    weak = sum(1 for r in rs if float(r["opponent_rating"]) < 700) / n
    print(f"{d}  n={n:4d}  {wins:3d}-{n-wins:<3d} wr={wins/n:.3f}  oppmean={opp:7.1f}"
          f"  <700={weak:5.1%}  implied={implied(wins,n,opp):7.1f}")

print()
print("=== v29 by opponent band ===")
bands = [(0, 700), (700, 850), (850, 950), (950, 1050), (1050, 9999)]
for lo, hi in bands:
    rs = [r for r in rows if r["version"] == "v29"
          and lo <= float(r["opponent_rating"]) < hi]
    if not rs:
        continue
    n = len(rs)
    wins = sum(int(r["won"]) for r in rs)
    print(f"  {lo:4d}-{hi:<5d} n={n:3d}  {wins:2d}-{n-wins:<2d}  wr={wins/n:.3f}")

print()
print("=== v29 opponent family ===")
fam = collections.Counter()
famw = collections.Counter()
for r in rows:
    if r["version"] != "v29":
        continue
    fam[r["opponent_family"]] += 1
    famw[r["opponent_family"]] += int(r["won"])
for f, c in fam.most_common():
    print(f"  {f:34s} n={c:3d}  {famw[f]:2d}-{c-famw[f]:<2d}  wr={famw[f]/c:.3f}")
