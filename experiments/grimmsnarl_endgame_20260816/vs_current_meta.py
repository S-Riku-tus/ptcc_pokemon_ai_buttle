"""Our all-version record against the families that make up the CURRENT top
meta, weighted by how much of that meta each family is.

The ladder pool we are actually paired with is not the top meta, so a good
pooled win rate can coexist with a bad expected result against the board.
"""
import csv
import collections
import math

rows = list(csv.DictReader(open("experiments/grimmsnarl_endgame_20260816/version_games.csv",
                                encoding="utf-8-sig")))

# Share of 60-card decks observed among top-35 teams' latest games, 2026-08-16.
CURRENT_META = {
    "Dragapult": 0.242,
    "other: Conkeldurr": 0.210,
    "other: Hydrapple ex": 0.177,
    "Mega Lopunny / Froslass": 0.129,
    "Alakazam": 0.081,
    "Mega Lucario": 0.081,
    "Kangaskhan / Crustle": 0.048,
    "other: Team Rocket's Mewtwo ex": 0.016,
    "other: Arboliva ex": 0.016,
}

rec = collections.Counter()
games = collections.Counter()
for r in rows:
    f = r["opponent_family"]
    games[f] += 1
    rec[f] += int(r["won"])

print("=== our record against each family of the CURRENT top meta ===")
print(f"{'family':34s} {'meta%':>6s} {'n':>4s} {'record':>9s} {'wr':>6s}")
cov = 0.0
exp = 0.0
for f, share in sorted(CURRENT_META.items(), key=lambda x: -x[1]):
    n = games.get(f, 0)
    w = rec.get(f, 0)
    wr = w / n if n else None
    txt = f"{w}-{n-w}" if n else "-"
    print(f"{f:34s} {share:6.1%} {n:4d} {txt:>9s} "
          f"{(f'{wr:.3f}' if wr is not None else 'UNSEEN'):>6s}")
    if n >= 5:
        cov += share
        exp += share * wr
print()
print(f"covered share of the current meta: {cov:.1%}")
print(f"expected win rate on the covered part: {exp/cov:.3f}")
w = min(max(exp / cov, 1e-6), 1 - 1e-6)
print(f"  -> against a 1050-rated top-50 field that is worth "
      f"{1050 + 400*math.log10(w/(1-w)):.0f}")

print()
print("=== what we actually met on the ladder (all 552 games) ===")
tot = sum(games.values())
for f, n in games.most_common(12):
    print(f"  {f:34s} n={n:3d} {n/tot:6.1%}  "
          f"{rec[f]}-{n-rec[f]}  wr={rec[f]/n:.3f}")
