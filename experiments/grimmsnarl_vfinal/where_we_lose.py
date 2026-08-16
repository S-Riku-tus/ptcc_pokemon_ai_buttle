"""Where the 552 stored games are actually lost, controlling for the draw.

Rating is a fixed point at ``mean(opponent) + 400*log10(w/(1-w))``, so the only
thing worth optimising is the win rate against the band we are paired into.
This splits that win rate every way the stored table allows and reports each
cell's Elo contribution: (share of games) x (Elo the cell is below the mean).
"""

from __future__ import annotations

import collections
import csv
import math
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV = ROOT / "experiments" / "grimmsnarl_endgame_20260816" / "version_games.csv"
rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))


def f(row, key, default=None):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def i(row, key, default=None):
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return default


def elo(w, n):
    if not n:
        return None
    p = min(max(w / n, 1e-3), 1 - 1e-3)
    return 400 * math.log10(p / (1 - p))


def table(title, keyfn, subset=None, minimum=8):
    data = subset if subset is not None else rows
    groups = collections.defaultdict(list)
    for row in data:
        key = keyfn(row)
        if key is not None:
            groups[key].append(row)
    total = len(data)
    base_w = sum(int(r["won"]) for r in data)
    print(f"\n=== {title}   (base {base_w}/{total} = {base_w/total:.3f}) ===")
    print(f"{'cell':34} {'n':>5} {'share':>7} {'wr':>7} {'oppR':>7} {'implied':>8}")
    out = []
    for key, group in groups.items():
        n = len(group)
        if n < minimum:
            continue
        w = sum(int(r["won"]) for r in group)
        opp = [f(r, "opponent_rating") for r in group if f(r, "opponent_rating")]
        mean_opp = sum(opp) / len(opp) if opp else 0.0
        out.append((key, n, w / n, mean_opp, mean_opp + (elo(w, n) or 0)))
    for key, n, wr, opp, implied in sorted(out, key=lambda t: t[4]):
        print(f"{str(key)[:34]:34} {n:>5} {n/total:7.1%} {wr:7.3f} "
              f"{opp:7.0f} {implied:8.0f}")


strong = [r for r in rows if (f(r, "opponent_rating") or 0) >= 950]
v22 = [r for r in rows if r["version"].startswith("v22")]

print(f"total stored games: {len(rows)}")
table("opponent family (all versions)", lambda r: r["opponent_family"], minimum=5)
table("opponent family, opponents 950+", lambda r: r["opponent_family"],
      subset=strong, minimum=5)
table("turn order", lambda r: r["went_first"])
table("turn order, opponents 950+", lambda r: r["went_first"], subset=strong)
table("version", lambda r: r["version"], minimum=20)


def bucket(value, edges):
    if value is None:
        return None
    for edge in edges:
        if value <= edge:
            return f"<={edge}"
    return f">{edges[-1]}"


table("our shadow attacks", lambda r: bucket(i(r, "shadow_attacks"), [0, 1, 2, 3, 4, 5, 6]))
table("own turn of first Shadow Bullet",
      lambda r: bucket(i(r, "own_first_shadow_turn"), [1, 2, 3, 4, 5]))
table("Grimmsnarl evolutions", lambda r: bucket(i(r, "grim_evolutions"), [0, 1, 2, 3]))
table("game length in our turns", lambda r: bucket(i(r, "our_turns"), [3, 4, 5, 6, 7, 8, 10]))
table("bodies left at the end", lambda r: bucket(i(r, "our_bodies_left"), [0, 1, 2, 3, 4, 5]))
table("Boss's Orders played", lambda r: bucket(i(r, "bosses"), [0, 1, 2]))
table("Adrena-Brain uses", lambda r: bucket(i(r, "adrena_brains"), [0, 1, 2, 3, 5]))
table("Unfair Stamps", lambda r: bucket(i(r, "stamps"), [0, 1]))
table("Petrel/Lillie count", lambda r: bucket(i(r, "lillies"), [0, 1, 2, 3]))
table("our overage used (s)", lambda r: bucket(f(r, "our_overage_used"), [2, 5, 10, 20, 40]))

print("\n=== loss anatomy: how the losses end ===")
losses = [r for r in rows if r["won"] == "0"]
wins = [r for r in rows if r["won"] == "1"]
for label, group in (("losses", losses), ("wins", wins)):
    boardout = sum(1 for r in group if r["board_out"] == "1")
    deckout = sum(1 for r in group if r["deck_out"] == "1")
    prizes = [i(r, "our_prize_left") for r in group if i(r, "our_prize_left") is not None]
    opp_prizes = [i(r, "opp_prize_left") for r in group
                  if i(r, "opp_prize_left") is not None]
    turns = [i(r, "our_turns") for r in group if i(r, "our_turns") is not None]
    print(f"  {label:8} n={len(group):3}  board_out={boardout:3} deck_out={deckout:3} "
          f"our_prizes_left={sum(prizes)/len(prizes):.2f} "
          f"opp_prizes_left={sum(opp_prizes)/len(opp_prizes):.2f} "
          f"our_turns={sum(turns)/len(turns):.2f}")

print("\n=== how close are the losses? opponent prizes left when we lose ===")
dist = collections.Counter(i(r, "opp_prize_left") for r in losses)
for key in sorted(k for k in dist if k is not None):
    print(f"   opp had {key} prizes left: {dist[key]:3}  ({dist[key]/len(losses):5.1%})")
