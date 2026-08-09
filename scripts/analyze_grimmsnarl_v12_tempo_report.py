"""Read the cached tempo rows and answer the v12 design questions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROWS = [json.loads(line) for line in
        (Path(__file__).resolve().parents[1] / "experiments"
         / "grimmsnarl_ml_v12" / "tempo_rows.jsonl").read_text("utf-8").splitlines()
        if line.strip()]
OURS = {"v8", "v9", "v11", "v11_a", "v11_b"}


def wilson(k, n):
    if not n:
        return (0.0, 0.0)
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return round((c - m) / d, 3), round((c + m) / d, 3)


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    from math import comb
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def p(x):
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
    obs = p(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return round(sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * 1.0000001), 5)


def rate(rows):
    n = len(rows)
    k = sum(r["won"] for r in rows)
    return f"{k}-{n - k} ({k / n:.3f}) CI{wilson(k, n)}" if n else "n=0"


def mean_at(rows, key, ordinal):
    vals = [r[key].get(str(ordinal)) for r in rows if r[key].get(str(ordinal)) is not None]
    return sum(vals) / len(vals) if vals else None


def line(rows, key, label):
    cells = []
    for t in range(1, 7):
        v = mean_at(rows, key, t)
        cells.append(f"{v:5.2f}" if v is not None else "    -")
    return f"    {label:<22}" + " ".join(cells)


def block(rows, label):
    if not rows:
        print(f"  {label}: n=0")
        return
    print(f"  {label:<34} {rate(rows)}")
    print(line(rows, "bodies", "bodies at own turn"))
    print(line(rows, "grimm", "GrimmEx in play"))
    print(line(rows, "attacked", "attacked"))
    fa = [r["first_attack"] for r in rows if r["first_attack"]]
    never = sum(1 for r in rows if not r["first_attack"])
    print(f"    first attack own-turn  mean={sum(fa)/len(fa):.2f}"
          f"  never attacked={never}/{len(rows)}" if fa else "    never attacked")


field = [r for r in ROWS if r["tag"] == "field"]
ours = [r for r in ROWS if r["tag"] in OURS]

print("=" * 78)
print("1. TURN ORDER, same 60 cards")
for name, pool in (("FIELD", field), ("OURS(v8+v9+v11*)", ours)):
    print(f"\n{name}")
    block([r for r in pool if r["went_first"] is True], "going first")
    block([r for r in pool if r["went_first"] is False], "going second")

print("\n" + "=" * 78)
print("2. GOING SECOND, by opponent archetype (n>=25 in field)")
counts = defaultdict(int)
for r in field:
    counts[r["opponent"]] += 1
for opp in sorted(counts, key=lambda k: -counts[k]):
    fsec = [r for r in field if r["went_first"] is False and r["opponent"] == opp]
    osec = [r for r in ours if r["went_first"] is False and r["opponent"] == opp]
    if len(fsec) < 25:
        continue
    fw = sum(r["won"] for r in fsec)
    ow = sum(r["won"] for r in osec)
    p = fisher(ow, len(osec) - ow, fw, len(fsec) - fw) if osec else None
    print(f"  {opp:<26} field {fw}-{len(fsec)-fw} ({fw/len(fsec):.3f})"
          f"   ours {ow}-{len(osec)-ow}"
          f" ({ow/len(osec):.3f})" if osec else
          f"  {opp:<26} field {fw}-{len(fsec)-fw} ({fw/len(fsec):.3f})   ours n=0")
    if osec:
        print(f"    {'':<24} fisher p={p}")

print("\n" + "=" * 78)
print("3. Does early board width predict winning, inside the FIELD?")
for first in (True, False):
    pool = [r for r in field if r["went_first"] is first]
    label = "first" if first else "second"
    for t in (1, 2, 3, 4):
        w = [r["bodies"][str(t)] for r in pool if r["won"] and r["bodies"].get(str(t)) is not None]
        l = [r["bodies"][str(t)] for r in pool if not r["won"] and r["bodies"].get(str(t)) is not None]
        if w and l:
            print(f"  {label:<7} own turn {t}: win {sum(w)/len(w):5.2f} (n={len(w):4d})"
                  f"   loss {sum(l)/len(l):5.2f} (n={len(l):4d})"
                  f"   delta {sum(w)/len(w) - sum(l)/len(l):+.2f}")

print("\n" + "=" * 78)
print("4. Board width beyond 3 bodies: field win rate by bodies at own turn 3")
for first in (True, False):
    pool = [r for r in field if r["went_first"] is first and r["bodies"].get("3") is not None]
    print(f"  going {'first' if first else 'second'}")
    buckets = defaultdict(list)
    for r in pool:
        buckets[min(int(r["bodies"]["3"]), 6)].append(r["won"])
    for b in sorted(buckets):
        v = buckets[b]
        if len(v) >= 20:
            print(f"    {b} bodies: {sum(v)}-{len(v)-sum(v)} ({sum(v)/len(v):.3f})"
                  f" CI{wilson(sum(v), len(v))}  n={len(v)}")

print("\n" + "=" * 78)
print("5. First attack timing (own-turn ordinal), field vs ours, going second")
for name, pool in (("field", field), ("ours", ours)):
    sec = [r for r in pool if r["went_first"] is False]
    fa = [r["first_attack"] for r in sec if r["first_attack"]]
    won = [r["first_attack"] for r in sec if r["first_attack"] and r["won"]]
    lost = [r["first_attack"] for r in sec if r["first_attack"] and not r["won"]]
    print(f"  {name}: mean={sum(fa)/len(fa):.2f} n={len(fa)}"
          f"  | wins {sum(won)/len(won):.2f}  losses {sum(lost)/len(lost):.2f}")

print("\n" + "=" * 78)
print("6. Our versions, going second only")
for tag in ("v8", "v9", "v11_a", "v11_b", "v11"):
    pool = [r for r in ROWS if r["tag"] == tag and r["went_first"] is False]
    if pool:
        print(f"  {tag:<6} {rate(pool)}   first attack "
              f"{sum(r['first_attack'] for r in pool if r['first_attack']) / max(1, sum(1 for r in pool if r['first_attack'])):.2f}")
