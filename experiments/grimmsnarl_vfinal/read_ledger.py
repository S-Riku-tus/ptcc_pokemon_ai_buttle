"""Test the three board-reading damage formulas against the stored ledger."""

from __future__ import annotations

import collections
import csv
import pathlib
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = pathlib.Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "attack_ledger.csv").open(encoding="utf-8")))


def num(row, key, default=0):
    value = row.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


print(f"attacks aimed at us: {len(rows)}\n")

print("=== attacks that most often KO us, by opponent card ===")
counter = collections.Counter()
kos = collections.Counter()
for row in rows:
    key = (row["attacker_name"], row["attack_name"])
    counter[key] += 1
    kos[key] += num(row, "ko")
print(f"{'attacker':28} {'attack':26} {'n':>4} {'ko':>4} {'ko%':>6} {'meandmg':>8}")
damage_by = collections.defaultdict(list)
for row in rows:
    if row["damage"] != "":
        damage_by[(row["attacker_name"], row["attack_name"])].append(num(row, "damage"))
for key, n in counter.most_common(22):
    dmgs = damage_by.get(key) or []
    mean = f"{statistics.mean(dmgs):8.0f}" if dmgs else "       -"
    print(f"{key[0][:28]:28} {key[1][:26]:26} {n:>4} {kos[key]:>4} "
          f"{kos[key] / n:6.1%} {mean}")

GRIM = "Marnie's Grimmsnarl ex"

print("\n=== H1  Myriad Leaf Shower vs our Active Grimmsnarl ex ===")
print("     predicted = (30 + 30*(our_E + their_E)) * 2 for Grass Weakness")
sub = [
    r for r in rows
    if r["attack_name"] == "Myriad Leaf Shower" and r["our_active_name"] == GRIM
]
print(f"{'ourE':>5} {'theirE':>7} {'pred':>6} {'actual':>7} {'ko':>3} {'ourHP':>6} {'n':>3}")
buckets = collections.defaultdict(list)
for r in sub:
    ours, theirs = num(r, "our_active_energy"), num(r, "their_active_energy")
    pred = (30 + 30 * (ours + theirs)) * 2
    buckets[(ours, theirs)].append((pred, r["damage"], num(r, "ko"), num(r, "our_active_hp")))
for (ours, theirs), items in sorted(buckets.items()):
    actual = [i[1] for i in items if i[1] != ""]
    mean = f"{statistics.mean(int(a) for a in actual):7.0f}" if actual else "      -"
    print(f"{ours:>5} {theirs:>7} {items[0][0]:>6} {mean} "
          f"{sum(i[2] for i in items):>3} {statistics.mean(i[3] for i in items):6.0f} "
          f"{len(items):>3}")
if sub:
    e = [num(r, "our_active_energy") for r in sub]
    print(f"  our Active energy when it lands: mean {statistics.mean(e):.2f} "
          f"median {statistics.median(e)}  n={len(sub)}")
    survivable = sum(
        1 for r in sub
        if (30 + 30 * (2 + num(r, "their_active_energy"))) * 2 < num(r, "our_active_hp")
        and (30 + 30 * (num(r, "our_active_energy") + num(r, "their_active_energy"))) * 2
        >= num(r, "our_active_hp")
    )
    print(f"  KOs that a 2-Energy Active would have SURVIVED: {survivable} / {len(sub)}")

print("\n=== H2  Resentful Refrain vs anything of ours ===")
print("     predicted = 50 * our hand size")
sub = [r for r in rows if r["attack_name"] == "Resentful Refrain"]
hands = [num(r, "our_hand") for r in sub]
if sub:
    print(f"  n={len(sub)}  our hand size mean {statistics.mean(hands):.2f} "
          f"median {statistics.median(hands)}  max {max(hands)}")
    print(f"{'hand':>5} {'pred':>6} {'actual':>7} {'ko':>3} {'ourHP':>6} {'n':>3}")
    b2 = collections.defaultdict(list)
    for r in sub:
        b2[num(r, "our_hand")].append(r)
    for hand, items in sorted(b2.items()):
        actual = [num(i, "damage") for i in items if i["damage"] != ""]
        mean = f"{statistics.mean(actual):7.0f}" if actual else "      -"
        print(f"{hand:>5} {50 * hand:>6} {mean} {sum(num(i,'ko') for i in items):>3} "
              f"{statistics.mean(num(i,'our_active_hp') for i in items):6.0f} {len(items):>3}")
    saved = sum(
        1 for r in sub
        if num(r, "ko") and 50 * max(0, num(r, "our_hand") - 2) < num(r, "our_active_hp")
    )
    print(f"  KOs that 2 fewer cards in hand would have survived: {saved} / "
          f"{sum(num(r,'ko') for r in sub)} KOs")

print("\n=== H3  Syrup Storm (Hydrapple ex) ===")
sub = [r for r in rows if r["attack_name"] == "Syrup Storm"]
if sub:
    g = [num(r, "their_grass_energy") for r in sub]
    print(f"  n={len(sub)}  their total {{G}} mean {statistics.mean(g):.2f} "
          f"max {max(g)}  KOs {sum(num(r,'ko') for r in sub)}")
    for r in sub[:12]:
        print(f"   turn {r['turn']:>3} theirG={num(r,'their_grass_energy'):>2} "
              f"pred={(30 + 30 * num(r,'their_grass_energy')) * 2:>4} "
              f"actual={r['damage']:>4} ourHP={r['our_active_hp']:>4} "
              f"target={r['our_active_name']}")

print("\n=== H4  what the Slowking / Kangaskhan family actually does ===")
sub = [r for r in rows if r["opp_family"] == "other: Conkeldurr"]
for r in sub:
    print(f"  ep {r['episode']} turn {r['turn']:>3} {r['attacker_name']:24} "
          f"{r['attack_name']:20} dmg={r['damage']:>4} ko={r['ko']} "
          f"target={r['our_active_name']:24} hp={r['our_active_hp']}")
