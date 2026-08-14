"""What decides the mirror, our biggest cell above 950 and our worst (0.452)?

The 194-game v22 pool retired every absolute tempo count: own_first_shadow_turn,
own_first_ready_turn, Stamp, Boss, Rare Candy and Grimmsnarl evolutions are all
measured nulls.  But an absolute turn number is the wrong instrument in a
mirror, where both seats run the same 60 and the same clock.  What can still
matter is the *relative* one: whether we started attacking before or after the
opponent did, and by how much.

This tests the differential directly, inside the mirror, controlled for
opponent rating and turn order, and separately for the >=950 band where the
rating is actually decided.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
ELO = 400.0 / math.log(10.0)


def fnum(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def block(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "n=0"
    wins = sum(1 for r in rows if r["won"])
    low, high = wilson(wins, n)
    return (f"n={n:>3} {wins:>3}-{n - wins:<3} {wins / n:.3f} "
            f"[{low:.3f},{high:.3f}]")


def fit(rows: list[dict], name: str, value: Callable[[dict], float | None]) -> str:
    X, y = [], []
    for r in rows:
        v = value(r)
        if r["opponent_rating"] is None or r["went_first"] is None or v is None:
            continue
        X.append([r["opponent_rating"] / 400.0, float(r["went_first"]), float(v)])
        y.append(int(r["won"]))
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(y) < 12 or len(set(y.tolist())) < 2 or len(set(X[:, 2].tolist())) < 2:
        return f"{name}: insufficient variation (n={len(y)})"
    model = LogisticRegression(penalty=None, max_iter=8000).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    design = np.hstack([X, np.ones((len(X), 1))])
    try:
        cov = np.linalg.inv(design.T @ np.diag(p * (1 - p)) @ design)
    except np.linalg.LinAlgError:
        return f"{name}: singular"
    se = float(np.sqrt(np.diag(cov))[2])
    beta = float(model.coef_[0][2])
    z = beta / se
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (f"{name:<34} n={len(y):>3}  {beta * ELO:+8.1f} Elo/unit  "
            f"z={z:+5.2f}  p={pv:.4f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith(("v22", "v24")):
            continue
        row = dict(raw)
        row["won"] = raw["won"] == "True"
        row["went_first"] = (
            None if raw["went_first"] == "" else raw["went_first"] == "True")
        row["opponent_rating"] = fnum(raw, "opponent_rating")
        ours = fnum(raw, "own_first_shadow_turn")
        theirs = fnum(raw, "opp_first_attack_turn")
        row["ours"] = ours
        row["theirs"] = theirs
        row["lead"] = (theirs - ours) if (ours and theirs) else None
        rows.append(row)

    mirror = [r for r in rows if r["opponent_family"] == "Grimmsnarl (mirror)"]
    strong = [r for r in mirror
              if r["opponent_rating"] is not None and r["opponent_rating"] >= 950]

    print(f"mirror, all ratings : {block(mirror)}")
    print(f"mirror, opp >= 950  : {block(strong)}\n")

    print("=== the attack-race differential (their first attack - our first Shadow) ===")
    for label, pool in (("all mirrors", mirror), ("mirror opp>=950", strong)):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            lead = r["lead"]
            if lead is None:
                buckets["unknown"].append(r)
            elif lead >= 2:
                buckets["we attack 2+ turns first"].append(r)
            elif lead == 1:
                buckets["we attack 1 turn first"].append(r)
            elif lead == 0:
                buckets["same turn"].append(r)
            else:
                buckets["they attack first"].append(r)
        print(f"  [{label}]")
        for key in ("we attack 2+ turns first", "we attack 1 turn first",
                    "same turn", "they attack first", "unknown"):
            if key in buckets:
                print(f"    {key:<28}{block(buckets[key])}")
    print()

    print("=== controlled fits inside the mirror ===")
    for pool_label, pool in (("all mirrors", mirror), ("opp>=950", strong)):
        print(f"  [{pool_label}]")
        for name, value in (
            ("attack lead (turns)", lambda r: r["lead"]),
            ("own_first_shadow_turn", lambda r: r["ours"]),
            ("opp_first_attack_turn", lambda r: r["theirs"]),
            ("we attacked first (bool)",
             lambda r: None if r["lead"] is None else float(r["lead"] > 0)),
            ("grim_evolutions", lambda r: fnum(r, "grim_evolutions")),
            ("stamps", lambda r: fnum(r, "stamps")),
            ("bosses", lambda r: fnum(r, "bosses")),
            ("our_deck_left", lambda r: fnum(r, "our_deck_left")),
        ):
            print("    " + fit(pool, name, value))
    print()

    print("=== who attacks first in the mirror, by turn order ===")
    for first in (True, False):
        pool = [r for r in mirror if r["went_first"] is first and r["lead"] is not None]
        if not pool:
            continue
        counts = Counter(int(r["lead"]) for r in pool)
        mean = sum(r["lead"] for r in pool) / len(pool)
        print(f"  went {'first' if first else 'second'}: n={len(pool)}  "
              f"mean lead {mean:+.2f}  distribution "
              f"{dict(sorted(counts.items()))}")
        print(f"    {block(pool)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
