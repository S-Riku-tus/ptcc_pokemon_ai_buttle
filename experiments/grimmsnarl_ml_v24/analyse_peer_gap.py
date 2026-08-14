"""Where does the 1095-rated pilot of the identical 60 actually beat us?

AlphaTCG (team 16381823, submission 55350342, rank 22, 1095.3) is one of only
two same-deck pilots left in the top 60, and 120 of their replays are now on
disk.  Same 60 cards, same engine, ~110 Elo above our pooled strength - so any
difference is play, not list.

The comparison is only honest inside matched conditions, so everything here is
cut by opponent rating band and by matchup family, and the date range of each
corpus is printed first: their submission is older than ours and the meta moved
(Grimmsnarl went from 51% of the top 50 to 5% of the top 60), so a pooled
head-to-head number would mostly measure which field each of us was drawn into.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_with_peer_games.csv"
PEER = "peer_alphatcg"


def fnum(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def block(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return f"{'n=0':>28}"
    wins = sum(1 for r in rows if r["won"])
    low, high = wilson(wins, n)
    return (f"n={n:>3} {wins:>3}-{n - wins:<3} {wins / n:.3f} "
            f"[{low:.3f},{high:.3f}]")


def band(rating: float | None) -> str:
    if rating is None:
        return "unknown"
    for edge in (700, 800, 900, 1000, 1100):
        if rating < edge:
            return f"<{edge}"
    return ">=1100"


def mean(rows: list[dict], key: str) -> float | None:
    values = [fnum(r, key) for r in rows]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ours: list[dict[str, Any]] = []
    peer: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        row = dict(raw)
        row["won"] = raw["won"] == "True"
        row["rating"] = fnum(raw, "opponent_rating")
        if raw["version"] == PEER:
            peer.append(row)
        elif raw["version"].startswith(("v22", "v24")):
            ours.append(row)

    for label, pool in (("us (v22+v24)", ours), ("AlphaTCG 1095.3", peer)):
        times = sorted(r["create_time"] for r in pool if r.get("create_time"))
        print(f"{label:<18} {block(pool)}  window "
              f"{times[0][:10] if times else '?'} .. "
              f"{times[-1][:10] if times else '?'}")
    print("\nNOTE: the peer corpus predates ours; the meta moved in between, so\n"
          "      compare cells, not pooled numbers.\n")

    print("=== by opponent rating band ===")
    print(f"  {'band':<10}{'us':<32}{'AlphaTCG'}")
    for key in ("<700", "<800", "<900", "<1000", "<1100", ">=1100"):
        a = [r for r in ours if band(r["rating"]) == key]
        b = [r for r in peer if band(r["rating"]) == key]
        if not a and not b:
            continue
        print(f"  {key:<10}{block(a):<32}{block(b)}")
    print()

    print("=== by matchup family, opponents >= 950 ===")
    strong_a = [r for r in ours if (r["rating"] or 0) >= 950]
    strong_b = [r for r in peer if (r["rating"] or 0) >= 950]
    print(f"  {'family':<26}{'us':<32}{'AlphaTCG'}")
    families = Counter(r["opponent_family"] for r in strong_a + strong_b)
    for fam, _ in families.most_common():
        a = [r for r in strong_a if r["opponent_family"] == fam]
        b = [r for r in strong_b if r["opponent_family"] == fam]
        if len(a) + len(b) < 4:
            continue
        print(f"  {fam:<26}{block(a):<32}{block(b)}")
    print()

    print("=== behaviour, opponents >= 950 (all matchups) ===")
    keys = [
        "turns", "attacks", "shadow_attacks", "adrena_brains", "bosses",
        "stamps", "rare_candies", "grim_evolutions", "froslass_evolves",
        "own_first_shadow_turn", "own_first_ready_turn", "our_deck_left",
        "our_bodies_left", "our_prize_left", "opp_prize_left",
    ]
    print(f"  {'metric':<24}{'us':>10}{'AlphaTCG':>12}{'delta':>10}")
    for key in keys:
        a, b = mean(strong_a, key), mean(strong_b, key)
        if a is None or b is None:
            continue
        print(f"  {key:<24}{a:>10.2f}{b:>12.2f}{b - a:>+10.2f}")
    print()

    print("=== the same behaviour split by their own result, >= 950 ===")
    print(f"  {'metric':<24}{'us W':>8}{'us L':>8}{'peer W':>9}{'peer L':>9}")
    for key in ("attacks", "adrena_brains", "bosses", "stamps",
                "grim_evolutions", "own_first_shadow_turn", "our_bodies_left"):
        aw, al = (mean([r for r in strong_a if r["won"]], key),
                  mean([r for r in strong_a if not r["won"]], key))
        bw, bl = (mean([r for r in strong_b if r["won"]], key),
                  mean([r for r in strong_b if not r["won"]], key))
        if None in (aw, al, bw, bl):
            continue
        print(f"  {key:<24}{aw:>8.2f}{al:>8.2f}{bw:>9.2f}{bl:>9.2f}")
    print()

    print("=== turn order ===")
    for label, pool in (("us", ours), ("AlphaTCG", peer)):
        first = [r for r in pool if r["went_first"] == "True"]
        second = [r for r in pool if r["went_first"] == "False"]
        print(f"  {label:<10} first {block(first)}   second {block(second)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
