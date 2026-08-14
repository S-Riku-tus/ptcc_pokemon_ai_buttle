"""The only band that can raise the rating: opponents at 950+.

`elo_income.json` shows the run banks almost all of its Elo below 900 and
earns ~2.5-3.0 Elo per game from 900 up.  Those cheap wins are the transient
climb from Kaggle's 600 start and vanish once the rating converges, so the
equilibrium is set entirely by the 950+ band.  This slices the pooled 281
v22+v24 games to that band only and reports matchup, turn order and game shape
inside it, with Wilson bounds, so a 6-game cell is not mistaken for a verdict.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
CUT = 950.0


def fnum(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def block(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "n=0"
    wins = sum(1 for r in rows if r["won"])
    low, high = wilson(wins, n)
    return (f"n={n:>3} {wins:>3}-{n - wins:<3} {wins / n:.3f} "
            f"[{low:.3f},{high:.3f}]")


def show(title: str, rows: list[dict], key: Callable[[dict], str],
         min_n: int = 1) -> None:
    print(f"--- {title} ---")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    for label, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(items) < min_n:
            continue
        print(f"  {label:<30}{block(items)}")
    print()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith(("v22", "v24")):
            continue
        rating = fnum(raw, "opponent_rating")
        row = dict(raw)
        row["won"] = raw["won"] == "True"
        row["rating"] = rating
        rows.append(row)

    strong = [r for r in rows if r["rating"] is not None and r["rating"] >= CUT]
    weak = [r for r in rows if r["rating"] is not None and r["rating"] < CUT]

    print(f"pooled {len(rows)} games; opponents >= {CUT:.0f}: {block(strong)}")
    print(f"                          opponents <  {CUT:.0f}: {block(weak)}\n")

    show(f"matchup family, opponents >= {CUT:.0f}", strong,
         lambda r: r["opponent_family"] or "unknown")
    show(f"turn order, opponents >= {CUT:.0f}", strong,
         lambda r: "first" if r["went_first"] == "True" else "second")
    show(f"our version, opponents >= {CUT:.0f}", strong, lambda r: r["version"])

    print(f"--- loss shape, opponents >= {CUT:.0f} ---")
    losses = [r for r in strong if not r["won"]]
    wins = [r for r in strong if r["won"]]
    for label, group in (("win", wins), ("loss", losses)):
        if not group:
            continue
        def mean(key: str) -> float:
            vals = [fnum(r, key) for r in group]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        print(
            f"  {label:<6} n={len(group):>3}  turns {mean('turns'):5.2f}  "
            f"our prizes left {mean('our_prize_left'):4.2f}  "
            f"opp prizes left {mean('opp_prize_left'):4.2f}  "
            f"bodies left {mean('our_bodies_left'):4.2f}  "
            f"deck left {mean('our_deck_left'):5.2f}"
        )
        print(
            f"         first shadow turn {mean('own_first_shadow_turn'):5.2f}  "
            f"first ready {mean('own_first_ready_turn'):5.2f}  "
            f"attacks {mean('attacks'):5.2f}  shadow {mean('shadow_attacks'):5.2f}  "
            f"grim evo {mean('grim_evolutions'):4.2f}  "
            f"adrena {mean('adrena_brains'):5.2f}  "
            f"boss {mean('bosses'):4.2f}  stamp {mean('stamps'):4.2f}"
        )
    print()

    print(f"--- how the >= {CUT:.0f} losses ended (prizes we took) ---")
    counter: dict[int, int] = defaultdict(int)
    for r in losses:
        left = fnum(r, "our_prize_left")
        if left is not None:
            counter[int(6 - left)] += 1
    for taken in sorted(counter):
        print(f"  took {taken} prize(s): {counter[taken]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
