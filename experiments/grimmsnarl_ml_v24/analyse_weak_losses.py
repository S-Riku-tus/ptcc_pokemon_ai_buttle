"""The 24 losses to sub-900 opponents, and what they cost.

The peer comparison changes the target.  Against 1000-1100 opponents we score
0.540 (n=50) and the 1095-rated pilot of the identical 60 scores 0.571 (n=49) -
indistinguishable.  So the ~110 Elo between us is not obviously bought in that
band.  `elo_income.json` says we conceded **885 Elo to opponents rated under
900**, at 45-55 Elo per loss, across 24 losses in 117 games.  Those are the
most expensive games on the ladder and the cheapest to win.

This asks what those 24 losses look like: a coin-flip race lost on the last
prize, or a game that never started.  A structural floor (mulligan, prized
line, board-out) is variance; a full-length game lost from an even board is
policy.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_with_peer_games.csv"


def fnum(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def mean(rows: list[dict], key: str) -> float:
    values = [fnum(r, key) for r in rows]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith(("v22", "v24")):
            continue
        row = dict(raw)
        row["won"] = raw["won"] == "True"
        row["rating"] = fnum(raw, "opponent_rating")
        rows.append(row)

    weak = [r for r in rows if r["rating"] is not None and r["rating"] < 900]
    losses = [r for r in weak if not r["won"]]
    wins = [r for r in weak if r["won"]]
    n, w = len(weak), len(wins)
    low, high = wilson(w, n)
    print(f"opponents < 900: n={n}  {w}-{len(losses)}  {w / n:.3f} "
          f"[{low:.3f},{high:.3f}]")
    print(f"  a 1095-rated player would be expected to score ~"
          f"{1 / (1 + 10 ** ((mean(weak, 'opponent_rating') - 1095) / 400)):.3f} "
          f"here; a 985 one ~"
          f"{1 / (1 + 10 ** ((mean(weak, 'opponent_rating') - 985) / 400)):.3f}")
    print(f"  mean opponent rating {mean(weak, 'opponent_rating'):.1f}\n")

    print("=== shape of the 24 sub-900 losses ===")
    print("  prizes we took: "
          + str(dict(sorted(Counter(
              int(6 - (fnum(r, 'our_prize_left') or 6)) for r in losses
          ).items()))))
    print("  bodies left:    "
          + str(dict(sorted(Counter(
              int(fnum(r, 'our_bodies_left') or 0) for r in losses).items()))))
    print("  deck left:      "
          + str(dict(sorted(Counter(
              int(fnum(r, 'our_deck_left') or 0) for r in losses).items()))))
    print("  by family:      "
          + str(dict(Counter(r["opponent_family"] for r in losses).most_common())))
    print()

    print(f"  {'metric':<26}{'win':>9}{'loss':>9}")
    for key in ("turns", "attacks", "shadow_attacks", "adrena_brains",
                "own_first_shadow_turn", "own_first_ready_turn",
                "our_bodies_left", "our_deck_left", "grim_evolutions"):
        print(f"  {key:<26}{mean(wins, key):>9.2f}{mean(losses, key):>9.2f}")
    print()

    print("=== classification of the sub-900 losses ===")
    never_started = [r for r in losses
                     if (fnum(r, "own_first_shadow_turn") or 99) >= 5
                     or (6 - (fnum(r, "our_prize_left") or 6)) <= 1]
    boardout = [r for r in losses if (fnum(r, "our_bodies_left") or 9) <= 1]
    close = [r for r in losses
             if (6 - (fnum(r, "our_prize_left") or 6)) >= 4]
    print(f"  never got going (first Shadow >= turn 5, or <=1 prize): "
          f"{len(never_started)}")
    print(f"  board wiped to <=1 body:                               "
          f"{len(boardout)}")
    print(f"  close race, we were on 4-5 prizes:                     "
          f"{len(close)}")
    print()

    print("=== every sub-900 loss ===")
    print(f"  {'episode':<10}{'ver':<7}{'opp':>6}{'turns':>6}{'we':>4}"
          f"{'they':>5}{'shadow':>7}{'bodies':>7}{'deck':>6}  family")
    for r in sorted(losses, key=lambda r: r["rating"]):
        print(
            f"  {r['episode_id']:<10}{r['version']:<7}{r['rating']:>6.0f}"
            f"{fnum(r, 'turns') or 0:>6.0f}"
            f"{6 - (fnum(r, 'our_prize_left') or 6):>4.0f}"
            f"{6 - (fnum(r, 'opp_prize_left') or 6):>5.0f}"
            f"{fnum(r, 'own_first_shadow_turn') or -1:>7.0f}"
            f"{fnum(r, 'our_bodies_left') or 0:>7.0f}"
            f"{fnum(r, 'our_deck_left') or 0:>6.0f}  {r['opponent_family']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
