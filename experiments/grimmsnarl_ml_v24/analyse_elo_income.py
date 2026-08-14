"""Where is the rating actually earned and lost, in Elo, not in win rate?

Win rate weights a 575-rated opponent and a 1030-rated opponent equally.  The
ladder does not: beating a much weaker opponent pays almost nothing while
losing to them costs a lot.  A policy can therefore win 65% of its games and
still stall, if the wins are cheap and the losses are expensive.

This reads the per-episode ``initialScore``/``updatedScore`` deltas straight
out of ``episodes.csv`` and sums them by opponent-rating band, by matchup
family and by turn order.  The sum over all games *is* the run's final rating
minus its starting rating, so the decomposition is exact and not a model.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

RUNS = ROOT / "data/runs/grimmsnarl"
GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/elo_income.json"

SUBMISSIONS = {
    "v22_a": 55479857, "v22_b": 55483874, "v22_c": 55486680, "v22_d": 55486691,
    "v24_a": 55496021, "v24_b": 55496665,
}


def band(rating: float | None) -> str:
    if rating is None:
        return "unknown"
    for edge in (700, 800, 900, 1000, 1100):
        if rating < edge:
            return f"<{edge}"
    return ">=1100"


def episode_deltas() -> dict[str, dict[str, Any]]:
    """episode_id -> per-game rating delta and opponent rating."""
    out: dict[str, dict[str, Any]] = {}
    for label, submission in SUBMISSIONS.items():
        directory = next(
            (p for p in RUNS.iterdir()
             if p.is_dir() and p.name.endswith(f"sub{submission}")), None)
        if directory is None:
            continue
        for row in csv.DictReader(
                (directory / "episodes.csv").open(encoding="utf-8-sig")):
            seat = 0 if row["agent_0_submission_id"] == str(submission) else 1
            try:
                initial = float(row[f"agent_{seat}_initial_score"])
                updated = float(row[f"agent_{seat}_updated_score"])
            except ValueError:
                continue
            try:
                opponent = float(row[f"agent_{1 - seat}_initial_score"])
            except ValueError:
                opponent = None
            out[str(row["episode_id"])] = {
                "version": label,
                "delta": updated - initial,
                "our_rating": initial,
                "opponent_rating": opponent,
            }
    return out


def summarise(rows: list[dict[str, Any]], key) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    out = {}
    for label, items in sorted(buckets.items(), key=lambda kv: -sum(
            r["delta"] for r in kv[1])):
        wins = sum(1 for r in items if r["won"])
        gain = sum(r["delta"] for r in items if r["delta"] > 0)
        loss = sum(r["delta"] for r in items if r["delta"] < 0)
        out[label] = {
            "games": len(items),
            "record": f"{wins}-{len(items) - wins}",
            "win_rate": round(wins / len(items), 3),
            "elo_net": round(sum(r["delta"] for r in items), 1),
            "elo_gained": round(gain, 1),
            "elo_conceded": round(loss, 1),
            "elo_per_win": round(gain / wins, 1) if wins else None,
            "elo_per_loss": round(loss / (len(items) - wins), 1)
            if len(items) - wins else None,
            "elo_per_game": round(sum(r["delta"] for r in items) / len(items), 1),
        }
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    deltas = episode_deltas()
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        entry = deltas.get(raw["episode_id"])
        if entry is None:
            continue
        rows.append({
            **entry,
            "won": raw["won"] == "True",
            "went_first": raw["went_first"] == "True",
            "family": raw["opponent_family"] or "unknown",
            "deck_hash": raw["opponent_deck_hash"] or "unknown",
        })

    payload = {
        "games": len(rows),
        "total_elo": round(sum(r["delta"] for r in rows), 1),
        "by_opponent_band": summarise(rows, lambda r: band(r["opponent_rating"])),
        "by_family": summarise(rows, lambda r: r["family"]),
        "by_turn_order": summarise(rows, lambda r: "first" if r["went_first"] else "second"),
        "by_version": summarise(rows, lambda r: r["version"]),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for section in ("by_opponent_band", "by_family", "by_turn_order"):
        print(f"=== {section} ===")
        print(f"{'bucket':<28}{'n':>4}{'record':>9}{'wr':>7}"
              f"{'net':>9}{'+/win':>8}{'-/loss':>8}{'net/game':>10}")
        for label, b in payload[section].items():
            print(
                f"{label:<28}{b['games']:>4}{b['record']:>9}{b['win_rate']:>7.3f}"
                f"{b['elo_net']:>9.1f}{(b['elo_per_win'] or 0):>8.1f}"
                f"{(b['elo_per_loss'] or 0):>8.1f}{b['elo_per_game']:>10.1f}")
        print()
    print(f"Report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
