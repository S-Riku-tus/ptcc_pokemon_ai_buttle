"""Slot-weight the matchup table by the current top 40 and rank the deficits.

archetypes.json has a win rate per opponent deck and how many of the current top
40 submissions play it. This turns those two columns into the number that
actually matters for a ladder run - the win rate this 60-card list can expect
against the field it would meet at 1100+ - and attributes the shortfall to
individual archetypes, so a fix can be chosen by size rather than by whichever
gap was easiest to measure.

Cells with fewer than --min-games field games are reported but excluded from the
weighted expectation; with 1 game they are noise, not a matchup.

It also checks a feasibility claim the recommendations depend on: our stored
replays contain the *opponent's* observations and actions too, so a sparring
partner for a counter deck can be built from data we already hold.

Usage:
    python experiments/v6_meta_gap_analysis/measure_meta_pressure.py \
        --archetypes archetypes.json --out meta_pressure.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402

DECK_HASH = "9714ab5c3996f6cc"
CORPUS = ROOT / "data" / "kaggle_grimmsnarl_top50"


def opponent_decision_counts(deck_hashes: set[str]) -> dict:
    """Do we hold the counter pilots' own decisions? One replay per deck."""
    out: dict[str, dict] = {}
    for row in csv.DictReader(
        open(CORPUS / "indexes" / "replay_index.csv", encoding="utf-8-sig")
    ):
        if row["deck_hash"] != DECK_HASH or not deck_hashes - set(out):
            continue
        path = CORPUS / Path(row["replay_path"].replace(chr(92), "/"))
        if not path.exists():
            continue
        seat = int(row["seat_index"])
        try:
            head = extract_fast_header_from_file(path)
        except Exception:
            continue
        other = head["deck_hashes"][1 - seat]
        if other not in deck_hashes or other in out:
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        counts: Counter = Counter()
        for step in replay.get("steps") or []:
            for index in (seat, 1 - seat):
                if index >= len(step):
                    continue
                record = step[index] or {}
                if record.get("status") != "ACTIVE":
                    continue
                select = (record.get("observation") or {}).get("select") or {}
                if select:
                    key = "ours" if index == seat else "theirs"
                    counts[key] += 1
                    counts[f"{key}_ctx{int(select.get('context', -1))}"] += 1
        out[other] = {
            "episode": row["episode_id"],
            "our_selects": counts["ours"],
            "their_selects": counts["theirs"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetypes", type=Path, required=True)
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.archetypes.read_text("utf-8"))
    rows = [row for row in data["archetypes"] if row["top40_submissions"]]
    # Our own submission is one of the six same-deck slots in the top 40.
    slots = {
        row["deck_hash"]: row["top40_submissions"] - (1 if row["is_mirror"] else 0)
        for row in rows
    }
    scored = {
        row["deck_hash"]: row for row in rows
        if row["field_games"] >= args.min_games
    }
    weight_total = sum(slots[h] for h in scored)
    expected = sum(
        slots[h] * scored[h]["field_win_rate"] for h in scored
    ) / weight_total
    expected_top3 = None
    top3 = {
        h: row for h, row in scored.items()
        if row["top3_pilot_win_rate"] is not None and row["top3_pilot_games"] >= 10
    }
    if top3:
        weight3 = sum(slots[h] for h in top3)
        expected_top3 = sum(
            slots[h] * top3[h]["top3_pilot_win_rate"] for h in top3
        ) / weight3

    table = []
    for deck_hash, row in sorted(
        scored.items(), key=lambda kv: -slots[kv[0]] * (0.5 - kv[1]["field_win_rate"])
    ):
        share = slots[deck_hash] / weight_total
        table.append({
            "deck_hash": deck_hash,
            "headline": row["headline"],
            "top40_slots": slots[deck_hash],
            "field_share": round(share, 4),
            "field_games": row["field_games"],
            "field_win_rate": row["field_win_rate"],
            "top3_pilot_win_rate": row["top3_pilot_win_rate"],
            "our_games": row["our_games"],
            "deficit_vs_even": round(share * (0.5 - row["field_win_rate"]), 4),
            "gain_if_fixed_to_even": round(
                share * max(0.0, 0.5 - row["field_win_rate"]), 4
            ),
        })
    excluded = [
        {
            "deck_hash": row["deck_hash"], "headline": row["headline"],
            "top40_slots": row["top40_submissions"],
            "field_games": row["field_games"],
            "field_win_rate": row["field_win_rate"],
        }
        for row in rows if row["deck_hash"] not in scored
    ]

    report = {
        "min_games": args.min_games,
        "expected_win_rate_vs_top40_field": round(expected, 4),
        "expected_win_rate_top3_pilots": (
            round(expected_top3, 4) if expected_top3 is not None else None
        ),
        "weighted_slots": weight_total,
        "table": table,
        "excluded_low_sample": excluded,
        "opponent_decisions_available": opponent_decision_counts(
            {"0dede7cb8026e473", "a7ee29914c1dce64", "202ee2cec6cbe8b4"}
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"expected win rate vs the current top-40 field: {expected:.3f}"
        f"  (top-3 pilots' own rates: "
        f"{expected_top3:.3f})" if expected_top3 else ""
    )
    print(
        f"{'deck':<17}{'slots':>6}{'share':>7}{'N':>6}{'field%':>8}"
        f"{'top3%':>7}{'oursN':>6}{'deficit':>9}  headline"
    )
    for row in table:
        top3_cell = (
            f"{row['top3_pilot_win_rate']:>7.2f}"
            if row["top3_pilot_win_rate"] is not None else " " * 7
        )
        print(
            f"{row['deck_hash']:<17}{row['top40_slots']:>6}"
            f"{row['field_share']:>7.3f}{row['field_games']:>6}"
            f"{row['field_win_rate']:>8.3f}{top3_cell}{row['our_games']:>6}"
            f"{row['deficit_vs_even']:>9.4f}  {', '.join(row['headline'][:2])}"
        )
    print("\nexcluded (low sample):")
    for row in excluded:
        print(
            f"  {row['deck_hash']} slots={row['top40_slots']} "
            f"N={row['field_games']} win={row['field_win_rate']} "
            f"{', '.join(row['headline'][:2])}"
        )
    print("\nopponent decisions available in our own replays:")
    print(json.dumps(report["opponent_decisions_available"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
