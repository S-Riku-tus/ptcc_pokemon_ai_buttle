"""How does our record line up with the decks that actually占める the top of the board?

`probe_top60_decks.py` gives the deck hash behind every top-60 submission.
Our 281 pooled v22+v24 games give a win rate per opponent deck hash.  Joining
them answers the only question that matters for a rating target: is the
deficit spread evenly, or concentrated in the lists we are guaranteed to meet
on the way up?

Also prints the field-share-weighted expected win rate: what we would score if
every game were drawn uniformly from the top-60 slot distribution.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
TOP60 = ROOT / "experiments/grimmsnarl_ml_v24/top60_decks.json"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/meta_pressure.json"
OUR_HASH = "9714ab5c3996f6cc"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    games = [
        row for row in csv.DictReader(GAMES.open(encoding="utf-8-sig"))
        if row["version"].startswith(("v22", "v24"))
    ]
    record: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    family: dict[str, Counter] = defaultdict(Counter)
    for row in games:
        key = row["opponent_deck_hash"] or "unknown"
        record[key][0] += 1
        record[key][1] += int(row["won"] == "True")
        family[key][row["opponent_family"] or "?"] += 1

    top60 = json.loads(TOP60.read_text(encoding="utf-8")) if TOP60.exists() else None
    share = Counter()
    best_rating: dict[str, float] = {}
    if top60:
        for entry in top60["rows"]:
            h = entry.get("deck_hash")
            if not h:
                continue
            share[h] += 1
            best_rating[h] = max(best_rating.get(h, 0.0), entry["score"])

    rows: list[dict[str, Any]] = []
    total_slots = sum(share.values()) or 1
    for h, slots in share.most_common():
        n, wins = record.get(h, [0, 0])
        low, high = wilson(wins, n) if n else (None, None)
        rows.append({
            "deck_hash": h,
            "top60_slots": slots,
            "field_share": round(slots / total_slots, 4),
            "best_top60_rating": best_rating.get(h),
            "our_games": n,
            "our_record": f"{wins}-{n - wins}" if n else "-",
            "our_win_rate": round(wins / n, 4) if n else None,
            "wilson95": [round(low, 3), round(high, 3)] if n else None,
            "family": family[h].most_common(1)[0][0] if n else None,
        })

    measured = [r for r in rows if r["our_games"] >= 3]
    weight = sum(r["field_share"] for r in measured)
    expected = (
        sum(r["field_share"] * r["our_win_rate"] for r in measured) / weight
        if weight else None
    )

    unmeasured = [r for r in rows if r["our_games"] < 3]

    payload = {
        "our_deck_hash": OUR_HASH,
        "pooled_games": len(games),
        "top60_slot_weighted_expected_win_rate": round(expected, 4) if expected else None,
        "share_of_top60_we_have_measured": round(weight, 4),
        "by_deck": rows,
        "unmeasured_top60_share": round(sum(r["field_share"] for r in unmeasured), 4),
        "our_opponents_not_in_top60": [
            {
                "deck_hash": h,
                "games": n,
                "record": f"{w}-{n - w}",
                "win_rate": round(w / n, 4),
                "family": family[h].most_common(1)[0][0],
            }
            for h, (n, w) in sorted(record.items(), key=lambda kv: -kv[1][0])
            if h not in share and n >= 4
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pooled games: {len(games)}   "
          f"top-60 slot-weighted expected win rate: "
          f"{payload['top60_slot_weighted_expected_win_rate']}  "
          f"(covering {weight:.1%} of top-60 slots)\n")
    print(f"{'deck_hash':<18}{'slots':>6}{'share':>8}{'best':>8}"
          f"{'n':>5}{'record':>9}{'wr':>7}   family")
    for r in rows:
        print(
            f"{r['deck_hash']:<18}{r['top60_slots']:>6}{r['field_share']:>8.3f}"
            f"{r['best_top60_rating'] or 0:>8.0f}{r['our_games']:>5}"
            f"{r['our_record']:>9}"
            f"{(f'{r['our_win_rate']:.3f}' if r['our_win_rate'] is not None else '-'):>7}"
            f"   {r['family'] or ''}"
        )
    print("\nOpponents we met that are NOT in the current top 60 (n>=4):")
    for r in payload["our_opponents_not_in_top60"]:
        print(f"  {r['deck_hash']:<18} n={r['games']:>3} {r['record']:>7} "
              f"{r['win_rate']:.3f}  {r['family']}")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
