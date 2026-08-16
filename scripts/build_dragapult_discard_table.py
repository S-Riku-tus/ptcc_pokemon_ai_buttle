"""Derive the Ultra Ball discard order from the teachers, not from taste.

Ultra Ball's cost is a two-card discard, and ``minCount == maxCount == 2``
means the ranker never sees it: multi-pick selects fall to the deterministic
policy, and no imitation metric scores them.  On the frozen test split that
context agreed with the teachers 21.9% of the time, the worst of any context.

A hand-written priority is guesswork.  The teachers pitch these cards at very
different rates - Budew 97.5%, Buddy-Buddy Poffin 89.3%, Unfair Stamp 1.5% -
so the empirical rate *is* the policy, and it can be read off the corpus.

Only training-split episodes are used, so the test split stays untouched.

Usage:
  python scripts/build_dragapult_discard_table.py \
      --data-root data/kaggle_dragapult_exact \
      --split-report experiments/dragapult_ml_v2/train_full.json \
      --target agents/dragapult/dragapult_ml_v2/fallback_policy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEGIN = "# --- BEGIN GENERATED DISCARD TABLE ---"
END = "# --- END GENERATED DISCARD TABLE ---"

CTX_DISCARD = 8
AREA_HAND = 2
CARDS = {
    int(card["cardId"]): str(card["name"])
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument(
        "--effect", type=int, default=1121,
        help="Only count discards paid for this card (Ultra Ball by default).",
    )
    parser.add_argument("--min-offers", type=int, default=25)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    boundaries = json.loads(
        args.split_report.read_text(encoding="utf-8")
    ).get("split_boundaries") or {}
    index_path = args.index or args.data_root / "indexes" / "episodes.csv"
    rows = list(csv.DictReader(index_path.read_text(encoding="utf-8-sig").splitlines()))

    offered: Counter[int] = Counter()
    taken: Counter[int] = Counter()
    seen: set[tuple[str, int]] = set()
    episodes = 0
    for row in rows:
        if row.get("download_status") not in ("success", "skipped_existing"):
            continue
        key = (str(row["episode_id"]), int(row["seat_index"]))
        if key in seen:
            continue
        seen.add(key)
        boundary = boundaries.get(str(row.get("team_id")))
        if not boundary or int(row["episode_id"]) >= int(boundary[0]):
            continue  # validation/test episodes never contribute
        path = Path(row["replay_path"])
        if not path.is_absolute():
            path = args.data_root / path
        if not path.exists():
            continue
        episodes += 1
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        seat = int(row["seat_index"])
        for step_index, pair in enumerate(steps):
            record = pair[seat]
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            current = observation.get("current")
            select = observation.get("select")
            if not isinstance(current, dict) or not isinstance(select, dict):
                continue
            if int(select.get("context", -1)) != CTX_DISCARD:
                continue
            if int((select.get("effect") or {}).get("id", -1)) != args.effect:
                continue
            action = (
                steps[step_index + 1][seat].get("action")
                if step_index + 1 < len(steps) else None
            )
            if not isinstance(action, list):
                continue
            players = current.get("players") or [{}, {}]
            hand = (players[seat] if seat in (0, 1) else {}).get("hand") or []
            for position, option in enumerate(select.get("option") or []):
                if int(option.get("area", -1)) != AREA_HAND:
                    continue
                card_index = int(option.get("index", -1))
                if not 0 <= card_index < len(hand):
                    continue
                card = hand[card_index]
                if not isinstance(card, dict):
                    continue
                card_id = int(card.get("id", -1))
                offered[card_id] += 1
                if position in action:
                    taken[card_id] += 1

    table = {
        card_id: round(taken[card_id] / count, 4)
        for card_id, count in offered.items()
        if count >= args.min_offers
    }
    ordered = sorted(table.items(), key=lambda item: -item[1])
    print(f"training episodes {episodes}; cards with >= {args.min_offers} offers: "
          f"{len(table)}")
    for card_id, rate in ordered:
        print(f"  {CARDS.get(card_id, card_id)!s:30} {rate:.3f}  "
              f"({taken[card_id]}/{offered[card_id]})")

    lines = [
        BEGIN,
        "# Generated by scripts/build_dragapult_discard_table.py from the",
        "# training split only.  Value = fraction of the times the teachers",
        "# pitched this card when Ultra Ball offered it.  Do not edit by hand.",
        f"# training episodes: {episodes}; minimum offers per card:"
        f" {args.min_offers}",
        "TEACHER_DISCARD_RATE = {",
    ]
    for card_id, rate in sorted(table.items()):
        lines.append(
            f"    {card_id}: {rate},  # {CARDS.get(card_id, card_id)}"
            f" {taken[card_id]}/{offered[card_id]}"
        )
    lines.append("}")
    lines.append(END)
    generated = "\n".join(lines) + "\n"

    if args.target:
        source = args.target.read_text(encoding="utf-8")
        if BEGIN not in source:
            print(f"{args.target}: no marker block, skipped", file=sys.stderr)
            return 1
        pattern = re.compile(
            re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", re.DOTALL
        )
        args.target.write_text(pattern.sub(generated, source), encoding="utf-8")
        print(f"updated {args.target}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "episodes": episodes,
            "effect": args.effect,
            "min_offers": args.min_offers,
            "rates": {
                str(card_id): {
                    "name": CARDS.get(card_id, str(card_id)),
                    "offered": offered[card_id],
                    "discarded": taken[card_id],
                    "rate": table.get(card_id),
                }
                for card_id in sorted(offered)
            },
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
