"""Label the current top-N field by deck and report new exact-list teacher data.

Two questions have to be answered together before spending a training cycle on
this deck: is the archetype still worth imitating, and does new same-deck
teacher data exist right now?  Both need the same probe - one replay per
representative submission - so they are answered in one pass.

Read-only with respect to the corpus: replays land in a scratch directory.

Usage:
  python experiments/dragapult_ml_v1/probe_field.py \
      --submissions .tmp/lb_20260816/latest/public_submissions_top100.csv \
      --scratch .tmp/deck_probe_20260816 --top 60 \
      --out experiments/dragapult_ml_v1/field_20260816.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import deck_hash, extract_fast_header_from_file  # noqa: E402
from scripts.fetch_submission_logs import (  # noqa: E402
    download_replay,
    list_submission_episodes,
)

DECK_HASH = "202ee2cec6cbe8b4"
DRAGAPULT_EX = 121

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}


def archetype(deck: list[int]) -> str:
    """Name the deck by its heaviest, latest-stage Pokemon - the usual label."""
    pokemon = Counter(
        card_id for card_id in deck if CARDS.get(card_id, {}).get("cardType") == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple[int, int, int, int, int]:
        card_id, count = item
        card = CARDS[card_id]
        return (
            int(bool(card.get("stage2"))),
            int(bool(card.get("megaEx") or card.get("ex"))),
            int(bool(card.get("stage1"))),
            count,
            int(card.get("hp") or 0),
        )

    best = max(pokemon.items(), key=key)[0]
    return str(CARDS.get(best, {}).get("name") or best)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "experiments" / "dragapult_ml_v1" / "selected_episodes.csv",
    )
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    known_teams: set[str] = set()
    known_subs: set[str] = set()
    if args.selection.exists():
        for row in csv.DictReader(args.selection.open(encoding="utf-8-sig")):
            known_teams.add(str(row.get("team_id")))
            known_subs.add(str(row.get("submission_id")))

    rows = [
        row for row in csv.DictReader(args.submissions.open(encoding="utf-8-sig"))
        if str(row["is_representative"]).lower() in ("true", "1", "yes")
    ]
    rows.sort(key=lambda row: int(row["rank"]))
    rows = rows[: args.top]

    findings: list[dict] = []
    counts: Counter[str] = Counter()
    decks: Counter[str] = Counter()
    for row in rows:
        submission = str(row["public_submission_id"])
        entry = {
            "rank": int(row["rank"]),
            "team_id": str(row["team_id"]),
            "team_name": row.get("team_name", ""),
            "score": float(row["public_score"] or 0),
            "submission_id": submission,
            "team_known": str(row["team_id"]) in known_teams,
            "submission_known": submission in known_subs,
        }
        try:
            episodes = list_submission_episodes(int(submission))
        except Exception as error:  # noqa: BLE001
            entry["status"] = f"list_failed: {type(error).__name__}"
            findings.append(entry)
            counts["list_failed"] += 1
            continue
        done = [episode for episode in episodes if episode.state in ("COMPLETED", "")]
        entry["episodes_available"] = len(done)
        if not done:
            entry["status"] = "no_episodes"
            findings.append(entry)
            counts["no_episodes"] += 1
            continue
        newest = max(done, key=lambda episode: episode.create_time)
        entry["newest_episode_at"] = newest.create_time
        try:
            _, path = download_replay(newest.episode_id, args.scratch, overwrite=False)
        except Exception as error:  # noqa: BLE001
            entry["status"] = f"replay_failed: {type(error).__name__}"
            findings.append(entry)
            counts["replay_failed"] += 1
            continue
        header = extract_fast_header_from_file(path)
        seat = 0 if str(newest.agent_0_submission_id) == submission else 1
        deck = list(header.get("decks", [[], []])[seat] or [])
        entry["deck_cards"] = len(deck)
        entry["deck_hash"] = deck_hash(deck) if len(deck) == 60 else None
        entry["archetype"] = archetype(deck) if deck else "unknown"
        entry["has_dragapult_ex"] = DRAGAPULT_EX in deck
        decks[entry["archetype"]] += 1
        if entry["deck_hash"] == DECK_HASH:
            entry["status"] = "exact_list"
            counts["exact_list"] += 1
            if not entry["team_known"]:
                counts["exact_list_new_team"] += 1
            elif not entry["submission_known"]:
                counts["exact_list_new_submission"] += 1
        elif entry["has_dragapult_ex"]:
            entry["status"] = "dragapult_other_list"
            counts["dragapult_other_list"] += 1
        else:
            entry["status"] = "other_archetype"
            counts["other_archetype"] += 1
        findings.append(entry)
        time.sleep(args.sleep)

    report = {
        "deck_hash": DECK_HASH,
        "submissions_probed": len(rows),
        "counts": dict(counts),
        "archetypes": dict(decks.most_common()),
        "exact_list_episodes_on_offer": sum(
            entry.get("episodes_available", 0) for entry in findings
            if entry.get("status") == "exact_list"
        ),
        "exact_list_episodes_new_submissions": sum(
            entry.get("episodes_available", 0) for entry in findings
            if entry.get("status") == "exact_list" and not entry.get("submission_known")
        ),
        "findings": findings,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"{'rank':>4} {'team':>9} {'score':>7} {'sub':>10} {'eps':>5} {'known':>6}"
          f"  {'archetype':28} status")
    for entry in findings:
        known = (
            "sub" if entry.get("submission_known")
            else ("team" if entry.get("team_known") else "-")
        )
        print(
            f"{entry['rank']:>4} {entry['team_id']:>9} {entry['score']:>7.1f} "
            f"{entry['submission_id']:>10} {entry.get('episodes_available', 0):>5} "
            f"{known:>6}  {str(entry.get('archetype', '-')):28} {entry.get('status')}"
        )
    print()
    print(json.dumps(
        {k: v for k, v in report.items() if k != "findings"},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
