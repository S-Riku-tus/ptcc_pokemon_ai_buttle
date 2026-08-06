"""How much *new* same-deck teacher data exists right now, as a number.

The v5.1 refresh froze 4,097 games on 2026-08-05. The archive has had no new
same-deck game since, which reads like the meta stopped moving - and it did not.
It is the collection that stopped: the Kaggle EpisodeService only serves
episodes for a submission id, submissions get replaced, and the submissions our
21 pilots were tracked under have stopped playing. See
[[kaggle-teacher-log-refetch]].

So "should we retrain on more data" is not answerable from the archive. This
takes a current leaderboard snapshot's representative submissions, downloads one
replay per submission, and reports which of them play the imitation line's exact
60-card deck. Teams already in the frozen selection are marked, so what is left
is the new teacher data actually on offer.

One replay per submission, so this is cheap and read-only: nothing is written
into data/kaggle_grimmsnarl_top50.

Usage:
    python experiments/grimmsnarl_ml_v6/measure_refresh_opportunity.py \
        --submissions .tmp/v6/lb_snapshot/latest/public_submissions_top60.csv \
        --scratch .tmp/v6/deck_probe --top 40
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

DECK_HASH = "9714ab5c3996f6cc"
GRIMMSNARL_EX_ID = 648


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v5"
        / "data_refresh_selection.csv",
    )
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    selection = list(csv.DictReader(
        args.selection.open(encoding="utf-8-sig")
    ))
    known_teams = {row["team_id"] for row in selection}
    known_subs = {row["submission_id"] for row in selection}

    rows = [
        row for row in csv.DictReader(
            args.submissions.open(encoding="utf-8-sig")
        )
        if row["is_representative"].lower() in ("true", "1", "yes")
    ]
    rows.sort(key=lambda row: int(row["rank"]))
    rows = rows[: args.top]

    findings: list[dict] = []
    counts: Counter[str] = Counter()
    for row in rows:
        submission = row["public_submission_id"]
        entry = {
            "rank": int(row["rank"]),
            "team_id": row["team_id"],
            "score": float(row["public_score"] or 0),
            "submission_id": submission,
            "team_known": row["team_id"] in known_teams,
            "submission_known": submission in known_subs,
        }
        try:
            episodes = list_submission_episodes(int(submission))
        except Exception as error:  # noqa: BLE001
            entry["status"] = f"list_failed: {type(error).__name__}"
            findings.append(entry)
            counts["list_failed"] += 1
            continue
        done = [
            episode for episode in episodes
            if episode.state in ("COMPLETED", "")
        ]
        entry["episodes_available"] = len(done)
        if not done:
            entry["status"] = "no_episodes"
            findings.append(entry)
            counts["no_episodes"] += 1
            continue
        newest = max(done, key=lambda episode: episode.create_time)
        entry["newest_episode_at"] = newest.create_time
        try:
            status, path = download_replay(
                newest.episode_id, args.scratch, overwrite=False
            )
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
        entry["has_grimmsnarl_ex"] = GRIMMSNARL_EX_ID in deck
        if entry["deck_hash"] == DECK_HASH:
            entry["status"] = "same_deck"
            counts["same_deck"] += 1
            if not entry["team_known"]:
                counts["same_deck_new_team"] += 1
            elif not entry["submission_known"]:
                counts["same_deck_new_submission"] += 1
        elif entry["has_grimmsnarl_ex"]:
            entry["status"] = "grimmsnarl_other_list"
            counts["grimmsnarl_other_list"] += 1
        else:
            entry["status"] = "other_archetype"
            counts["other_archetype"] += 1
        findings.append(entry)
        time.sleep(args.sleep)

    report = {
        "deck_hash": DECK_HASH,
        "submissions_probed": len(rows),
        "counts": dict(counts),
        "same_deck_new_episodes_available": sum(
            entry.get("episodes_available", 0) for entry in findings
            if entry.get("status") == "same_deck"
            and not entry.get("submission_known")
        ),
        "findings": findings,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")

    print(f"{'rank':>4} {'team':>9} {'score':>7} {'sub':>10} {'eps':>5} "
          f"{'known':>6} {'status'}")
    for entry in findings:
        known = (
            "sub" if entry["submission_known"]
            else ("team" if entry["team_known"] else "-")
        )
        print(
            f"{entry['rank']:>4} {entry['team_id']:>9} {entry['score']:>7.1f} "
            f"{entry['submission_id']:>10} "
            f"{entry.get('episodes_available', 0):>5} {known:>6} "
            f"{entry.get('status')}"
        )
    print()
    print(json.dumps(
        {k: v for k, v in report.items() if k != "findings"},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
