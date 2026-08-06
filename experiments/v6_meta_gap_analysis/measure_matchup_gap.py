"""Where our ladder rating is actually lost: win rate per opponent archetype.

Every Grimmsnarl version so far has been ranked by one pooled rating and fixed by
comparing per-decision imitation rates. Neither tells us which *matchup* pays.
This reads the replay headers only - both 60-card lists and the reward pair - so
it can label every game by opponent deck hash and compare, on the same field:

* our own ladder runs (v4, v4.5, v5; v6 differs from v5 on 18 stored decisions);
* the pilots on the identical 60-card list who are currently near the top;
* the whole same-deck corpus as a field baseline.

Self-play validation episodes are excluded: both seats are the same submission.

Usage:
    python experiments/v6_meta_gap_analysis/measure_matchup_gap.py \
        --run data/runs/grimmsnarl/20260806_grimmsnarl_ml_v5_sub55275642 \
        --corpus-team 16452116 --out matchup_gap.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402

DECK_HASH = "9714ab5c3996f6cc"
CORPUS = ROOT / "data" / "kaggle_grimmsnarl_top50"


def header(job: tuple[str, int]) -> tuple[str, str, int] | None:
    """(our deck hash, opponent deck hash, our reward) for one replay."""
    path, seat = job
    try:
        head = extract_fast_header_from_file(path)
    except Exception:
        return None
    hashes = head.get("deck_hashes") or ["", ""]
    rewards = head.get("rewards") or [None, None]
    if len(hashes) < 2 or hashes[seat] != DECK_HASH:
        return None
    try:
        mine = int(rewards[seat])
        theirs = int(rewards[1 - seat])
    except (TypeError, ValueError):
        return None
    return hashes[seat], hashes[1 - seat], 1 if mine > theirs else 0


def run_jobs(run_dir: Path) -> list[tuple[str, int, str]]:
    """(replay path, our seat, created_at) for one of our own ladder runs."""
    seats = {
        row["episode_id"]: row["detected_submission_agent_index"]
        for row in csv.DictReader(
            open(run_dir / "manifest.csv", encoding="utf-8-sig")
        )
    }
    jobs: list[tuple[str, int, str]] = []
    for row in csv.DictReader(open(run_dir / "episodes.csv", encoding="utf-8-sig")):
        episode = row["episode_id"]
        if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
            continue
        seat = seats.get(episode, "")
        if seat not in ("0", "1"):
            continue
        path = run_dir / "episodes" / episode / "replay" / f"episode_{episode}.json"
        if path.exists():
            jobs.append((str(path), int(seat), row["create_time"][:10]))
    return jobs


def corpus_jobs(team_id: int | None) -> list[tuple[str, int, str]]:
    """(replay path, pilot seat, created_at) for same-deck corpus replays."""
    index = CORPUS / "indexes" / "replay_index.csv"
    seen: set[tuple[str, str]] = set()
    jobs: list[tuple[str, int, str]] = []
    for row in csv.DictReader(open(index, encoding="utf-8-sig")):
        if row["deck_hash"] != DECK_HASH:
            continue
        if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
            continue
        if team_id is not None and int(row["team_id"]) != team_id:
            continue
        key = (row["episode_id"], row["seat_index"])
        if key in seen:
            continue
        seen.add(key)
        path = CORPUS / Path(row["replay_path"].replace(chr(92), "/"))
        if path.exists():
            jobs.append((str(path), int(row["seat_index"]), row["created_at"][:10]))
    return jobs


def summarise(results: list[tuple[str, str, int]], dates: list[str]) -> dict:
    per_hash: dict[str, Counter] = defaultdict(Counter)
    for _, opponent, win in results:
        per_hash[opponent]["games"] += 1
        per_hash[opponent]["wins"] += win
    total = sum(row["games"] for row in per_hash.values())
    wins = sum(row["wins"] for row in per_hash.values())
    rows = sorted(per_hash.items(), key=lambda kv: -kv[1]["games"])
    return {
        "games": total,
        "wins": wins,
        "win_rate": round(wins / total, 4) if total else None,
        "date_range": [min(dates), max(dates)] if dates else None,
        "by_opponent_deck": {
            opponent: {
                "games": counts["games"],
                "wins": counts["wins"],
                "win_rate": round(counts["wins"] / counts["games"], 4),
            }
            for opponent, counts in rows
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--corpus-team", type=int, action="append", default=[])
    parser.add_argument(
        "--corpus-field", action="store_true",
        help="Add every same-deck corpus game as one pooled baseline.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    labelled: dict[str, list[tuple[str, int, str]]] = {}
    for run_dir in args.run:
        labelled[f"ours:{run_dir.name}"] = run_jobs(run_dir)
    for team in args.corpus_team:
        labelled[f"pilot:{team}"] = corpus_jobs(team)
    if args.corpus_field:
        labelled["field:same_deck_corpus"] = corpus_jobs(None)

    flat = [(label, job) for label, jobs in labelled.items() for job in jobs]
    print(f"replays={len(flat)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        headers = list(pool.map(
            header, [(path, seat) for _, (path, seat, _) in flat], chunksize=32
        ))

    grouped: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    dated: dict[str, list[str]] = defaultdict(list)
    for (label, (_, _, date)), head in zip(flat, headers):
        if head is None:
            continue
        grouped[label].append(head)
        dated[label].append(date)

    report = {
        "deck_hash": DECK_HASH,
        "labels": {
            label: summarise(grouped[label], dated[label])
            for label in labelled
            if grouped[label]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ranked = sorted(
        {h for row in report["labels"].values() for h in row["by_opponent_deck"]},
        key=lambda h: -sum(
            row["by_opponent_deck"].get(h, {}).get("games", 0)
            for row in report["labels"].values()
        ),
    )
    width = max(len(label) for label in report["labels"]) + 1
    print(f"{'label':<{width}} {'games':>6} {'win%':>6} | " + " ".join(
        f"{h[:8]:>13}" for h in ranked[:8]
    ))
    for label, row in report["labels"].items():
        cells = []
        for h in ranked[:8]:
            cell = row["by_opponent_deck"].get(h)
            cells.append(
                f"{cell['wins']:>3}/{cell['games']:<3}{cell['win_rate']:>6.2f}"
                if cell else " " * 13
            )
        print(
            f"{label:<{width}} {row['games']:>6} "
            f"{(row['win_rate'] or 0):>6.3f} | " + " ".join(cells)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
