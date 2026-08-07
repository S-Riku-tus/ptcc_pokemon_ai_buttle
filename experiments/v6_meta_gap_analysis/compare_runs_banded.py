"""Rank our ladder runs the only way this repo's own evidence allows.

A final rating cannot rank two versions (see the ~40-84 point spreads on
identical code), and a pooled win rate tracks the opponent pool: the v4/v4.5/v5
runs were served mean opponent ratings of 967, 843 and 860. So this reports, per
run, the record banded by the opponent's pre-game rating from episodes.csv, the
seat split, and the per-opponent-archetype record from the replay headers.

Both a reward-based and a rating-delta win are computed; they agreed
exactly over the 183 games of v4/v4.5/v5, and a disagreement means a draw or a
missing score rather than a bug to be ignored.

Usage:
    python experiments/v6_meta_gap_analysis/compare_runs_banded.py \
        --run data/runs/grimmsnarl/20260806_grimmsnarl_ml_v6_sub55290882 \
        --out runs_banded.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402

DECK_HASH = "9714ab5c3996f6cc"
BANDS = ((0, 900), (900, 1000), (1000, 1050), (1050, 9999))


def header(path: str) -> dict:
    try:
        return extract_fast_header_from_file(path)
    except Exception:
        return {}


def rows_for(run_dir: Path) -> list[dict]:
    seats = {
        row["episode_id"]: row["detected_submission_agent_index"]
        for row in csv.DictReader(
            open(run_dir / "manifest.csv", encoding="utf-8-sig")
        )
    }
    out: list[dict] = []
    for row in csv.DictReader(
        open(run_dir / "episodes.csv", encoding="utf-8-sig")
    ):
        episode = row["episode_id"]
        seat = seats.get(episode, "")
        if seat not in ("0", "1"):
            continue
        if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
            continue
        path = (
            run_dir / "episodes" / episode / "replay"
            / f"episode_{episode}.json"
        )
        if not path.exists():
            continue
        index = int(seat)
        entry = {"episode": episode, "seat": index, "path": str(path)}
        try:
            entry["mine_before"] = float(row[f"agent_{index}_initial_score"])
            entry["mine_after"] = float(row[f"agent_{index}_updated_score"])
            entry["opponent_before"] = float(
                row[f"agent_{1 - index}_initial_score"]
            )
        except (KeyError, ValueError):
            entry["mine_before"] = entry["mine_after"] = None
            entry["opponent_before"] = None
        out.append(entry)
    return out


def enrich(entry: dict) -> dict:
    head = header(entry["path"])
    hashes = head.get("deck_hashes") or ["", ""]
    rewards = head.get("rewards") or [None, None]
    seat = entry["seat"]
    entry["our_deck"] = hashes[seat] if len(hashes) > seat else ""
    entry["opponent_deck"] = hashes[1 - seat] if len(hashes) > 1 else ""
    try:
        entry["won_reward"] = int(
            int(rewards[seat]) > int(rewards[1 - seat])
        )
    except (TypeError, ValueError, IndexError):
        entry["won_reward"] = None
    if entry.get("mine_after") is not None:
        entry["won_rating"] = int(entry["mine_after"] > entry["mine_before"])
    else:
        entry["won_rating"] = None
    entry["won"] = (
        entry["won_reward"] if entry["won_reward"] is not None
        else entry["won_rating"]
    )
    return entry


def band_of(score: float | None) -> str | None:
    if score is None:
        return None
    return next(f"{low}-{high}" for low, high in BANDS if low <= score < high)


def summarise(rows: list[dict]) -> dict:
    played = [row for row in rows if row["won"] is not None]
    wins = sum(row["won"] for row in played)
    opponents = [
        row["opponent_before"] for row in played
        if row["opponent_before"] is not None
    ]
    bands: dict[str, Counter] = defaultdict(Counter)
    for row in played:
        key = band_of(row["opponent_before"])
        if key is None:
            continue
        bands[key]["games"] += 1
        bands[key]["wins"] += row["won"]
    decks: dict[str, Counter] = defaultdict(Counter)
    for row in played:
        decks[row["opponent_deck"]]["games"] += 1
        decks[row["opponent_deck"]]["wins"] += row["won"]
    disagree = sum(
        1 for row in played
        if row["won_reward"] is not None and row["won_rating"] is not None
        and row["won_reward"] != row["won_rating"]
    )
    # Episode ids rise with time, so the last one has the latest rating.
    ordered = sorted(
        (row for row in played if row["mine_after"] is not None),
        key=lambda row: int(row["episode"]),
    )
    return {
        "games": len(played),
        "wins": wins,
        "losses": len(played) - wins,
        "win_rate": round(wins / len(played), 4) if played else None,
        "mean_opponent_rating": (
            round(statistics.fmean(opponents), 1) if opponents else None
        ),
        "reward_vs_rating_disagreements": disagree,
        "latest_rating_in_index": (
            ordered[-1]["mine_after"] if ordered else None
        ),
        "by_opponent_band": {
            band: {
                "games": counts["games"], "wins": counts["wins"],
                "win_rate": round(counts["wins"] / counts["games"], 4),
            }
            for band, counts in sorted(bands.items())
        },
        "pooled_1000_plus": pooled(
            bands, ("1000-1050", "1050-9999")
        ),
        "pooled_900_plus": pooled(
            bands, ("900-1000", "1000-1050", "1050-9999")
        ),
        "by_opponent_deck": {
            deck: {
                "games": counts["games"], "wins": counts["wins"],
                "win_rate": round(counts["wins"] / counts["games"], 4),
            }
            for deck, counts in sorted(
                decks.items(), key=lambda kv: -kv[1]["games"]
            )
        },
        "mirror_share": (
            round(
                sum(
                    1 for row in played if row["opponent_deck"] == DECK_HASH
                ) / len(played), 4
            ) if played else None
        ),
    }


def pooled(bands: dict[str, Counter], keys: tuple[str, ...]) -> dict:
    games = sum(bands[key]["games"] for key in keys if key in bands)
    wins = sum(bands[key]["wins"] for key in keys if key in bands)
    return {
        "games": games, "wins": wins,
        "win_rate": round(wins / games, 4) if games else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    labelled: dict[str, list[dict]] = {}
    for run_dir in args.run:
        label = run_dir.name.split("_grimmsnarl_")[-1].split("_sub")[0]
        labelled[label] = rows_for(run_dir)
    flat = [(label, row) for label, rows in labelled.items() for row in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        enriched = list(
            pool.map(enrich, [row for _, row in flat], chunksize=8)
        )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for (label, _), row in zip(flat, enriched):
        grouped[label].append(row)

    report = {label: summarise(rows) for label, rows in grouped.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"{'run':<10}{'N':>5}{'W':>4}{'L':>4}{'win%':>7}{'oppRating':>11}"
        f"{'<900':>12}{'900-1000':>12}{'1000+':>12}{'mirror%':>9}"
    )
    for label, row in report.items():
        def cell(block: dict | None) -> str:
            if not block or not block.get("games"):
                return " " * 12
            return (
                f"{block['wins']:>3}/{block['games']:<3}"
                f"{block['win_rate']:>5.2f}"
            )

        print(
            f"{label:<10}{row['games']:>5}{row['wins']:>4}{row['losses']:>4}"
            f"{row['win_rate']:>7.3f}{row['mean_opponent_rating'] or 0:>11.1f}"
            f"{cell(row['by_opponent_band'].get('0-900'))}"
            f"{cell(row['by_opponent_band'].get('900-1000'))}"
            f"{cell(row['pooled_1000_plus'])}"
            f"{row['mirror_share'] or 0:>9.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
