"""Compare current top-40 Alakazam submissions on deck and turn metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_alakazam_ladder_strategy import aggregate, analyze_replay


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def deck_counts(row: dict[str, Any]) -> dict[int, int]:
    return dict(Counter(int(card["id"]) for card in row.get("target_deck", [])))


def compact_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = aggregate(rows)
    keys = (
        "games",
        "wins",
        "losses",
        "win_rate",
        "end_reasons",
        "games_with_attack_rate",
        "attacks_per_game",
        "avg_first_attack_own_main_turn",
        "post_first_idle_turns_per_game",
        "post_first_idle_turns_per_loss",
        "main_turn_attack_rate",
        "attack_opportunity_conversion_rate",
        "post_first_main_turn_attack_rate",
        "boss_plays",
        "boss_same_turn_attack_rate",
        "selected_play_card_ids",
        "attacker_ids",
        "two_turn_boss_opportunities",
        "two_turn_boss_opportunities_with_boss_play",
        "dual_kadabra_choices",
        "dual_kadabra_active_choices",
        "dual_kadabra_bench_choices",
        "grimmsnarl_games",
        "grimmsnarl_win_rate",
        "matchups",
        "totals",
    )
    return {key: full[key] for key in keys}


def load_v18_deck(path: Path) -> dict[int, int]:
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.reader(handle):
            if raw and raw[0].strip().isdigit():
                counts[int(raw[0])] += 1
    return dict(counts)


def deck_diff(reference: dict[int, int], candidate: dict[int, int]) -> list[dict[str, int]]:
    return [
        {
            "card_id": card_id,
            "v18": reference.get(card_id, 0),
            "top_deck": candidate.get(card_id, 0),
            "top_minus_v18": candidate.get(card_id, 0) - reference.get(card_id, 0),
        }
        for card_id in sorted(set(reference) | set(candidate))
        if reference.get(card_id, 0) != candidate.get(card_id, 0)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root", type=Path)
    parser.add_argument("--v18-deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.collection_root.resolve()
    indexed = read_csv(root / "indexes" / "episodes.csv")
    submission_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    submission_meta: dict[int, dict[str, Any]] = {}
    seen: set[tuple[int, int]] = set()

    for number, index_row in enumerate(indexed, start=1):
        submission_id = int(index_row["submission_id"])
        episode_id = int(index_row["episode_id"])
        relation = (submission_id, episode_id)
        if relation in seen:
            continue
        seen.add(relation)
        replay_path = root / index_row["replay_path"]
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        seat = int(index_row["seat_index"])
        analyzed = analyze_replay(replay, seat)
        analyzed["rank"] = int(index_row["leaderboard_rank"])
        analyzed["team_name"] = index_row["team_name"]
        analyzed["submission_id"] = submission_id
        analyzed["deck_hash"] = index_row["deck_hash"]
        submission_rows[submission_id].append(analyzed)
        submission_meta[submission_id] = {
            "rank": int(index_row["leaderboard_rank"]),
            "team_name": index_row["team_name"],
            "submission_id": submission_id,
            "deck_hash": index_row["deck_hash"],
        }
        if number % 100 == 0:
            print(f"analyzed_relations={number}", flush=True)

    all_rows = [
        row
        for submission_id in sorted(
            submission_rows,
            key=lambda item: submission_meta[item]["rank"],
        )
        for row in submission_rows[submission_id]
    ]
    top10 = [row for row in all_rows if row["rank"] <= 10]
    lower = [row for row in all_rows if row["rank"] > 10]
    wins = [row for row in all_rows if row["won"]]
    losses = [row for row in all_rows if not row["won"]]

    v18_deck = load_v18_deck(args.v18_deck.resolve())
    by_deck: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_deck[row["deck_hash"]].append(row)

    submissions = []
    for submission_id in sorted(
        submission_rows,
        key=lambda item: submission_meta[item]["rank"],
    ):
        rows = submission_rows[submission_id]
        submissions.append(
            {
                **submission_meta[submission_id],
                "deck": deck_counts(rows[0]),
                "deck_diff_from_v18": deck_diff(v18_deck, deck_counts(rows[0])),
                "metrics": compact_aggregate(rows),
            }
        )

    deck_groups = []
    for deck_hash, rows in sorted(by_deck.items(), key=lambda item: -len(item[1])):
        deck = deck_counts(rows[0])
        deck_groups.append(
            {
                "deck_hash": deck_hash,
                "submission_count": len({row["submission_id"] for row in rows}),
                "ranks": sorted({row["rank"] for row in rows}),
                "teams": sorted({row["team_name"] for row in rows}),
                "deck": deck,
                "deck_diff_from_v18": deck_diff(v18_deck, deck),
                "metrics": compact_aggregate(rows),
            }
        )

    report = {
        "collection_root": str(root),
        "relation_count": len(all_rows),
        "unique_episode_count": len({row["episode_id"] for row in all_rows}),
        "submission_count": len(submission_rows),
        "deck_count": len(by_deck),
        "v18_deck": v18_deck,
        "overall": compact_aggregate(all_rows),
        "top10": compact_aggregate(top10),
        "rank11_40": compact_aggregate(lower),
        "wins": compact_aggregate(wins),
        "losses": compact_aggregate(losses),
        "submissions": submissions,
        "deck_groups": deck_groups,
        "episodes": [
            {
                key: row[key]
                for key in (
                    "episode_id",
                    "rank",
                    "team_name",
                    "submission_id",
                    "deck_hash",
                    "won",
                    "end_reason",
                    "opponent_archetype",
                    "attack_turns",
                    "attack_opportunity_turns",
                    "post_first_main_turns",
                    "post_first_attack_turns",
                    "first_attack_own_main_turn",
                    "boss_plays",
                )
            }
            for row in all_rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"episodes", "submissions"}},
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
