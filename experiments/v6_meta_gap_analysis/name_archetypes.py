"""Name every opponent deck hash and weight it by the *current* leaderboard.

measure_matchup_gap.py reports win rate per opponent deck hash. Two things are
missing before that can be turned into a priority list:

* what the hash *is* - the headline Pokemon of the list, from cards.json;
* how much of the field it currently is - our corpus reflects the field of late
  July, and by 2026-08-06 the top 40 had turned over.

It also answers the confound in our own numbers: our ladder runs win 63-70%
while the top pilots win 54-62%, because a 965-rated agent is matched against
860-rated opponents. Win rate by opponent initial score says how we do against
the pool that actually populates 1100+.

Usage:
    python experiments/v6_meta_gap_analysis/name_archetypes.py \
        --matchup matchup_gap.json --snapshot refresh_opportunity.json \
        --out archetypes.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402

DECK_HASH = "9714ab5c3996f6cc"
CORPUS = ROOT / "data" / "kaggle_grimmsnarl_top50"
RUNS = ROOT / "data" / "runs" / "grimmsnarl"
POKEMON_TYPE = 0  # cardType 0 is a Pokemon; 1/3/5 are item/supporter/energy


def card_names() -> dict[int, dict]:
    path = ROOT / "vendor" / "cg" / "cards.json"
    cards = json.loads(path.read_text("utf-8"))
    return {int(card["cardId"]): card for card in cards}


def representative_decks() -> dict[str, list[int]]:
    """One 60-card list per opponent deck hash, from the corpus replays."""
    decks: dict[str, list[int]] = {}
    for row in csv.DictReader(
        open(CORPUS / "indexes" / "replay_index.csv", encoding="utf-8-sig")
    ):
        if row["deck_hash"] != DECK_HASH:
            continue
        path = CORPUS / Path(row["replay_path"].replace(chr(92), "/"))
        if not path.exists():
            continue
        seat = int(row["seat_index"])
        try:
            head = extract_fast_header_from_file(path)
        except Exception:
            continue
        hashes = head.get("deck_hashes") or ["", ""]
        if len(hashes) < 2 or hashes[seat] != DECK_HASH:
            continue
        other = hashes[1 - seat]
        if other and other not in decks:
            decks[other] = head["decks"][1 - seat]
    return decks


def headline(
    deck: list[int], cards: dict[int, dict], limit: int = 4
) -> list[str]:
    """The deck's identifying Pokemon: ex/stage-2 first, then by copy count."""
    counts = Counter(deck)
    pokemon = [
        (card_id, count) for card_id, count in counts.items()
        if cards.get(card_id, {}).get("cardType") == POKEMON_TYPE
    ]

    def key(item: tuple[int, int]) -> tuple:
        card = cards[item[0]]
        return (
            0 if card.get("ex") or card.get("megaEx") else
            1 if card.get("stage2") else
            2 if card.get("stage1") else 3,
            -item[1],
        )

    return [
        f"{cards[card_id]['name']} x{count}"
        for card_id, count in sorted(pokemon, key=key)[:limit]
    ]


def our_games_by_opponent_score() -> dict:
    """Our pooled ladder record, banded by the opponent's pre-game rating."""
    bands = [(0, 900), (900, 1000), (1000, 1050), (1050, 1100), (1100, 9999)]
    per_band: dict[str, Counter] = defaultdict(Counter)
    per_run: dict[str, Counter] = defaultdict(Counter)
    for run_dir in sorted(RUNS.glob("20260806_grimmsnarl_ml_v5_*")) + \
            sorted(RUNS.glob("20260806_grimmsnarl_ml_v4_5_*")) + \
            sorted(RUNS.glob("20260805_grimmsnarl_ml_v4_*")):
        seats = {
            row["episode_id"]: row["detected_submission_agent_index"]
            for row in csv.DictReader(
                open(run_dir / "manifest.csv", encoding="utf-8-sig")
            )
        }
        for row in csv.DictReader(
            open(run_dir / "episodes.csv", encoding="utf-8-sig")
        ):
            episode = row["episode_id"]
            seat = seats.get(episode, "")
            if seat not in ("0", "1"):
                continue
            if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
                continue
            try:
                mine = float(row[f"agent_{seat}_initial_score"])
                after = float(row[f"agent_{seat}_updated_score"])
                theirs = float(row[f"agent_{1 - int(seat)}_initial_score"])
            except (KeyError, ValueError):
                continue
            won = after > mine
            label = next(
                f"{low}-{high}" for low, high in bands if low <= theirs < high
            )
            per_band[label]["games"] += 1
            per_band[label]["wins"] += int(won)
            per_run[run_dir.name]["games"] += 1
            per_run[run_dir.name]["opponent_score_sum"] += theirs
    return {
        "by_opponent_score_band": {
            band: {
                "games": counts["games"], "wins": counts["wins"],
                "win_rate": round(counts["wins"] / counts["games"], 4),
            }
            for band, counts in sorted(per_band.items())
        },
        "mean_opponent_score": {
            run: round(counts["opponent_score_sum"] / counts["games"], 1)
            for run, counts in per_run.items() if counts["games"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    matchup = json.loads(args.matchup.read_text("utf-8"))
    snapshot = json.loads(args.snapshot.read_text("utf-8"))
    cards = card_names()
    decks = representative_decks()

    top40: Counter = Counter()
    best_rank: dict[str, int] = {}
    for row in snapshot["findings"]:
        deck = row["deck_hash"]
        top40[deck] += 1
        best_rank.setdefault(deck, row["rank"])

    field = matchup["labels"]["field:same_deck_corpus"]["by_opponent_deck"]
    ours: Counter = Counter()
    our_wins: Counter = Counter()
    for label, row in matchup["labels"].items():
        if not label.startswith("ours:"):
            continue
        for deck, cell in row["by_opponent_deck"].items():
            ours[deck] += cell["games"]
            our_wins[deck] += cell["wins"]
    tops = [
        label for label in matchup["labels"]
        if label in ("pilot:16452116", "pilot:16422241", "pilot:16561259")
    ]
    top_games: Counter = Counter()
    top_wins: Counter = Counter()
    for label in tops:
        for deck, cell in matchup["labels"][label]["by_opponent_deck"].items():
            top_games[deck] += cell["games"]
            top_wins[deck] += cell["wins"]

    rows = []
    for deck, cell in field.items():
        rows.append({
            "deck_hash": deck,
            "is_mirror": deck == DECK_HASH,
            "headline": headline(decks.get(deck, []), cards),
            "top40_submissions": top40.get(deck, 0),
            "top40_best_rank": best_rank.get(deck),
            "field_games": cell["games"],
            "field_win_rate": cell["win_rate"],
            "top3_pilot_games": top_games.get(deck, 0),
            "top3_pilot_win_rate": (
                round(top_wins[deck] / top_games[deck], 4)
                if top_games.get(deck) else None
            ),
            "our_games": ours.get(deck, 0),
            "our_win_rate": (
                round(our_wins[deck] / ours[deck], 4)
                if ours.get(deck) else None
            ),
        })
    rows.sort(key=lambda row: (-row["top40_submissions"], -row["field_games"]))

    report = {
        "deck_hash": DECK_HASH,
        "field_label": "field:same_deck_corpus",
        "our_runs": [
            label for label in matchup["labels"] if label.startswith("ours:")
        ],
        "top3_pilots": tops,
        "archetypes": rows,
        "our_ladder_pool": our_games_by_opponent_score(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"{'hash':<17}{'t40':>4}{'rank':>5}{'fieldN':>7}{'field%':>7}"
        f"{'top3N':>6}{'top3%':>7}{'oursN':>6}{'ours%':>7}  headline"
    )
    for row in rows:
        if not row["top40_submissions"] and row["field_games"] < 40:
            continue

        def pct(value):
            return f"{value:>7.2f}" if value is not None else " " * 7

        print(
            f"{row['deck_hash']:<17}{row['top40_submissions']:>4}"
            f"{row['top40_best_rank'] or 0:>5}{row['field_games']:>7}"
            f"{pct(row['field_win_rate'])}{row['top3_pilot_games']:>6}"
            f"{pct(row['top3_pilot_win_rate'])}{row['our_games']:>6}"
            f"{pct(row['our_win_rate'])}  {', '.join(row['headline'][:3])}"
        )
    print()
    print(json.dumps(report["our_ladder_pool"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
