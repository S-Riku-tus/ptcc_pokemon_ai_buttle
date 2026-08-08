"""Where the 1100+ same-deck pilots gain, matchup by matchup, against us.

Two questions this answers and nothing in the line has answered together:

1. **Is 1100 reachable on this 60?** Not an opinion - how many pilots playing
   the identical deck hash are rated above it, and what win rate they run.
2. **Which matchups carry their surplus over us?** Our own 50 games are too
   thin per archetype to rank matchups, but the archive is not: split the
   field's games by opponent archetype, split the field by rating band, and the
   difference between the bands is where skill still buys something on this
   deck. Anything the elite band does *not* win either is a deck limit, and no
   amount of policy work will move it.

Reads only the replay header (team names, rewards, both 60-card lists) with the
prefix regex from ``ml.core.replay_io``, so it covers the whole archive in
about a minute instead of the half hour a full parse costs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import (  # noqa: E402
    deck_hash, extract_fast_header_from_file,
)

OUR_DECK_HASH = "9714ab5c3996f6cc"
CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) * z
    return [
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    ]


def archetype(deck: list[int] | None) -> str:
    """Name a deck by its heaviest evolution lines. Same rule as the v2 report."""
    if not deck:
        return "unknown"
    pokemon = Counter(
        card_id for card_id in deck
        if CARDS.get(card_id, {}).get("cardType") == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple:
        card_id, count = item
        card = CARDS[card_id]
        return (
            bool(card.get("stage2")),
            bool(card.get("megaEx") or card.get("ex")),
            bool(card.get("stage1")),
            count,
            int(card.get("hp", 0)),
        )

    return CARDS.get(max(pokemon.items(), key=key)[0], {}).get("name", "?")


def family(deck: list[int] | None) -> str:
    """Collapse the long tail into the families the meta actually has."""
    name = archetype(deck)
    ids = set(deck or ())
    if "Grimmsnarl" in name:
        return "Grimmsnarl (mirror)"
    if "Alakazam" in name or 743 in ids:
        return "Alakazam"
    if "Lopunny" in name or "Froslass" in name:
        return "Mega Lopunny / Froslass"
    if "Ogerpon" in name:
        return "Ogerpon"
    if "Kangaskhan" in name or "Crustle" in name:
        return "Kangaskhan / Crustle"
    if "Dragapult" in name or "Drakloak" in name:
        return "Dragapult"
    if "Lucario" in name or "Hariyama" in name:
        return "Mega Lucario"
    if "Dipplin" in name or "Thwackey" in name:
        return "Dipplin"
    if "Archaludon" in name or "Cinderace" in name:
        return "Archaludon"
    return f"other: {name}"


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(row["won"]) for row in rows)
    return {
        "games": len(rows), "wins": wins, "losses": len(rows) - wins,
        "win_rate": round(wins / len(rows), 4) if rows else None,
        "wilson95": wilson(wins, len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--ratings", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50" / "indexes"
        / "submissions.csv",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument("--elite", type=float, default=1100.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ratings: dict[int, float] = {}
    for row in csv.DictReader(args.ratings.open(encoding="utf-8-sig")):
        try:
            ratings[int(row["team_id"])] = float(row["submission_score"])
        except (KeyError, TypeError, ValueError):
            continue

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(
            encoding="utf-8-sig"
        )
    ):
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id, seat = int(raw["episode_id"]), int(raw["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = args.data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        try:
            header = extract_fast_header_from_file(path)
        except Exception:  # noqa: BLE001
            continue
        decks = header["decks"]
        if len(decks) < 2 or not decks[seat]:
            continue
        if deck_hash(decks[seat]) != OUR_DECK_HASH:
            continue
        rewards = header["rewards"]
        if rewards[seat] is None:
            continue
        other = rewards[1 - seat]
        team = int(raw["team_id"])
        rows.append({
            "who": "field",
            "team": team,
            "rating": ratings.get(team),
            "won": bool(rewards[seat] > (other if other is not None else 0)),
            "opponent_family": family(decks[1 - seat]),
            "opponent_hash": (
                deck_hash(decks[1 - seat]) if decks[1 - seat] else ""
            ),
        })

    ours: list[dict[str, Any]] = []
    for raw in csv.DictReader(
        (args.run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
            continue
        episode_id = int(raw["episode_id"])
        path = (
            args.run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        seat = 0 if a0 == args.submission else 1
        header = extract_fast_header_from_file(path)
        decks, rewards = header["decks"], header["rewards"]
        if rewards[seat] is None or not decks[1 - seat]:
            continue
        other = rewards[1 - seat]
        ours.append({
            "who": "v8", "team": None, "rating": None,
            "won": bool(rewards[seat] > (other if other is not None else 0)),
            "opponent_family": family(decks[1 - seat]),
            "opponent_hash": deck_hash(decks[1 - seat]),
        })

    elite = [r for r in rows if (r["rating"] or 0) >= args.elite]
    rest = [r for r in rows if r["rating"] is not None
            and r["rating"] < args.elite]

    families = sorted(
        {r["opponent_family"] for r in rows},
        key=lambda name: -sum(1 for r in rows if r["opponent_family"] == name),
    )
    by_family = {}
    for name in families:
        e = [r for r in elite if r["opponent_family"] == name]
        f = [r for r in rest if r["opponent_family"] == name]
        o = [r for r in ours if r["opponent_family"] == name]
        by_family[name] = {
            "elite": block(e), "rest": block(f), "v8": block(o),
            "share_of_field": round(
                sum(1 for r in rows if r["opponent_family"] == name)
                / len(rows), 4
            ),
            "elite_minus_rest": (
                round(block(e)["win_rate"] - block(f)["win_rate"], 4)
                if e and f else None
            ),
        }

    per_pilot = {}
    for team in sorted({r["team"] for r in rows}):
        games = [r for r in rows if r["team"] == team]
        per_pilot[str(team)] = {
            "rating": ratings.get(team), **block(games),
        }

    report = {
        "elite_threshold": args.elite,
        "field_games": len(rows),
        "our_games": len(ours),
        "pilots_total": len(per_pilot),
        "pilots_at_or_above_threshold": sum(
            1 for row in per_pilot.values()
            if (row["rating"] or 0) >= args.elite
        ),
        "overall": {
            "elite": block(elite), "rest": block(rest), "v8": block(ours),
        },
        "per_pilot": per_pilot,
        "by_opponent_family": by_family,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"field games={len(rows)}  our games={len(ours)}  "
          f"pilots={len(per_pilot)} "
          f"(>={args.elite:.0f}: {report['pilots_at_or_above_threshold']})")
    print()
    print("same-deck pilots, by rating:")
    for team, row in sorted(
        per_pilot.items(), key=lambda kv: -(kv[1]["rating"] or 0)
    ):
        print(f"  {team:>10} {(row['rating'] or 0):7.1f}  "
              f"{row['wins']:4d}-{row['losses']:<4d} {row['win_rate']} "
              f"{row['wilson95']}")
    print(f"  {'v8 (ours)':>10} {0:7.1f}  "
          f"{block(ours)['wins']:4d}-{block(ours)['losses']:<4d} "
          f"{block(ours)['win_rate']} {block(ours)['wilson95']}")
    print()
    print(f"{'opponent family':28s} {'share':>6} {'elite':>14} "
          f"{'rest':>14} {'gain':>6} {'v8':>12}")
    for name, row in by_family.items():
        if row["elite"]["games"] < 30:
            continue
        print(
            f"{name:28s} {row['share_of_field']:6.3f} "
            f"{row['elite']['win_rate']:6.3f}({row['elite']['games']:5d}) "
            f"{row['rest']['win_rate']:6.3f}({row['rest']['games']:5d}) "
            f"{(row['elite_minus_rest'] or 0):+6.3f} "
            f"{_fmt(row['v8']['win_rate'])}({row['v8']['games']:3d})"
        )
    return 0


def _fmt(value) -> str:
    return f"{value:6.3f}" if value is not None else "     -"


if __name__ == "__main__":
    raise SystemExit(main())
