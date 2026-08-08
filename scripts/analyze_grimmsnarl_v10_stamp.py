"""Re-verify the Petrel -> Unfair Stamp gap before shipping it.

v6 measured this class, found the only rating gradient in the line that
survives a significance test, and then deliberately did not ship it so that one
ladder run measured one change. v7 shipped a value-search layer instead and v8
removed it, so the class has still never been deployed and it is pre-registered
as the next one.

Nothing about that is taken on trust here. This recomputes, from the replays:

* per pilot, the rate of taking an Unfair Stamp out of a Petrel search *on a
  turn when it cannot be played* - the opponent took no prize during their last
  turn, so the "only if your opponent took a Prize card during their last turn"
  clause is false and the card is a dead draw this turn;
* the Spearman gradient of that rate against pilot rating, with its p;
* the same rate for the v8 ladder run, so the gap is stated on the boards v8
  actually reaches rather than on v6's.

The live case is reported beside it as the control: a class escalation that
also suppressed *playable* Stamps would be a loss, not a fix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_v10_turn_order import spearman  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
CTX_TO_HAND = 7
STAMP_ID = mf.UNFAIR_STAMP_ID
PETREL_ID = mf.PETREL_ID
BOSS_ID = mf.BOSS_ID


def deck_hash(card_ids: list[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def nested_id(value: Any) -> int:
    if isinstance(value, dict):
        if "id" in value:
            try:
                return int(value["id"])
            except (TypeError, ValueError):
                return -1
        for item in value.values():
            found = nested_id(item)
            if found >= 0:
                return found
    elif isinstance(value, list):
        for item in value:
            found = nested_id(item)
            if found >= 0:
                return found
    return -1


def scan(replay: dict[str, Any], seat: int) -> Counter:
    """Petrel search decisions in one game, split by whether a Stamp is live."""
    counts: Counter = Counter()
    steps = replay.get("steps") or []
    opponent_prize_by_turn: dict[int, int] = {}
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if not select:
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        your = int(current.get("yourIndex", seat))
        if len(players) < 2 or your >= len(players):
            continue
        me, opponent = players[your], players[1 - your]
        action = (steps[index + 1][seat] or {}).get("action")
        if not isinstance(action, list) or not action:
            continue
        turn = int(current.get("turn", -1))
        opponent_prize = len(opponent.get("prize") or [])
        opponent_prize_by_turn.setdefault(turn, opponent_prize)
        if int(select.get("context", -1)) != CTX_TO_HAND:
            continue
        if nested_id(select.get("effect")) != PETREL_ID:
            continue
        options = list(select.get("option") or [])
        ids = [
            int((mf.resolve_option(current, select, option)[0] or {})
                .get("id", -1))
            for option in options
        ]
        picked = {
            ids[slot] for slot in action
            if isinstance(slot, int) and 0 <= slot < len(ids)
        }
        in_hand = Counter()
        for card in me.get("hand") or []:
            if isinstance(card, dict):
                in_hand[int(card.get("id", -1))] += 1
        counts["petrel_searches"] += 1
        counts["max_count_one"] += int(int(select.get("maxCount") or 0) == 1)
        if STAMP_ID in ids and not in_hand[STAMP_ID]:
            earlier = [t for t in opponent_prize_by_turn if t < turn]
            prior = (
                opponent_prize_by_turn[max(earlier)] if earlier else 6
            )
            live = opponent_prize < prior
            key = "live" if live else "dead"
            counts[f"stamp_offered_{key}"] += 1
            counts[f"stamp_taken_{key}"] += int(STAMP_ID in picked)
        if BOSS_ID in ids:
            counts["boss_offered"] += 1
            counts["boss_taken"] += int(BOSS_ID in picked)
    return counts


def rate(counts: Counter, numerator: str, denominator: str):
    total = counts[denominator]
    return round(counts[numerator] / total, 4) if total else None


def block(counts: Counter) -> dict[str, Any]:
    return {
        "petrel_searches": counts["petrel_searches"],
        "max_count_one": counts["max_count_one"],
        "dead_offered": counts["stamp_offered_dead"],
        "dead_take_rate": rate(counts, "stamp_taken_dead",
                               "stamp_offered_dead"),
        "live_offered": counts["stamp_offered_live"],
        "live_take_rate": rate(counts, "stamp_taken_live",
                               "stamp_offered_live"),
        "boss_offered": counts["boss_offered"],
        "boss_take_rate": rate(counts, "boss_taken", "boss_offered"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--ratings", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50" / "indexes"
        / "submissions.csv",
    )
    parser.add_argument("--field-limit", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ours: Counter = Counter()
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
        ours += scan(json.loads(path.read_text(encoding="utf-8")), seat)

    ratings: dict[int, float] = {}
    for row in csv.DictReader(args.ratings.open(encoding="utf-8-sig")):
        try:
            ratings[int(row["team_id"])] = float(row["submission_score"])
        except (KeyError, TypeError, ValueError):
            continue

    per_team: dict[int, Counter] = {}
    field: Counter = Counter()
    read = 0
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(
            encoding="utf-8-sig"
        )
    ):
        if args.field_limit and read >= args.field_limit:
            break
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id = int(raw["episode_id"])
        seat = int(raw["seat_index"])
        path = args.data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        steps = replay.get("steps") or []
        deck = None
        if len(steps) > 1:
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                deck = [int(v) for v in action]
        if deck is None or deck_hash(deck) != OUR_DECK_HASH:
            continue
        counts = scan(replay, seat)
        team = int(raw["team_id"])
        per_team.setdefault(team, Counter())
        per_team[team] += counts
        field += counts
        read += 1

    teams = {
        str(team): {"rating": ratings.get(team), **block(counts)}
        for team, counts in sorted(
            per_team.items(), key=lambda kv: -(ratings.get(kv[0]) or 0)
        )
    }
    gradient = {
        kind: spearman([
            (row["rating"], row[f"{kind}_take_rate"], row[f"{kind}_offered"])
            for row in teams.values()
            if row["rating"] is not None
            and row[f"{kind}_take_rate"] is not None
            and row[f"{kind}_offered"] >= minimum
        ])
        for kind, minimum in (("dead", 10), ("live", 10), ("boss", 10))
    }
    report = {
        "field_replays_read": read,
        "v8_ladder": block(ours),
        "field": block(field),
        "per_team": teams,
        "rating_gradient": gradient,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"field replays read={read}")
    print("v8 ladder :", json.dumps(report["v8_ladder"]))
    print("field     :", json.dumps(report["field"]))
    print()
    print(f"{'team':>10} {'rating':>7} {'dead':>7} {'n':>5} "
          f"{'live':>7} {'n':>5} {'boss':>7} {'n':>5}")
    for team, row in teams.items():
        def show(value):
            return f"{value:7.3f}" if value is not None else "      -"
        print(f"{team:>10} {(row['rating'] or 0):7.1f} "
              f"{show(row['dead_take_rate'])} {row['dead_offered']:5d} "
              f"{show(row['live_take_rate'])} {row['live_offered']:5d} "
              f"{show(row['boss_take_rate'])} {row['boss_offered']:5d}")
    print()
    for kind, value in gradient.items():
        print(f"  {kind:5s} vs rating: n={value['n']} rho={value['rho']} "
              f"p={value['p']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
