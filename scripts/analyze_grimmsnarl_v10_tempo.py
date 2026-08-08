"""How fast the deck comes online, by turn order, for us and for the field.

The per-turn take rates say v8 takes the resources it is offered going second
about as often as the field does. That leaves the other half of the question:
whether it is *offered* them as early. This measures the tempo landmarks -
first turn Grimmsnarl ex is in play, first attack, first prize - which are
where a "we never got going" loss and a "we lost the race" loss separate.

Uses the same turn-order rule as the rest of this line: ``firstPlayer`` read
from a late step, never the seat index and never the first observation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
GRIMMSNARL_EX_ID = mf.GRIMMSNARL_EX_ID


def deck_hash(card_ids: list[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def turn_order(replay: dict[str, Any], seat: int) -> bool | None:
    for step in reversed(replay.get("steps") or []):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(current, dict) and current.get("players"):
            first = int(current.get("firstPlayer", -1))
            return (first == seat) if first >= 0 else None
    return None


def landmarks(replay: dict[str, Any], seat: int) -> dict[str, Any]:
    """First turn of: Grimmsnarl ex in play, an attack, a prize taken."""
    steps = replay.get("steps") or []
    first_grimm = first_attack = first_prize = None
    prizes_before = None
    own_turns = 0
    seen_turns: set[int] = set()
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        observation = record.get("observation") or {}
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) < 2:
            continue
        turn = int(current.get("turn", -1))
        me = players[seat]
        prizes_left = len(me.get("prize") or [])
        if prizes_before is None:
            prizes_before = prizes_left
        elif prizes_left < prizes_before:
            if first_prize is None:
                first_prize = turn
            prizes_before = prizes_left
        if first_grimm is None and any(
            int(card.get("id", -1)) == GRIMMSNARL_EX_ID
            for card in mf._in_play(me)
        ):
            first_grimm = turn
        if record.get("status") != "ACTIVE":
            continue
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
            continue
        if turn not in seen_turns:
            seen_turns.add(turn)
            own_turns += 1
        options = list(select.get("option") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        if not options or not isinstance(action, list) or len(action) != 1:
            continue
        if not isinstance(action[0], int) or not 0 <= action[0] < len(options):
            continue
        if first_attack is None and mf.action_type(
            current, options[action[0]], select
        ) == "attack":
            first_attack = turn
    return {
        "first_grimmsnarl_turn": first_grimm,
        "first_attack_turn": first_attack,
        "first_prize_turn": first_prize,
        "own_turns": own_turns,
    }


def describe(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if row.get(key) is not None]
    return {
        "n": len(values),
        "missing": len(rows) - len(values),
        "median": statistics.median(values) if values else None,
        "mean": round(statistics.fmean(values), 2) if values else None,
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
        "--field-limit", type=int, default=400,
        help="Field replays are 5 MB each; this is a declared sample, and "
             "the number actually read is written into the report.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
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
        replay = json.loads(path.read_text(encoding="utf-8"))
        first = turn_order(replay, seat)
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        rows.append({
            "who": "v8",
            "episode_id": episode_id,
            "went_first": first,
            "won": bool(rewards[seat] > (other if other is not None else 0)),
            **landmarks(replay, seat),
        })

    read = 0
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(
            encoding="utf-8-sig"
        )
    ):
        if read >= args.field_limit:
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
        first = turn_order(replay, seat)
        if first is None:
            continue
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        rows.append({
            "who": "field",
            "episode_id": episode_id,
            "went_first": first,
            "won": bool((rewards[seat] or 0)
                        > (other if other is not None else 0)),
            **landmarks(replay, seat),
        })
        read += 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["went_first"] is None:
            continue
        order = "first" if row["went_first"] else "second"
        groups[f"{row['who']}_{order}"].append(row)
        groups[f"{row['who']}_{order}_{'won' if row['won'] else 'lost'}"].append(row)

    keys = (
        "first_grimmsnarl_turn", "first_attack_turn", "first_prize_turn",
        "own_turns",
    )
    summary = {
        name: {
            "games": len(rows_),
            "wins": sum(bool(r["won"]) for r in rows_),
            **{key: describe(rows_, key) for key in keys},
        }
        for name, rows_ in sorted(groups.items())
    }
    report = {
        "field_replays_read": read,
        "field_limit": args.field_limit,
        "summary": summary,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"field replays read={read} (limit {args.field_limit})")
    print(f"{'group':26s} {'n':>4} {'win':>4} "
          f"{'grimm':>7} {'attack':>7} {'prize':>7} {'turns':>7}")
    for name in sorted(summary):
        row = summary[name]
        print(
            f"{name:26s} {row['games']:4d} {row['wins']:4d} "
            f"{str(row['first_grimmsnarl_turn']['median']):>7} "
            f"{str(row['first_attack_turn']['median']):>7} "
            f"{str(row['first_prize_turn']['median']):>7} "
            f"{str(row['own_turns']['median']):>7}"
        )
    print()
    print("missing (never reached) counts:")
    for name in sorted(summary):
        row = summary[name]
        print(f"  {name:26s} no_grimm={row['first_grimmsnarl_turn']['missing']:3d} "
              f"no_attack={row['first_attack_turn']['missing']:3d} "
              f"no_prize={row['first_prize_turn']['missing']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
