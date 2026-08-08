"""Turns ended on a single body while a Basic was still benchable.

Episode 90672919 was lost on turn 5 with Grimmsnarl ex alone in play and an
empty Bench: the knockout took the game, not the prizes - both sides were still
on five. That is not a preference the ranker could be wrong about, it is an
immediate loss condition, so if it repeats it is exactly the shape a residual
may gate: provable, narrow, and independent of any imitation evidence.

Counted per own turn, on our games and on the field's games with the same 60,
so "v8 does this and the field does not" is a measured statement rather than an
inference from one replay.
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
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"


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


def scan(replay: dict[str, Any], seat: int) -> list[dict[str, Any]]:
    """One row per own turn that ended with at most one body in play."""
    steps = replay.get("steps") or []
    out: list[dict[str, Any]] = []
    turn = None
    bench_offered = False
    last: dict[str, Any] | None = None
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
            continue
        options = list(select.get("option") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        if not options or not isinstance(action, list) or len(action) != 1:
            continue
        if not isinstance(action[0], int) or not 0 <= action[0] < len(options):
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) < 2:
            continue
        this_turn = int(current.get("turn", -1))
        if this_turn != turn:
            if last is not None:
                out.append(last)
            turn = this_turn
            bench_offered = False
            last = None
        actions = [mf.action_type(current, o, select) for o in options]
        if "bench" in actions:
            bench_offered = True
        played = action[0]
        if actions[played] not in ("end", "attack"):
            continue
        me = players[seat]
        in_play = len(mf._in_play(me))
        last = {
            "turn": this_turn,
            "in_play": in_play,
            "bench_offered_this_turn": bench_offered,
            "bench_offered_now": "bench" in actions,
            "closing_action": actions[played],
            "hand_basics": sum(
                int(int(card.get("id", -1)) in mf.BASIC_POKEMON_IDS)
                for card in (me.get("hand") or [])
                if isinstance(card, dict)
            ),
        }
    if last is not None:
        out.append(last)
    return out


def tally(rows: list[dict[str, Any]], counts: Counter[str], tag: str) -> None:
    for row in rows:
        counts[f"{tag}|closed_turns"] += 1
        if row["in_play"] <= 1:
            counts[f"{tag}|lone_body"] += 1
            if row["bench_offered_now"]:
                counts[f"{tag}|lone_body_bench_available_now"] += 1
            if row["bench_offered_this_turn"]:
                counts[f"{tag}|lone_body_bench_available_turn"] += 1
            if row["hand_basics"] > 0:
                counts[f"{tag}|lone_body_basic_in_hand"] += 1


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
    parser.add_argument("--field-limit", type=int, default=400)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    events: list[dict[str, Any]] = []
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
        rows = scan(replay, seat)
        first = turn_order(replay, seat)
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        won = bool(rewards[seat] > (other if other is not None else 0))
        tally(rows, counts, "v8")
        tally(rows, counts, f"v8_{'first' if first else 'second'}")
        for row in rows:
            if row["in_play"] <= 1 and row["bench_offered_now"]:
                events.append({
                    "episode_id": episode_id, "won": won,
                    "went_first": first, **row,
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
        tally(scan(replay, seat), counts, "field")
        read += 1

    report = {
        "field_replays_read": read,
        "counts": dict(counts),
        "rates": {
            tag: {
                "closed_turns": counts[f"{tag}|closed_turns"],
                "lone_body": counts[f"{tag}|lone_body"],
                "lone_body_rate": (
                    round(
                        counts[f"{tag}|lone_body"]
                        / counts[f"{tag}|closed_turns"], 4
                    )
                    if counts[f"{tag}|closed_turns"] else None
                ),
                "lone_body_bench_available_now":
                    counts[f"{tag}|lone_body_bench_available_now"],
                "lone_body_bench_available_turn":
                    counts[f"{tag}|lone_body_bench_available_turn"],
                "lone_body_basic_in_hand":
                    counts[f"{tag}|lone_body_basic_in_hand"],
            }
            for tag in ("v8", "v8_first", "v8_second", "field")
        },
        "v8_events": events,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["rates"], indent=2))
    print(f"v8 avoidable lone-body closes: {len(events)}")
    for event in events:
        print(f"  ep={event['episode_id']} turn={event['turn']} "
              f"won={event['won']} first={event['went_first']} "
              f"close={event['closing_action']} "
              f"basics_in_hand={event['hand_basics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
