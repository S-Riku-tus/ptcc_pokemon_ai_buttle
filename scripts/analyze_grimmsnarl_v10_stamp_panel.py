"""What every advisor would do on v8's own dead-Unfair-Stamp offers.

Joins the decision dump from ``probe_grimmsnarl_v10_advisors.py`` to a fresh
pass over the same replays that labels each Petrel search with whether the
Unfair Stamp on offer is playable this turn. The dump has what each advisor
picked; the pass has the one board fact that decides whether the pick is a dead
draw. Together they answer the only questions a class residual needs:

* how often is the class even reached (its override *ceiling*);
* which pilots refuse the dead Stamp on the boards v8 reaches, as opposed to on
  their own;
* how a k-of-n consensus among them would behave, per threshold, and what it
  would take instead.
"""

from __future__ import annotations

import argparse
import csv
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
from analyze_grimmsnarl_v10_stamp import nested_id  # noqa: E402

CTX_TO_HAND = 7
STAMP_ID = mf.UNFAIR_STAMP_ID
PETREL_ID = mf.PETREL_ID


def label(replay: dict[str, Any], seat: int) -> dict[int, dict[str, Any]]:
    """Per step index: the Petrel-search facts for that decision."""
    out: dict[int, dict[str, Any]] = {}
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
        current = observation.get("current") or {}
        players = current.get("players") or []
        your = int(current.get("yourIndex", seat))
        if not select or len(players) < 2 or your >= len(players):
            continue
        me, opponent = players[your], players[1 - your]
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
        in_hand = Counter()
        for card in me.get("hand") or []:
            if isinstance(card, dict):
                in_hand[int(card.get("id", -1))] += 1
        earlier = [t for t in opponent_prize_by_turn if t < turn]
        prior = opponent_prize_by_turn[max(earlier)] if earlier else 6
        out[index] = {
            "offered_ids": ids,
            "stamp_offered": STAMP_ID in ids,
            "stamp_in_hand": bool(in_hand[STAMP_ID]),
            "stamp_live": opponent_prize < prior,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument(
        "--panel", default="v9,t16422241,t16452116,t16561259,t16371703",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.decisions.open(encoding="utf-8")
    ]
    panel = [name.strip() for name in args.panel.split(",") if name.strip()]

    labels: dict[int, dict[int, dict[str, Any]]] = {}
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
        labels[episode_id] = label(
            json.loads(path.read_text(encoding="utf-8")), seat
        )

    events: list[dict[str, Any]] = []
    boss_events: list[dict[str, Any]] = []
    for row in rows:
        if row["context"] != CTX_TO_HAND:
            continue
        fact = labels.get(row["episode_id"], {}).get(row["step"])
        if fact is None:
            continue
        if mf.BOSS_ID in fact["offered_ids"]:
            boss_events.append({
                "episode_id": row["episode_id"],
                "turn": row["turn"],
                "v8_card": row["advisors"]["v8"].get("card"),
                "panel_cards": {
                    name: (row["advisors"].get(name) or {}).get("card")
                    for name in panel
                },
            })
        if not fact["stamp_offered"] or fact["stamp_in_hand"]:
            continue
        entry = {
            "episode_id": row["episode_id"],
            "turn": row["turn"],
            "won": row["won"],
            "went_first": row["went_first"],
            "opponent_deck_hash": row["opponent_deck_hash"],
            "live": fact["stamp_live"],
            "options": row["options"],
            "v8_card": row["advisors"]["v8"].get("card"),
            "v8_margin": row["advisors"]["v8"].get("margin"),
            "panel_cards": {
                name: (row["advisors"].get(name) or {}).get("card")
                for name in panel
            },
            "panel_slots": {
                name: (row["advisors"].get(name) or {}).get("chosen")
                for name in panel
            },
            "v8_slot": row["advisors"]["v8"].get("chosen"),
        }
        events.append(entry)

    dead = [e for e in events if not e["live"]]
    live = [e for e in events if e["live"]]

    def take_rate(entries: list[dict[str, Any]], who: str) -> Any:
        if not entries:
            return None
        got = sum(
            int((e["panel_cards"][who] if who in e["panel_cards"]
                 else e["v8_card"]) == STAMP_ID)
            for e in entries
        )
        return round(got / len(entries), 4)

    who_list = ["v8"] + panel
    def boss_rate(who: str) -> Any:
        if not boss_events:
            return None
        got = sum(
            int((e["panel_cards"][who] if who in e["panel_cards"]
                 else e["v8_card"]) == mf.BOSS_ID)
            for e in boss_events
        )
        return round(got / len(boss_events), 4)

    per_advisor = {
        who: {
            "dead_take_rate": take_rate(dead, who),
            "live_take_rate": take_rate(live, who),
            "petrel_boss_take_rate": boss_rate(who),
        }
        for who in who_list
    }

    thresholds: dict[str, Any] = {}
    for need in range(1, len(panel) + 1):
        fired = 0
        alternatives: Counter[int] = Counter()
        for event in dead:
            if event["v8_card"] != STAMP_ID:
                continue
            votes: Counter[int] = Counter()
            for name in panel:
                slot = event["panel_slots"].get(name)
                if slot is None or event["panel_cards"].get(name) == STAMP_ID:
                    continue
                votes[slot] += 1
            if not votes:
                continue
            slot, count = votes.most_common(1)[0]
            if count >= need:
                fired += 1
                for name in panel:
                    if event["panel_slots"].get(name) == slot:
                        alternatives[event["panel_cards"][name]] += 1
                        break
        thresholds[str(need)] = {
            "fires": fired,
            "of_dead_stamp_takes": sum(
                int(e["v8_card"] == STAMP_ID) for e in dead
            ),
            "alternatives": dict(alternatives.most_common()),
        }

    report = {
        "decisions_file": str(args.decisions),
        "panel": panel,
        "petrel_stamp_offers": len(events),
        "dead_offers": len(dead),
        "live_offers": len(live),
        "petrel_boss_offers": len(boss_events),
        "per_advisor": per_advisor,
        "consensus_thresholds": thresholds,
        "events": events,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"petrel searches offering a Stamp not already in hand: "
          f"{len(events)}  dead={len(dead)} live={len(live)}")
    print(f"petrel searches offering Boss: {len(boss_events)}")
    print(f"{'advisor':14s} {'dead take':>10} {'live take':>10} "
          f"{'boss take':>10}")
    for who in who_list:
        row = per_advisor[who]
        print(f"{who:14s} {str(row['dead_take_rate']):>10} "
              f"{str(row['live_take_rate']):>10} "
              f"{str(row['petrel_boss_take_rate']):>10}")
    print()
    for need, row in thresholds.items():
        print(f"  need {need}/{len(panel)}: fires {row['fires']} of "
              f"{row['of_dead_stamp_takes']} dead-Stamp takes -> "
              f"{row['alternatives']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
