"""Test whether Unfair Stamp is an actionable mirror lever after first attack.

The aggregate v15 mirror table shows the opponents that beat us using Unfair
Stamp more often.  A play-count difference is not a policy difference unless
the card was actually available, so this script measures three stages for both
seats in all 36 mirror games: live play offers, accepted live plays, and live
Petrel searches where Stamp was offered as a target.
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
for path in (ROOT, ROOT / "scripts",
             ROOT / "agents/grimmsnarl/grimmsnarl_ml_v16"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_v16_prize_conversion import (  # noqa: E402
    deck_label,
    matchup_of,
)

RUNS = (
    ("20260810_grimmsnarl_ml_v15_sub55404196", "55404196"),
    ("20260811_grimmsnarl_ml_v15_b_sub55409394", "55409394"),
)
MAIN = 0
CTX_TO_HAND = mf.CTX_TO_HAND
OPTION_PLAY = 7
STAMP = mf.UNFAIR_STAMP_ID
PETREL = mf.PETREL_ID


def _nested_id(value: Any) -> int:
    if isinstance(value, dict):
        if "id" in value:
            return mf._int(value.get("id"))
        for nested in value.values():
            found = _nested_id(nested)
            if found >= 0:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _nested_id(nested)
            if found >= 0:
                return found
    return -1


def _single_action(steps: list[Any], index: int, seat: int) -> int | None:
    if index + 1 >= len(steps) or seat >= len(steps[index + 1]):
        return None
    action = (steps[index + 1][seat] or {}).get("action")
    if not (isinstance(action, list) and len(action) == 1
            and isinstance(action[0], int)):
        return None
    return int(action[0])


def walk(replay: dict[str, Any], seat: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    live_offer_turns: set[int] = set()
    live_taken_turns: set[int] = set()
    steps = replay.get("steps") or []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        options = list(select.get("option") or [])
        chosen = _single_action(steps, index, seat)
        if chosen is None or not 0 <= chosen < len(options):
            continue

        context = int(select.get("context", -1))
        if context == MAIN:
            stamp_slots = []
            for slot, option in enumerate(options):
                if mf._int(option.get("type")) != OPTION_PLAY:
                    continue
                card = mf.candidate_card(current, option, select) or {}
                if int(card.get("id", -1)) == STAMP:
                    stamp_slots.append(slot)
            if stamp_slots:
                counts["live_play_offers"] += 1
                live_offer_turns.add(int(current.get("turn", -1)))
                if chosen in stamp_slots:
                    counts["live_play_taken"] += 1
                    live_taken_turns.add(int(current.get("turn", -1)))
                else:
                    counts["live_play_passed"] += 1

        if context == CTX_TO_HAND and _nested_id(select.get("effect")) == PETREL:
            resolved = [
                int((mf.resolve_option(current, select, option)[0] or {})
                    .get("id", -1))
                for option in options
            ]
            if STAMP not in resolved:
                continue
            players = current.get("players") or [{}, {}]
            me = players[seat]
            already = any(
                int(card.get("id", -1)) == STAMP
                for card in mf._cards(me, "hand")
            )
            counts["petrel_stamp_offers"] += 1
            if already:
                counts["petrel_stamp_already_in_hand"] += 1
            if resolved[chosen] == STAMP:
                counts["petrel_stamp_taken"] += 1
            else:
                counts["petrel_stamp_passed"] += 1
    counts["live_play_turns_offered"] = len(live_offer_turns)
    counts["live_play_turns_taken"] = len(live_taken_turns)
    counts["live_play_turns_missed"] = len(
        live_offer_turns - live_taken_turns
    )
    return counts


def _add(target: Counter[str], source: Counter[str]) -> None:
    target.update(source)
    target["games"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    blocks: dict[str, Counter[str]] = {
        name: Counter() for name in (
            "us", "them", "us_in_wins", "us_in_losses",
            "them_in_our_wins", "them_in_our_losses",
        )
    }
    episodes: list[dict[str, Any]] = []
    for run, submission in RUNS:
        run_dir = ROOT / "data/runs/grimmsnarl" / run
        for raw in csv.DictReader(
            (run_dir / "episodes.csv").open(encoding="utf-8-sig")
        ):
            if raw.get("state") != "COMPLETED":
                continue
            a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
            if a0 == a1:
                continue
            seat = 0 if a0 == submission else 1
            episode_id = int(raw["episode_id"])
            path = (run_dir / "episodes" / str(episode_id) / "replay"
                    / f"episode_{episode_id}.json")
            if not path.exists():
                continue
            replay = json.loads(path.read_text(encoding="utf-8"))
            steps = replay.get("steps") or []
            decks: list[list[int] | None] = [None, None]
            for side in (0, 1):
                action = (steps[1][side] or {}).get("action")
                if isinstance(action, list) and len(action) == 60:
                    decks[side] = [int(v) for v in action]
            if matchup_of(deck_label(decks[1 - seat])) != "mirror":
                continue
            rewards = replay.get("rewards") or [0, 0]
            won = rewards[seat] > rewards[1 - seat]
            ours, theirs = walk(replay, seat), walk(replay, 1 - seat)
            _add(blocks["us"], ours)
            _add(blocks["them"], theirs)
            _add(blocks["us_in_wins" if won else "us_in_losses"], ours)
            _add(
                blocks[
                    "them_in_our_wins" if won else "them_in_our_losses"
                ],
                theirs,
            )
            episodes.append({
                "episode_id": episode_id,
                "won": won,
                "us": dict(ours),
                "them": dict(theirs),
            })

    output = {
        "blocks": {name: dict(value) for name, value in blocks.items()},
        "episodes": episodes,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
