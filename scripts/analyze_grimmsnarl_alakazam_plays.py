"""What the field plays on its first turns into Alakazam, and what we do not.

The divergence probe asks "would v8 pick something else on the field's board".
This asks the other half, which needs no model and so can run over every replay:
**on each own turn, was the card even on offer, and if so was it taken?**

That split is the whole point. Five versions have optimised take rates and the
rating scan found no gradient in any of them, so a tempo gap that shows up as a
*take* difference is a policy bug we already know how to move, and one that
shows up as an *offer* difference is a deck-flow problem - we never drew or
searched the card - which no ranker override can fix.

Reported per own turn, not per decision: a supporter can be played once a turn,
so "per decision" divides by an option count that has nothing to do with the
question (see the per-turn denominator note in the v5 line).

Cohorts are the same three as the divergence probe plus our own runs, and our
side pools v7 and v8 because they share the setup line and five games is not a
sample. Turn order always comes from ``firstPlayer``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from analyze_grimmsnarl_alakazam_stage1 import (  # noqa: E402
    OUR_DECK_HASH, cohort_of, replay_meta,
)

CARDS = {
    int(card["cardId"]): card.get("name", "?")
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
MAIN = mf.MAIN_CONTEXT


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not n or not r1 or not c1 or r1 == n or c1 == n:
        return 1.0

    def probability(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    reference = probability(a)
    return round(min(1.0, sum(
        probability(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1)
        if probability(x) <= reference + 1e-12
    )), 4)


def per_turn(
    replay: dict[str, Any], seat: int, max_turn: int
) -> list[dict[str, Any]]:
    """One record per own turn: what was offered, what was played, board size."""
    steps = replay.get("steps") or []
    turns: dict[int, dict[str, Any]] = {}
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != MAIN:
            continue
        options = list(select.get("option") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        current = observation.get("current") or {}
        players = current.get("players") or []
        if not options or len(players) < 2:
            continue
        turn = int(current.get("turn", -1))
        if turn > max_turn:
            break
        me = players[seat]
        entry = turns.setdefault(turn, {
            "turn": turn, "offered": set(), "played": set(),
            "offered_actions": set(), "played_actions": set(),
            "bodies": len(mf._in_play(me)),
            "hand": len(me.get("hand") or []),
            "deck": len(me.get("deck") or []),
            "decisions": 0,
        })
        entry["decisions"] += 1
        # The board snapshot for a turn is its *first* MAIN decision, so
        # "bodies at turn 4" means what we started the turn with.
        for option in options:
            try:
                kind = mf.action_type(current, option, select)
                card = mf.candidate_card(current, option, select) or {}
            except Exception:  # noqa: BLE001
                continue
            entry["offered_actions"].add(kind)
            card_id = int(card.get("id", -1))
            if card_id >= 0:
                entry["offered"].add(card_id)
        if isinstance(action, list) and len(action) == 1 and \
                isinstance(action[0], int) and 0 <= action[0] < len(options):
            option = options[action[0]]
            try:
                kind = mf.action_type(current, option, select)
                card = mf.candidate_card(current, option, select) or {}
            except Exception:  # noqa: BLE001
                continue
            entry["played_actions"].add(kind)
            card_id = int(card.get("id", -1))
            if card_id >= 0:
                entry["played"].add(card_id)
    return [turns[t] for t in sorted(turns)]


def summarise(games: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Offer / take rates per own turn, plus the board size trajectory."""
    offered: Counter = Counter()
    taken: Counter = Counter()
    offered_kind: Counter = Counter()
    taken_kind: Counter = Counter()
    board: dict[int, list[int]] = defaultdict(list)
    own_turns = 0
    for turns in games:
        for entry in turns:
            own_turns += 1
            board[entry["turn"]].append(entry["bodies"])
            for card_id in entry["offered"]:
                offered[card_id] += 1
            for card_id in entry["played"]:
                taken[card_id] += 1
            for kind in entry["offered_actions"]:
                offered_kind[kind] += 1
            for kind in entry["played_actions"]:
                taken_kind[kind] += 1
    return {
        "games": len(games),
        "own_turns": own_turns,
        "turns_per_game": (
            round(own_turns / len(games), 2) if games else None
        ),
        "bodies_by_turn": {
            str(turn): {
                "n": len(values),
                "mean": round(sum(values) / len(values), 2),
            }
            for turn, values in sorted(board.items())
        },
        "cards": {
            str(card_id): {
                "name": CARDS.get(card_id, "?"),
                "offered": offered[card_id],
                "taken": taken[card_id],
                "offer_rate": round(offered[card_id] / own_turns, 4),
                "take_rate": (
                    round(taken[card_id] / offered[card_id], 4)
                    if offered[card_id] else None
                ),
                "play_rate": round(taken[card_id] / own_turns, 4),
            }
            for card_id in sorted(offered, key=lambda c: -offered[c])
        },
        "actions": {
            kind: {
                "offered": offered_kind[kind],
                "taken": taken_kind[kind],
                "offer_rate": round(offered_kind[kind] / own_turns, 4),
                "take_rate": (
                    round(taken_kind[kind] / offered_kind[kind], 4)
                    if offered_kind[kind] else None
                ),
                "play_rate": round(taken_kind[kind] / own_turns, 4),
            }
            for kind in sorted(offered_kind, key=lambda k: -offered_kind[k])
        },
    }


def load_field(
    data_root: Path, max_turn: int
) -> dict[str, list[list[dict[str, Any]]]]:
    cohorts: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for raw in csv.DictReader(
        (data_root / "indexes" / "episodes.csv").open(encoding="utf-8-sig")
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
        path = data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        meta = replay_meta(replay, seat)
        if meta is None:
            continue
        cohort = cohort_of(meta)
        if cohort is None:
            continue
        turns = per_turn(replay, seat, max_turn)
        if not turns:
            continue
        cohorts[cohort].append(turns)
        if cohort == "alakazam_second":
            cohorts["alakazam_second_won" if meta["won"]
                    else "alakazam_second_lost"].append(turns)
    return cohorts


def load_ours(
    runs: list[tuple[Path, str]], max_turn: int
) -> dict[str, list[list[dict[str, Any]]]]:
    cohorts: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for run_dir, submission in runs:
        for raw in csv.DictReader(
            (run_dir / "episodes.csv").open(encoding="utf-8-sig")
        ):
            a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
            if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
                continue
            episode_id = int(raw["episode_id"])
            path = (
                run_dir / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not path.exists():
                continue
            seat = 0 if a0 == submission else 1
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            meta = replay_meta(replay, seat)
            if meta is None:
                continue
            cohort = cohort_of(meta)
            if cohort is None:
                continue
            turns = per_turn(replay, seat, max_turn)
            if turns:
                cohorts[f"ours_{cohort}"].append(turns)
    return cohorts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--run", action="append", default=[],
        help="dir=submission_id, repeatable. Defaults to the v8 and v7 runs.",
    )
    parser.add_argument("--max-turn", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not args.run:
        base = ROOT / "data" / "submissions"
        args.run = [
            f"{base / 'submission_55317804'}=55317804",
            f"{base / 'submission_55302846'}=55302846",
        ]
    runs = [
        (Path(spec.rsplit("=", 1)[0]), spec.rsplit("=", 1)[1])
        for spec in args.run
    ]

    cohorts = load_field(args.data_root, args.max_turn)
    cohorts.update(load_ours(runs, args.max_turn))
    report = {
        "max_turn": args.max_turn,
        "runs": [str(r) for r, _ in runs],
        "cohorts": {
            name: summarise(games) for name, games in sorted(cohorts.items())
        },
    }

    # The comparison the whole thing exists for, with a test attached: our
    # play rate per own turn against the field's, same matchup and same seat.
    us = report["cohorts"].get("ours_alakazam_second")
    them = report["cohorts"].get("alakazam_second")
    gaps = []
    if us and them:
        for key in sorted(set(us["cards"]) | set(them["cards"])):
            ours = us["cards"].get(key)
            field = them["cards"].get(key)
            if field is None or field["offered"] < 20:
                continue
            o_taken = ours["taken"] if ours else 0
            o_turns = us["own_turns"]
            gaps.append({
                "card": key,
                "name": field["name"],
                "field_offer_rate": field["offer_rate"],
                "ours_offer_rate": ours["offer_rate"] if ours else 0.0,
                "field_play_rate": field["play_rate"],
                "ours_play_rate": ours["play_rate"] if ours else 0.0,
                "delta_play_rate": round(
                    (ours["play_rate"] if ours else 0.0) - field["play_rate"], 4
                ),
                "offer_fisher": fisher(
                    ours["offered"] if ours else 0,
                    o_turns - (ours["offered"] if ours else 0),
                    field["offered"], them["own_turns"] - field["offered"],
                ),
                "play_fisher": fisher(
                    o_taken, o_turns - o_taken,
                    field["taken"], them["own_turns"] - field["taken"],
                ),
            })
        gaps.sort(key=lambda g: g["delta_play_rate"])
    report["ours_vs_field_alakazam_second"] = gaps

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for name, block in report["cohorts"].items():
        print(f"{name:28s} games={block['games']:4d} "
              f"own_turns={block['own_turns']:5d} "
              f"turns/game={block['turns_per_game']}")
    print()
    print(f"{'card':34s} {'offer f/o':>17s} {'play f/o':>17s} "
          f"{'d':>7s} {'pOffer':>7s} {'pPlay':>7s}")
    for gap in gaps[:25]:
        print(f"{gap['name'][:33]:34s} "
              f"{gap['field_offer_rate']:8.3f}/{gap['ours_offer_rate']:<8.3f} "
              f"{gap['field_play_rate']:8.3f}/{gap['ours_play_rate']:<8.3f} "
              f"{gap['delta_play_rate']:+7.3f} "
              f"{gap['offer_fisher']:7.4f} {gap['play_fisher']:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
