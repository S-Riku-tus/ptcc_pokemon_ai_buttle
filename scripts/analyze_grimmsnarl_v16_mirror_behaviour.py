"""In the mirror, the opponent is a teacher on our own 60 cards.

36 of v15's 110 rated games are Grimmsnarl mirrors, and a mirror is the one
matchup where a behavioural difference is not confounded by the deck: both
seats draw the same list, so any per-game count that separates the two seats is
a policy difference and nothing else.  21-15 with a mean rating delta of -0.56
says the difference is not in our favour.

Every count here is taken for both seats of the same games and then split by
who won, so "what the winner of a mirror does" is measured on the same board
distribution as "what we do".

    python scripts/analyze_grimmsnarl_v16_mirror_behaviour.py \
        --run data/runs/grimmsnarl/20260810_grimmsnarl_ml_v15_sub55404196 \
        --submission 55404196
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v15"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_v16_prize_conversion import (  # noqa: E402
    deck_label,
    matchup_of,
    own_turn,
)

MAIN_CONTEXT = 0
CTX_REMOVE_DAMAGE_COUNTER = 16
CTX_DAMAGE_COUNTER = 13
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14

COUNTED_CARDS = {
    mf.BOSS_ID: "boss",
    mf.UNFAIR_STAMP_ID: "unfair_stamp",
    mf.PETREL_ID: "petrel",
    mf.POFFIN_ID: "poffin",
    mf.RARE_CANDY_ID: "rare_candy",
    mf.NIGHT_STRETCHER_ID: "night_stretcher",
    mf.POKE_PAD_ID: "poke_pad",
    mf.LILLIE_ID: "lillie",
    mf.DAWN_ID: "dawn",
}
EVOLVE_TARGETS = {
    mf.GRIMMSNARL_EX_ID: "evolve_grimmsnarl",
    mf.MORGREM_ID: "evolve_morgrem",
    mf.FROSLASS_ID: "evolve_froslass",
}


def side_counts(
    replay: dict[str, Any], seat: int, first_player: int
) -> dict[str, Any]:
    steps = replay.get("steps") or []
    counts: dict[str, float] = {name: 0 for name in COUNTED_CARDS.values()}
    counts.update({name: 0 for name in EVOLVE_TARGETS.values()})
    counts.update({
        "adrena_brain": 0,
        "shadow_bullet": 0,
        "other_attack": 0,
        "retreat": 0,
        "manual_attach": 0,
        "main_decisions": 0,
        "own_turns": 0,
        "first_grim_turn": None,
        "first_shadow_turn": None,
        "prizes_taken": 0,
        "final_deck": None,
    })
    seen_turns: set[int] = set()
    final_prizes = 6

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(current, dict):
            continue
        players = current.get("players") or []
        if len(players) < 2:
            continue
        me = players[seat]
        turn = own_turn(current, seat, first_player)
        counts["final_deck"] = int(me.get("deckCount", 0) or 0)
        final_prizes = len(me.get("prize") or [])
        for card in mf._cards(me, "active") + mf._cards(me, "bench"):
            if (
                int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
                and counts["first_grim_turn"] is None
            ):
                counts["first_grim_turn"] = turn

        if not isinstance(select, dict):
            continue
        action = (steps[index + 1][seat] or {}).get("action")
        if not (isinstance(action, list) and len(action) == 1
                and isinstance(action[0], int)):
            continue
        context = int(select.get("context", -1))
        if context == CTX_REMOVE_DAMAGE_COUNTER:
            counts["adrena_brain"] += 1
            continue
        if context != MAIN_CONTEXT:
            continue
        seen_turns.add(turn)
        options = select.get("option") or []
        chosen = int(action[0])
        if not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        option_type = mf._int(option.get("type"))
        counts["main_decisions"] += 1
        if option_type == OPTION_ATTACK:
            if mf._int(option.get("attackId")) == mf.SHADOW_BULLET_ID:
                counts["shadow_bullet"] += 1
                if counts["first_shadow_turn"] is None:
                    counts["first_shadow_turn"] = turn
            else:
                counts["other_attack"] += 1
        elif option_type == OPTION_RETREAT:
            counts["retreat"] += 1
        elif option_type == OPTION_ATTACH:
            counts["manual_attach"] += 1
        elif option_type in (OPTION_PLAY, OPTION_EVOLVE):
            card = mf.candidate_card(current, option, select)
            card_id = int((card or {}).get("id", -1))
            if option_type == OPTION_EVOLVE and card_id in EVOLVE_TARGETS:
                counts[EVOLVE_TARGETS[card_id]] += 1
            elif card_id in COUNTED_CARDS:
                counts[COUNTED_CARDS[card_id]] += 1

    counts["own_turns"] = len(seen_turns)
    counts["prizes_taken"] = 6 - final_prizes
    return counts


def load(run_dir: Path, submission: str) -> list[dict[str, Any]]:
    games = []
    for raw in csv.DictReader(
        (run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if raw["state"] != "COMPLETED":
            continue
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC":
            continue
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if a0 == a1:
            continue
        seat = 0 if a0 == submission else 1
        episode_id = int(raw["episode_id"])
        path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        if len(steps) < 3:
            continue
        decks: list[list[int] | None] = [None, None]
        for side in (0, 1):
            action = (steps[1][side] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[side] = [int(v) for v in action]
        if matchup_of(deck_label(decks[1 - seat])) != "mirror":
            continue
        first_player = -1
        for step in reversed(steps):
            if seat >= len(step):
                continue
            current = (
                (step[seat] or {}).get("observation") or {}
            ).get("current")
            if isinstance(current, dict) and int(
                current.get("firstPlayer", -1)
            ) >= 0:
                first_player = int(current.get("firstPlayer", -1))
                break
        rewards = replay.get("rewards") or [None, None]
        won = None
        if rewards[seat] is not None:
            other = rewards[1 - seat]
            won = bool(rewards[seat] > (other if other is not None else 0))
        games.append({
            "episode_id": episode_id,
            "won": won,
            "us": side_counts(replay, seat, first_player),
            "them": side_counts(replay, 1 - seat, first_player),
            "we_went_first": (
                first_player == seat if first_player >= 0 else None
            ),
        })
    return games


def mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.fmean(clean), 3) if clean else None


def profile(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    keys = sorted(rows[0].keys()) if rows else []
    return {"name": name, "games": len(rows)} | {
        key: mean([row[key] for row in rows]) for key in keys
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--submission", action="append", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    games: list[dict[str, Any]] = []
    for run_dir, submission in zip(args.run, args.submission):
        games.extend(load(run_dir, submission))

    us = [g["us"] for g in games]
    them = [g["them"] for g in games]
    winners = [g["us"] if g["won"] else g["them"] for g in games]
    losers = [g["them"] if g["won"] else g["us"] for g in games]

    out = {
        "games": len(games),
        "record": (
            f"{sum(1 for g in games if g['won'])}-"
            f"{sum(1 for g in games if g['won'] is False)}"
        ),
        "us": profile(us, "us"),
        "them": profile(them, "them"),
        "winner": profile(winners, "winner"),
        "loser": profile(losers, "loser"),
        "us_in_wins": profile([g["us"] for g in games if g["won"]], "us_win"),
        "us_in_losses": profile(
            [g["us"] for g in games if g["won"] is False], "us_loss"
        ),
        "them_in_our_losses": profile(
            [g["them"] for g in games if g["won"] is False], "them_beat_us"
        ),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
