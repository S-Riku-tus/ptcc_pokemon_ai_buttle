"""Was a better prize route on the board when v15 pulled the trigger?

``prizes per Shadow Bullet`` separates v15's wins from its losses (1.06 against
0.50 over 110 rated games) but that is an outcome, not a decision: a losing
board has fewer knock-outable bodies on it.  This asks the decision-level
question instead, with the agent's own feature code:

At every own MAIN decision where v15 attacked, ``ml_features.turn_routes``
already computes the prizes each Boss's Orders target would yield from the same
single Shadow Bullet.  So for every attack we can ask whether a Boss in hand,
with the Supporter for the turn still unspent, would have turned that swing
into more prizes than the one actually taken - and how many.

The same walk records the two states the v16 plan needs sized:

* **stall**: our turn attacked, took no prize, and left their board unchanged.
* **no-attacker**: our turn ended with no Grimmsnarl ex in play at all, which
  is the half of the first-Shadow tail ``attack_access`` cannot reach.

    python scripts/analyze_grimmsnarl_v16_boss_routes.py \
        --run data/runs/grimmsnarl/20260810_grimmsnarl_ml_v15_sub55404196 \
        --submission 55404196
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
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
OPTION_PLAY = 7
OPTION_ATTACK = 13
OPTION_END = 14
BOSS_ID = mf.BOSS_ID
GRIMMSNARL_EX_ID = mf.GRIMMSNARL_EX_ID
SHADOW_BULLET_ID = mf.SHADOW_BULLET_ID


def walk(replay: dict[str, Any], seat: int) -> dict[str, Any]:
    steps = replay.get("steps") or []
    first_player = -1
    for step in reversed(steps):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(current, dict) and int(
            current.get("firstPlayer", -1)
        ) >= 0:
            first_player = int(current.get("firstPlayer", -1))
            break

    attacks: list[dict[str, Any]] = []
    ends: list[dict[str, Any]] = []
    boss_plays = 0
    seen_turns: set[int] = set()

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(select, dict) or not isinstance(current, dict):
            continue
        if int(select.get("context", -1)) != MAIN_CONTEXT:
            continue
        players = current.get("players") or []
        if len(players) < 2:
            continue
        action = (steps[index + 1][seat] or {}).get("action")
        if not (isinstance(action, list) and len(action) == 1
                and isinstance(action[0], int)):
            continue
        options = select.get("option") or []
        chosen = int(action[0])
        if not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        option_type = mf._int(option.get("type"))
        me, opponent = players[seat], players[1 - seat]
        turn = own_turn(current, seat, first_player)
        seen_turns.add(turn)

        if option_type == OPTION_PLAY:
            card = mf.candidate_card(current, option, select)
            if int((card or {}).get("id", -1)) == BOSS_ID:
                boss_plays += 1

        if option_type not in (OPTION_ATTACK, OPTION_END):
            continue

        routes = mf.turn_routes(current, opponent)
        hand = mf._cards(me, "hand")
        boss_in_hand = any(
            int(c.get("id", -1)) == BOSS_ID for c in hand
        )
        supporter_free = not bool(current.get("supporterPlayed"))
        boss_playable = boss_in_hand and supporter_free
        in_play = mf._cards(me, "active") + mf._cards(me, "bench")
        bodies, ready = mf._attacker_state(in_play)
        row = {
            "turn": turn,
            "closing": "attack" if option_type == OPTION_ATTACK else "end",
            "attack_id": (
                mf._int(option.get("attackId"))
                if option_type == OPTION_ATTACK else -1
            ),
            "no_boss_prizes": routes["no_boss_prizes"],
            "best_boss_prizes": routes["best_boss_prizes"],
            "boss_gain": routes["boss_gain"],
            "boss_playable": boss_playable,
            "boss_in_hand": boss_in_hand,
            "active_walled": routes["active_walled"],
            "grimmsnarl_bodies": bodies,
            "grimmsnarl_ready": ready,
            "deck_count": int(me.get("deckCount", 0) or 0),
            "our_prizes": len(me.get("prize") or []),
            "their_prizes": len(opponent.get("prize") or []),
        }
        if option_type == OPTION_ATTACK:
            attacks.append(row)
        else:
            ends.append(row)

    rewards = replay.get("rewards") or [None, None]
    won = None
    if rewards[seat] is not None:
        other = rewards[1 - seat]
        won = bool(rewards[seat] > (other if other is not None else 0))
    decks: list[list[int] | None] = [None, None]
    for side in (0, 1):
        raw = (steps[1][side] or {}).get("action") if len(steps) > 1 else None
        if isinstance(raw, list) and len(raw) == 60:
            decks[side] = [int(v) for v in raw]
    label = deck_label(decks[1 - seat])
    return {
        "won": won,
        "matchup": matchup_of(label),
        "opponent_deck": label,
        "attacks": attacks,
        "ends": ends,
        "boss_plays": boss_plays,
        "own_turns": len(seen_turns),
    }


def load(run_dir: Path, submission: str) -> list[dict[str, Any]]:
    rows = []
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
        row = walk(json.loads(path.read_text(encoding="utf-8")), seat)
        row["episode_id"] = episode_id
        rows.append(row)
    return rows


def summarise(games: list[dict[str, Any]]) -> dict[str, Any]:
    attacks = [a for g in games for a in g["attacks"]]
    shadow = [a for a in attacks if a["attack_id"] == SHADOW_BULLET_ID]
    missed = [
        a for a in shadow
        if a["boss_playable"] and a["boss_gain"] > 0
    ]
    zero = [a for a in shadow if a["no_boss_prizes"] == 0]
    zero_rescuable = [
        a for a in zero if a["boss_playable"] and a["best_boss_prizes"] > 0
    ]
    ends = [e for g in games for e in g["ends"]]
    end_no_body = [e for e in ends if e["grimmsnarl_bodies"] == 0]
    end_ready = [e for e in ends if e["grimmsnarl_ready"] > 0]
    return {
        "games": len(games),
        "wins": sum(1 for g in games if g["won"]),
        "shadow_bullets": len(shadow),
        "shadow_taking_no_prize": len(zero),
        "shadow_taking_no_prize_share": round(
            len(zero) / max(1, len(shadow)), 3
        ),
        "boss_would_have_added_prizes": len(missed),
        "boss_missed_share_of_shadow": round(
            len(missed) / max(1, len(shadow)), 3
        ),
        "prizes_forgone_to_no_boss": sum(a["boss_gain"] for a in missed),
        "zero_prize_swings_a_boss_would_rescue": len(zero_rescuable),
        "boss_plays": sum(g["boss_plays"] for g in games),
        "end_turns": len(ends),
        "end_with_no_grimmsnarl_body": len(end_no_body),
        "end_with_ready_attacker": len(end_ready),
        "mean_deck_at_attack": round(
            statistics.fmean([a["deck_count"] for a in attacks]), 2
        ) if attacks else None,
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

    by_matchup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_matchup[game["matchup"]].append(game)

    out = {
        "overall": summarise(games),
        "wins": summarise([g for g in games if g["won"]]),
        "losses": summarise([g for g in games if g["won"] is False]),
        "by_matchup": {
            key: {
                "all": summarise(value),
                "losses": summarise([g for g in value if g["won"] is False]),
            }
            for key, value in sorted(
                by_matchup.items(), key=lambda kv: -len(kv[1])
            )
        },
        "boss_gain_histogram": dict(Counter(
            a["boss_gain"] for g in games for a in g["attacks"]
            if a["attack_id"] == SHADOW_BULLET_ID and a["boss_playable"]
        )),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
