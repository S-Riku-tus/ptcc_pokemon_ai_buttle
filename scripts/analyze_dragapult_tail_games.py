"""Why did this game produce zero or one Phantom Dive?

Win rate is a function of Phantom Dive count alone, and on the 26-game v2 run
the whole remaining deficit against the teachers is the low tail: 30.8% of our
games yield <=1 dive against the teachers' 14.7%.  Everything at 2+ is at
teacher parity, so the only question worth asking is what happens in the tail.

A tail game has exactly one of a small number of causes and they need very
different fixes:

* dead draw - we never had a second Basic to bench, or never drew the line.
  Nothing an agent can do; it is the deck's own mulligan rate.
* line starved - the pieces were in the deck but search was not played, or was
  played on something else.  Addressable by the policy.
* attacker but no energy - Dragapult ex on board and never powered.
* attacker powered but never attacked - access denied (wall, gust, sleep) or an
  END was taken with the attack legal.
* run over early - we were knocked out of the game before the line could mature.

This walks each game, classifies it, and prints the teachers' tail for the same
classification so the two can be compared shape to shape.

Usage:
  python scripts/analyze_dragapult_tail_games.py \
      --run data/submissions/submission_55550682_dragapult_v2 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --report experiments/dragapult_ml_v2/tail_games_v2.json
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}

ATTACKS = {
    int(attack["attackId"]): attack
    for attack in json.loads(
        (ROOT / "vendor" / "cg" / "attacks.json").read_text(encoding="utf-8")
    )
}

DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
FIRE, PSYCHIC = 2, 5
PHANTOM_DIVE = 154
OPT_ATTACK, OPT_END = 13, 14
# Cards that find a Basic or bring the line back; a tail game where these sat
# in hand unplayed is a policy defect, one where they never appeared is variance.
SEARCH_CARDS = {
    1121: "Ultra Ball", 1086: "Buddy-Buddy Poffin", 1097: "Night Stretcher",
    1152: "Poke Pad", 1227: "Lillie's Determination",
}


def name(card_id: Any) -> str:
    try:
        return str(CARDS.get(int(card_id), {}).get("name") or card_id)
    except (TypeError, ValueError):
        return str(card_id)


def bodies(player: dict[str, Any]) -> list[dict[str, Any]]:
    active = player.get("active") or []
    if isinstance(active, dict):
        active = [active]
    return [card for card in list(active) + list(player.get("bench") or [])
            if isinstance(card, dict)]


def line_counts(player: dict[str, Any]) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for card in bodies(player):
        card_id = card.get("id")
        if card_id is None:
            continue
        counter[int(card_id)] += 1
        # The stage-1 and basic under an evolved Pokemon are `preEvolution`.
        for under in card.get("preEvolution") or []:
            if isinstance(under, dict) and under.get("id") is not None:
                counter[int(under["id"])] += 1
    return counter


def walk(replay: dict[str, Any], seat: int) -> dict[str, Any]:
    steps = replay.get("steps") or []
    dives = 0
    own_turns = 0
    seen_turns: set[int] = set()
    max_pult = 0
    max_line = 0
    first_pult_turn: int | None = None
    ends_with_attack_legal = 0
    attacks = 0
    search_offered = 0
    search_taken = 0
    opening_basics: int | None = None
    mulligans = 0
    ever_had_dreepy_in_hand = 0
    final_prizes = (6, 6)
    started = False
    opponent_cards: Counter[str] = Counter()
    attacks_used: Counter[str] = Counter()
    pult_active_turns: set[int] = set()
    dive_ready_turns: set[int] = set()

    for index, pair in enumerate(steps):
        payload = pair[seat]
        observation = payload.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) != 2:
            continue
        your = int(current.get("yourIndex", seat))
        mine = players[your]
        theirs = players[1 - your]

        for card in bodies(theirs):
            if card.get("id") is not None:
                opponent_cards[name(card["id"])] += 1

        # `prize` is the list of remaining prize cards, and it is empty in the
        # pre-deal observation, so only a full six-card deal starts the count.
        prizes = (mine.get("prize"), theirs.get("prize"))
        if all(isinstance(value, list) for value in prizes):
            if started or (len(prizes[0]) == 6 and len(prizes[1]) == 6):
                started = True
                final_prizes = (len(prizes[0]), len(prizes[1]))

        counts = line_counts(mine)
        max_pult = max(max_pult, counts.get(DRAGAPULT, 0))
        max_line = max(max_line, counts.get(DREEPY, 0) + counts.get(DRAKLOAK, 0)
                       + counts.get(DRAGAPULT, 0))

        if payload.get("status") != "ACTIVE":
            continue
        select = observation.get("select")
        turn = current.get("turn")
        if isinstance(turn, int) and turn not in seen_turns:
            seen_turns.add(turn)
            own_turns += 1
        # Evolving happens mid-turn, so this has to be checked on every
        # observation and not only on the first one of a turn.
        if counts.get(DRAGAPULT, 0) and first_pult_turn is None:
            first_pult_turn = own_turns

        # Phantom Dive costs one Fire and one Psychic; a Dragapult ex on the
        # board and unpowered is a different defect from one that is powered
        # and walled, so track how far along the Active's cost is.
        active = mine.get("active") or []
        if isinstance(active, dict):
            active = [active]
        for card in active:
            if isinstance(card, dict) and int(card.get("id", -1)) == DRAGAPULT:
                colors = [int(value) for value in (card.get("energies") or [])]
                pult_active_turns.add(own_turns)
                if FIRE in colors and PSYCHIC in colors:
                    dive_ready_turns.add(own_turns)

        if select is None:
            continue
        options = select.get("option") or []
        action = (steps[index + 1][seat].get("action")
                  if index + 1 < len(steps) else None)
        if not isinstance(action, list) or len(action) != 1:
            continue
        picked = int(action[0])
        if not 0 <= picked < len(options):
            continue
        chosen = options[picked]
        chosen_type = int(chosen.get("type", -1))

        hand = mine.get("hand") or []
        if any(isinstance(card, dict) and card.get("id") == DREEPY
               for card in hand):
            ever_had_dreepy_in_hand = 1

        if int(select.get("context", -1)) == 0:
            search_here = {
                position for position, option in enumerate(options)
                if int(option.get("type", -1)) == 7
                and 0 <= int(option.get("index", -1)) < len(hand)
                and isinstance(hand[int(option["index"])], dict)
                and int(hand[int(option["index"])].get("id", -1)) in SEARCH_CARDS
            }
            if search_here:
                search_offered += 1
                if picked in search_here:
                    search_taken += 1
            attack_here = any(int(option.get("type", -1)) == OPT_ATTACK
                              for option in options)
            if chosen_type == OPT_END and attack_here:
                ends_with_attack_legal += 1

        if chosen_type == OPT_ATTACK:
            attacks += 1
            attack_id = int(chosen.get("attackId", -1))
            attacks_used[ATTACKS.get(attack_id, {}).get("name", attack_id)] += 1
            if attack_id == PHANTOM_DIVE:
                dives += 1

        if opening_basics is None and own_turns == 1:
            opening_basics = sum(
                1 for card in hand
                if isinstance(card, dict)
                and str(CARDS.get(int(card.get("id", -1)), {}).get("kind", ""))
                .upper().startswith("BASIC")
            )

    rewards = replay.get("rewards") or [0, 0]
    result = "win" if rewards[seat] > rewards[1 - seat] else (
        "loss" if rewards[seat] < rewards[1 - seat] else "draw")

    return {
        "dives": dives, "attacks": attacks, "own_turns": own_turns,
        "max_dragapult": max_pult, "max_line_bodies": max_line,
        "first_dragapult_own_turn": first_pult_turn,
        "search_offered": search_offered, "search_taken": search_taken,
        "ends_with_attack_legal": ends_with_attack_legal,
        "our_prizes": 6 - final_prizes[0],
        "opp_prizes": 6 - final_prizes[1],
        "result": result,
        "ever_had_dreepy_in_hand": ever_had_dreepy_in_hand,
        "opponent_top": [card for card, _ in opponent_cards.most_common(4)],
        "attacks_used": dict(attacks_used.most_common()),
        "pult_active_turns": len(pult_active_turns),
        "dive_ready_turns": len(dive_ready_turns),
    }


def classify(game: dict[str, Any]) -> str:
    """One cause per tail game, in the order that makes a fix actionable."""
    if game["dives"] >= 2:
        return "not a tail game"
    if game["own_turns"] <= 3:
        return "run over before turn 4"
    if game["max_line_bodies"] == 0:
        return "line never reached the board"
    if game["max_dragapult"] == 0:
        return "never evolved to Dragapult ex"
    if game["attacks"] == 0:
        return "Dragapult ex on board, never attacked"
    return "attacked but not with Phantom Dive"


def cohort(games: list[dict[str, Any]], label: str) -> dict[str, Any]:
    tail = [game for game in games if game["dives"] <= 1]
    causes = Counter(classify(game) for game in tail)
    print(f"\n=== {label}: {len(tail)}/{len(games)} tail games "
          f"({len(tail) / max(1, len(games)):.3f})")
    for cause, count in causes.most_common():
        print(f"  {count:>5} ({count / max(1, len(tail)):.3f})  {cause}")

    def mean(key: str, block: list[dict[str, Any]]) -> float:
        values = [game[key] for game in block if isinstance(game[key], (int, float))]
        return round(sum(values) / len(values), 3) if values else float("nan")

    body = [game for game in games if game["dives"] >= 2]
    print(f"  {'':22} {'tail':>8} {'rest':>8}")
    for key in ("own_turns", "max_dragapult", "max_line_bodies",
                "first_dragapult_own_turn", "pult_active_turns",
                "dive_ready_turns", "search_offered", "search_taken",
                "ends_with_attack_legal", "attacks", "our_prizes",
                "opp_prizes"):
        print(f"  {key:22} {mean(key, tail):>8} {mean(key, body):>8}")
    take = (sum(game["search_taken"] for game in tail)
            / max(1, sum(game["search_offered"] for game in tail)))
    take_rest = (sum(game["search_taken"] for game in body)
                 / max(1, sum(game["search_offered"] for game in body)))
    print(f"  {'search take rate':22} {take:>8.3f} {take_rest:>8.3f}")
    return {
        "label": label, "games": len(games), "tail": len(tail),
        "tail_share": round(len(tail) / max(1, len(games)), 4),
        "causes": dict(causes),
        "tail_search_take": round(take, 4),
        "rest_search_take": round(take_rest, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--teacher-limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    summaries = []
    for run in args.run:
        manifest = list(csv.DictReader(
            (run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()))
        games = []
        for row in manifest:
            seat = row.get("detected_submission_agent_index", "")
            if seat not in ("0", "1"):
                continue
            path = (run / "episodes" / str(row["episode_id"]) / "replay"
                    / f"episode_{row['episode_id']}.json")
            if not path.exists():
                continue
            game = walk(json.loads(path.read_text(encoding="utf-8")), int(seat))
            game["episode_id"] = int(row["episode_id"])
            games.append(game)

        print(f"\n########## {run.name}: {len(games)} games")
        print(f"{'episode':>10} {'r':>3} {'PD':>3} {'atk':>4} {'turns':>6} "
              f"{'pult':>5} {'line':>5} {'t1pult':>7} {'srch':>7} {'endatk':>7} "
              f"{'prizes':>7}  cause / opponent")
        for game in sorted(games, key=lambda item: (item["dives"], -item["own_turns"])):
            first = game["first_dragapult_own_turn"]
            note = classify(game) if game["dives"] <= 1 else ""
            print(f"{game['episode_id']:>10} {game['result'][0]:>3} "
                  f"{game['dives']:>3} {game['attacks']:>4} "
                  f"{game['own_turns']:>6} {game['max_dragapult']:>5} "
                  f"{game['max_line_bodies']:>5} "
                  f"{(first if first is not None else '-'):>7} "
                  f"{game['search_taken']}/{game['search_offered']:>4} "
                  f"{game['ends_with_attack_legal']:>7} "
                  f"{game['our_prizes']}-{game['opp_prizes']:<5} "
                  f"{game['pult_active_turns']:>3}/{game['dive_ready_turns']:<3} "
                  f"{note}  | {', '.join(game['opponent_top'][:3])}"
                  f"  || {', '.join(f'{k} x{v}' for k, v in list(game['attacks_used'].items())[:3])}")
        summaries.append(cohort(games, run.name))

    if args.teacher_index:
        rows = list(csv.DictReader(
            args.teacher_index.read_text(encoding="utf-8-sig").splitlines()))
        seen: set[tuple[str, int]] = set()
        games = []
        for row in rows:
            key = (str(row["episode_id"]), int(row["seat_index"]))
            if key in seen:
                continue
            seen.add(key)
            path = Path(row["replay_path"])
            if not path.is_absolute():
                path = args.teacher_index.parent.parent / path
            if not path.exists():
                continue
            games.append(walk(json.loads(path.read_text(encoding="utf-8")),
                              int(row["seat_index"])))
            if args.teacher_limit and len(games) >= args.teacher_limit:
                break
        summaries.append(cohort(games, "teachers"))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
