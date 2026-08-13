"""Opening-turn detail: setup board, hand contents and card-level offers.

The first pass (``extract.py``) showed the v21 / field curves separate at own
turn 2.  Own turn 2 is the only turn where Marnie's Grimmsnarl ex can reach the
board at all (Rare Candy from Impidimp; the Morgrem route needs own turn 3), so
this pass records, for own turns 0-4:

* the setup board - which basic was made Active and what was benched;
* the hand multiset at the FIRST MAIN decision of the turn, which separates
  "never drew it" from "drew it and did not play it";
* which key cards were *offered* anywhere in the turn's MAIN prompts, and which
  were taken.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

import extract as base  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
MAX_OWN_TURN = 4
KEY_IDS = list(mf.KEY_CARD_IDS)


def walk(path: Path, seat: int) -> dict[str, Any] | None:
    try:
        replay = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    steps = replay.get("steps") or []
    decks = base._decks(steps)
    if decks[seat] is None or deck_hash(decks[seat]) != base.OUR_DECK_HASH:
        return None

    went_first: bool | None = None
    for step in reversed(steps):
        for actor in (0, 1):
            if actor >= len(step):
                continue
            current = ((step[actor] or {}).get("observation") or {}).get("current")
            if isinstance(current, dict):
                first = int(current.get("firstPlayer", -1))
                if first >= 0:
                    went_first = first == seat
                    break
        if went_first is not None:
            break
    if went_first is None:
        return None

    rewards = replay.get("rewards") or [None, None]
    ours, theirs = rewards[seat], rewards[1 - seat]
    if ours is None:
        return None

    rows: dict[int, dict[str, Any]] = {}
    setup: dict[str, Any] = {}
    seen_turn0_board = False

    for index, step in enumerate(steps[:-1]):
        for actor in (0, 1):
            if actor >= len(step) or actor >= len(steps[index + 1]):
                continue
            record = step[actor] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            current = observation.get("current") or {}
            players = current.get("players") or []
            if len(players) < 2:
                continue
            turn = int(current.get("turn", -1))
            if turn < 0:
                continue
            mine = players[seat]

            if turn >= 1 and not seen_turn0_board:
                # First observation of turn 1: the board as it stood after the
                # setup phase, before anybody has drawn for their first turn.
                active = (mf._cards(mine, "active") or [{}])[0]
                bench = mf._cards(mine, "bench")
                setup = {
                    "setup_active_id": int(active.get("id", -1)),
                    "setup_bench_size": len(bench),
                    "setup_bench_impidimp": sum(
                        int(int(c.get("id", -1)) == mf.IMPIDIMP_ID) for c in bench
                    ),
                    "setup_bench_snorunt": sum(
                        int(int(c.get("id", -1)) == mf.SNORUNT_ID) for c in bench
                    ),
                    "setup_bench_munkidori": sum(
                        int(int(c.get("id", -1)) == mf.MUNKIDORI_ID) for c in bench
                    ),
                    "setup_impidimp_total": sum(
                        int(int(c.get("id", -1)) == mf.IMPIDIMP_ID)
                        for c in [active] + bench
                    ),
                }
                seen_turn0_board = True

            if actor != seat or turn < 1:
                continue
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            if not options or int(select.get("context", -1)) != mf.MAIN_CONTEXT:
                continue
            own = base._own_turn(turn, went_first)
            if own < 1 or own > MAX_OWN_TURN:
                continue

            row = rows.get(own)
            if row is None:
                hand = Counter(
                    int(card.get("id", -1)) for card in mf._cards(mine, "hand")
                )
                in_play = Counter(
                    int(card.get("id", -1)) for card in mf._in_play(mine)
                )
                row = {
                    "own_turn": own,
                    "offered": set(),
                    "taken": Counter(),
                    "hand_size": len(mf._cards(mine, "hand")),
                    "deck_left": mine.get("deckCount"),
                }
                for card_id in KEY_IDS:
                    row[f"hand_{card_id}"] = hand.get(card_id, 0)
                    row[f"play_{card_id}"] = in_play.get(card_id, 0)
                # Rare Candy needs a benched/active Impidimp *that did not just
                # arrive*; the engine decides, so record both hand and board.
                rows[own] = row

            action = (steps[index + 1][actor] or {}).get("action")
            picked = [
                int(v) for v in action
                if isinstance(v, int) and 0 <= int(v) < len(options)
            ] if isinstance(action, list) else []
            for position, option in enumerate(options):
                try:
                    kind = mf.action_type(current, option, select)
                except Exception:  # noqa: BLE001
                    continue
                card = mf.candidate_card(current, option, select) or {}
                card_id = int(card.get("id", -1))
                if card_id in set(KEY_IDS):
                    row["offered"].add(card_id)
                    if position in picked:
                        row["taken"][card_id] += 1
                if kind == "evolve":
                    row["offered"].add(("evolve", card_id))
                    if position in picked:
                        row["taken"][("evolve", card_id)] += 1

    out_rows = []
    for own, row in sorted(rows.items()):
        item = {
            "own_turn": own,
            "hand_size": row["hand_size"],
            "deck_left": row["deck_left"],
        }
        for card_id in KEY_IDS:
            item[f"hand_{card_id}"] = row[f"hand_{card_id}"]
            item[f"play_{card_id}"] = row[f"play_{card_id}"]
            item[f"offer_{card_id}"] = int(card_id in row["offered"])
            item[f"take_{card_id}"] = row["taken"].get(card_id, 0)
        item["offer_evolve_grim"] = int(
            ("evolve", mf.GRIMMSNARL_EX_ID) in row["offered"]
        )
        item["take_evolve_grim"] = row["taken"].get(
            ("evolve", mf.GRIMMSNARL_EX_ID), 0
        )
        out_rows.append(item)

    game = {
        "won": int(ours > (theirs if theirs is not None else 0)),
        "went_first": int(went_first),
        "opponent_family": family(decks[1 - seat]),
    }
    game.update(setup)
    return {"game": game, "rows": out_rows}


def _job(task: tuple) -> dict[str, Any] | None:
    source, episode_id, path, seat, pilot, rating = task
    result = walk(Path(path), seat)
    if result is None:
        return None
    result["key"] = {
        "source": source, "episode_id": episode_id, "seat": seat,
        "pilot": pilot, "rating": rating,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="data/runs/grimmsnarl/20260813_grimmsnarl_ml_v21_sub55456713",
    )
    parser.add_argument("--limit-per-pilot", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = base.v21_tasks(ROOT / args.run)
    tasks += base.field_tasks(args.limit_per_pilot or None)
    print(f"tasks: {len(tasks)}", flush=True)

    games: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_job, tasks, chunksize=4):
            done += 1
            if done % 400 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
            if result is None:
                continue
            key = result["key"]
            game = dict(key)
            game.update(result["game"])
            games.append(game)
            for row in result["rows"]:
                merged = dict(key)
                merged.update(row)
                turn_rows.append(merged)

    import pandas as pd

    pd.DataFrame(games).to_parquet(OUT / "open_games.parquet")
    pd.DataFrame(turn_rows).to_parquet(OUT / "open_turns.parquet")
    print(f"games={len(games)} rows={len(turn_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
