"""Matched-decision extractor: our v21 ladder run vs same-deck field pilots.

One pass over a replay produces three tables for the seat that is playing our
exact 60 (deck hash 9714ab5c3996f6cc):

* ``games``     - one row per (episode, seat): outcome, turn order, opponent
                  family, number of own turns.
* ``turns``     - one row per (episode, seat, own_turn): what was offered, what
                  was taken, and the board snapshot at the end of that own turn.
* ``decisions`` - aggregated counters keyed by (context, chosen kind) and by
                  (context, offered kind) so take rates can be built per
                  decision class without keeping every raw record.

Everything is per OWN turn, never per decision: current.turn is shared between
seats, so a shared turn is halved with the turn-order flag read from a LATE
step (firstPlayer is -1 until the coin flip resolves).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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

OUR_DECK_HASH = "9714ab5c3996f6cc"
OUT = Path(__file__).resolve().parent / "out"


def _decks(steps: list[Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[seat] = [int(value) for value in action]
    return decks


def _energy_in_play(player: dict[str, Any]) -> int:
    return sum(mf._energy_count(card) for card in mf._in_play(player))


def _dark_in_play(player: dict[str, Any]) -> int:
    return sum(mf._dark_energy_count(card) for card in mf._in_play(player))


def _hand_count(player: dict[str, Any]) -> int:
    """The opponent's hand list is hidden, so prefer the public count."""
    value = player.get("handCount")
    if isinstance(value, int):
        return value
    return len(mf._cards(player, "hand"))


def _prize_left(player: dict[str, Any]) -> int | None:
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    if isinstance(prize, int):
        return prize
    return None


def _own_turn(turn: int, went_first: bool) -> int:
    return (turn + 1) // 2 if went_first else turn // 2


def _snapshot(player: dict[str, Any], opponent: dict[str, Any]) -> dict[str, Any]:
    in_play = mf._in_play(player)
    ids = [int(card.get("id", -1)) for card in in_play]
    grim = [
        card for card in in_play
        if int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
    ]
    active = (mf._cards(player, "active") or [{}])[0]
    return {
        "bench": len(mf._cards(player, "bench")),
        "bodies": len(in_play),
        "energy_in_play": sum(mf._energy_count(card) for card in in_play),
        "dark_in_play": sum(mf._dark_energy_count(card) for card in in_play),
        "grim_ex_on_board": int(bool(grim)),
        "grim_ex_ready": int(any(
            mf._dark_energy_count(card) >= mf.SHADOW_BULLET_COST for card in grim
        )),
        "grim_ex_active": int(int(active.get("id", -1)) == mf.GRIMMSNARL_EX_ID),
        "grim_ex_count": len(grim),
        "morgrem_on_board": int(mf.MORGREM_ID in ids),
        "impidimp_on_board": int(mf.IMPIDIMP_ID in ids),
        "froslass_on_board": int(mf.FROSLASS_ID in ids),
        "munkidori_on_board": int(mf.MUNKIDORI_ID in ids),
        "hand": _hand_count(player),
        "deck_left": player.get("deckCount"),
        "prize_left": _prize_left(player),
        "opp_prize_left": _prize_left(opponent),
        "active_hp": int(active.get("hp", 0)),
    }


def walk(path: Path, seat: int) -> dict[str, Any] | None:
    try:
        replay = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    steps = replay.get("steps") or []
    decks = _decks(steps)
    if decks[seat] is None or deck_hash(decks[seat]) != OUR_DECK_HASH:
        return None

    # Turn order from the latest ACTIVE view only; INACTIVE views are stale and
    # firstPlayer is -1 until the flip resolves.
    went_first: bool | None = None
    max_turn = 0
    for step in reversed(steps):
        for actor in (0, 1):
            if actor >= len(step):
                continue
            current = ((step[actor] or {}).get("observation") or {}).get("current")
            if isinstance(current, dict):
                first = int(current.get("firstPlayer", -1))
                if first >= 0 and went_first is None:
                    went_first = first == seat
                max_turn = max(max_turn, int(current.get("turn", 0)))
        if went_first is not None:
            break
    if went_first is None:
        return None

    rewards = replay.get("rewards") or [None, None]
    ours, theirs = rewards[seat], rewards[1 - seat]
    if ours is None:
        return None

    turns: dict[int, dict[str, Any]] = {}
    snapshots: dict[int, tuple[int, dict[str, Any]]] = {}
    end_of: dict[int, dict[str, Any]] = {}
    pending: set[int] = set()
    ctx_taken: Counter[tuple[int, str]] = Counter()
    ctx_offered: Counter[tuple[int, str]] = Counter()
    ctx_decisions: Counter[int] = Counter()
    ctx_multi: Counter[int] = Counter()
    card_taken: Counter[tuple[str, int]] = Counter()

    def turn_row(own: int, shared: int) -> dict[str, Any]:
        row = turns.get(own)
        if row is None:
            row = {
                "own_turn": own,
                "shared_turn": shared,
                "decisions": 0,
                "main_decisions": 0,
                "taken": Counter(),
                "offered_turn": set(),
                "cards": Counter(),
                "energy_attached_flag": 0,
                "supporter_flag": 0,
                "retreat_flag": 0,
            }
            turns[own] = row
        return row

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

            # Board snapshot: only the ACTIVE seat's view is fresh, and the
            # players array is indexed by absolute seat (verified: the
            # opponent's ``hand`` is hidden in the other seat's view while
            # ``deckCount``/``prize`` agree across both views).
            #
            # ``end_of[t]`` is the first observation with a turn strictly
            # greater than ``t``: that is the only point where a knock-out
            # taken on the last action of turn ``t`` is guaranteed to have
            # been credited.  ``last_in[t]`` is the fallback for the final
            # turn of the game.
            snap = _snapshot(players[seat], players[1 - seat])
            prev = snapshots.get(turn)
            if prev is None or index > prev[0]:
                snapshots[turn] = (index, snap)
            for done in list(pending):
                if turn > done:
                    end_of[done] = snap
                    pending.discard(done)
            pending.add(turn)

            if actor != seat:
                continue
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            if not options:
                continue
            context = int(select.get("context", -1))
            max_count = int(select.get("maxCount", 1) or 1)
            action = (steps[index + 1][actor] or {}).get("action")
            picked = [
                int(value) for value in action
                if isinstance(value, int) and 0 <= int(value) < len(options)
            ] if isinstance(action, list) else []

            own = _own_turn(turn, went_first) if turn > 0 else 0
            row = turn_row(own, turn)
            row["decisions"] += 1
            ctx_decisions[context] += 1
            if max_count > 1:
                ctx_multi[context] += 1
            if context == mf.MAIN_CONTEXT:
                row["main_decisions"] += 1
                row["energy_attached_flag"] = max(
                    row["energy_attached_flag"], int(bool(current.get("energyAttached")))
                )
                row["supporter_flag"] = max(
                    row["supporter_flag"], int(bool(current.get("supporterPlayed")))
                )
                row["retreat_flag"] = max(
                    row["retreat_flag"], int(bool(current.get("retreated")))
                )

            offered_kinds: set[str] = set()
            for option in options:
                try:
                    kind = mf.action_type(current, option, select)
                except Exception:  # noqa: BLE001
                    kind = "?"
                offered_kinds.add(kind)
            for kind in offered_kinds:
                ctx_offered[(context, kind)] += 1
            if context == mf.MAIN_CONTEXT:
                row["offered_turn"] |= offered_kinds

            for choice in picked:
                option = options[choice]
                try:
                    kind = mf.action_type(current, option, select)
                except Exception:  # noqa: BLE001
                    kind = "?"
                ctx_taken[(context, kind)] += 1
                card = mf.candidate_card(current, option, select) or {}
                card_id = int(card.get("id", -1))
                if kind == "attack":
                    card_id = mf._int(option.get("attackId"))
                card_taken[(kind, card_id)] += 1
                if context == mf.MAIN_CONTEXT:
                    row["taken"][kind] += 1
                    row["cards"][(kind, card_id)] += 1

    ordinal = [own for own in sorted(turns) if own > 0]
    game = {
        "won": int(ours > (theirs if theirs is not None else 0)),
        "went_first": int(went_first),
        "opponent_family": family(decks[1 - seat]),
        "opponent_deck_hash": deck_hash(decks[1 - seat]) if decks[1 - seat] else "",
        "shared_turns": max_turn,
        "own_turns": len(ordinal),
        "reward": ours,
    }
    last_turn = max(snapshots) if snapshots else 0
    if snapshots:
        game.update({
            "final_" + name: value
            for name, value in snapshots[last_turn][1].items()
        })

    turn_rows: list[dict[str, Any]] = []
    for own in sorted(turns):
        row = turns[own]
        shared = row["shared_turn"]
        snap = end_of.get(shared) or snapshots.get(shared, (0, {}))[1]
        out = {
            "own_turn": own,
            "shared_turn": shared,
            "decisions": row["decisions"],
            "main_decisions": row["main_decisions"],
            "energy_attached_flag": row["energy_attached_flag"],
            "supporter_flag": row["supporter_flag"],
            "retreat_flag": row["retreat_flag"],
        }
        for kind in mf.ACTION_TYPES:
            out["take_" + kind] = row["taken"].get(kind, 0)
            out["offer_" + kind] = int(kind in row["offered_turn"])
        out["take_shadow"] = row["cards"].get(("attack", mf.SHADOW_BULLET_ID), 0)
        out["take_adrena"] = row["cards"].get(("ability", mf.MUNKIDORI_ID), 0)
        out["take_grim_evolve"] = row["cards"].get(("evolve", mf.GRIMMSNARL_EX_ID), 0)
        out["take_morgrem_evolve"] = row["cards"].get(("evolve", mf.MORGREM_ID), 0)
        out["take_froslass_evolve"] = row["cards"].get(("evolve", mf.FROSLASS_ID), 0)
        out["take_rare_candy"] = row["cards"].get(("item", mf.RARE_CANDY_ID), 0)
        out["take_poffin"] = row["cards"].get(("item", mf.POFFIN_ID), 0)
        out["take_petrel"] = row["cards"].get(("supporter", mf.PETREL_ID), 0)
        out["take_lillie"] = row["cards"].get(("supporter", mf.LILLIE_ID), 0)
        out["take_dawn"] = row["cards"].get(("supporter", mf.DAWN_ID), 0)
        out["take_gym"] = row["cards"].get(("stadium", mf.SPIKEMUTH_GYM_ID), 0)
        out["idle"] = int(
            row["main_decisions"] > 0
            and sum(
                count for kind, count in row["taken"].items()
                if kind not in {"end"}
            ) == 0
        )
        for name, value in snap.items():
            out["board_" + name] = value
        turn_rows.append(out)

    return {
        "game": game,
        "turns": turn_rows,
        "ctx_taken": {f"{k[0]}|{k[1]}": v for k, v in ctx_taken.items()},
        "ctx_offered": {f"{k[0]}|{k[1]}": v for k, v in ctx_offered.items()},
        "ctx_decisions": {str(k): v for k, v in ctx_decisions.items()},
        "ctx_multi": {str(k): v for k, v in ctx_multi.items()},
        "card_taken": {f"{k[0]}|{k[1]}": v for k, v in card_taken.items()},
    }


def _job(task: tuple[str, str, str, int, str, float]) -> dict[str, Any] | None:
    source, episode_id, path, seat, pilot, rating = task
    result = walk(Path(path), seat)
    if result is None:
        return None
    result["key"] = {
        "source": source,
        "episode_id": episode_id,
        "seat": seat,
        "pilot": pilot,
        "rating": rating,
    }
    return result


def v21_tasks(run_dir: Path) -> list[tuple]:
    tasks: list[tuple] = []
    for entry in csv.DictReader((run_dir / "manifest.csv").open(encoding="utf-8-sig")):
        seat_text = entry.get("detected_submission_agent_index", "")
        if seat_text not in {"0", "1"}:
            continue
        episode_id = entry["episode_id"]
        path = run_dir / "episodes" / episode_id / "replay" / f"episode_{episode_id}.json"
        if path.exists():
            tasks.append(("v21", episode_id, str(path), int(seat_text), "v21", 948.2))
    return tasks


def field_tasks(limit_per_pilot: int | None) -> list[tuple]:
    index = ROOT / "data" / "kaggle_grimmsnarl_top50" / "indexes" / "replay_index.csv"
    base = index.parent.parent
    per_pilot: dict[str, list[tuple]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in csv.DictReader(index.open(encoding="utf-8-sig")):
        if row.get("deck_hash") != OUR_DECK_HASH:
            continue
        if row.get("download_status") not in {"success", "skipped_existing"}:
            continue
        seat_text = row.get("seat_index", "")
        if seat_text not in {"0", "1"}:
            continue
        seat = int(seat_text)
        episode_id = row["episode_id"]
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = base / row["replay_path"].replace("\\", "/")
        if not path.exists():
            continue
        pilot = row["submission_id"]
        per_pilot[pilot].append(
            ("field", episode_id, str(path), seat, pilot, float(row["submission_score"]))
        )
    tasks: list[tuple] = []
    for pilot, items in per_pilot.items():
        items.sort(key=lambda item: int(item[1]))
        tasks.extend(items if limit_per_pilot is None else items[-limit_per_pilot:])
    return tasks


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
    tasks = v21_tasks(ROOT / args.run)
    tasks += field_tasks(args.limit_per_pilot or None)
    print(f"tasks: {len(tasks)}", flush=True)

    games: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    ctx: dict[str, Counter] = defaultdict(Counter)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_job, tasks, chunksize=4):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
            if result is None:
                continue
            key = result["key"]
            game = dict(key)
            game.update(result["game"])
            games.append(game)
            for row in result["turns"]:
                merged = dict(key)
                merged.update(row)
                turn_rows.append(merged)
            for name in ("ctx_taken", "ctx_offered", "ctx_decisions", "ctx_multi",
                         "card_taken"):
                bucket = ctx[f"{key['source']}|{key['pilot']}|{name}"]
                for item, count in result[name].items():
                    bucket[item] += count

    import pandas as pd

    pd.DataFrame(games).to_parquet(OUT / "games.parquet")
    pd.DataFrame(turn_rows).to_parquet(OUT / "turns.parquet")
    flat = [
        {"source": k.split("|")[0], "pilot": k.split("|")[1],
         "table": k.split("|")[2], "item": item, "count": count}
        for k, bucket in ctx.items() for item, count in bucket.items()
    ]
    pd.DataFrame(flat).to_parquet(OUT / "contexts.parquet")
    print(f"games={len(games)} turn_rows={len(turn_rows)} ctx_rows={len(flat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
