"""Outcome decomposition for the v19/v20 Grimmsnarl ladder runs.

Stage 1 of the v20 post-mortem.  One row per episode, read from the stored
replay only, so every column is an observed fact and not a model opinion:

* opponent archetype family and the rating the opponent carried into the
  pairing (``initialScore`` from ``episodes.csv``, not the current board);
* turn order from ``current.firstPlayer`` read at a late step, because the
  field is ``-1`` until the coin flip resolves;
* the attack-access gate that has been the v15 regression check ever since:
  the first turn a ready Grimmsnarl ex existed anywhere on our board versus
  the first turn we actually used Shadow Bullet;
* how the game ended - prizes on both sides, deck counts, bodies left - so a
  blowout and a one-prize loss are not averaged into the same number.

The two runs are small (43 and 47 games), so every rate is printed with a
Wilson interval and the aggregate over both is printed as well.  Nothing here
promotes a version; it says where the losses are.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v20"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_matchup_ceiling import family, wilson  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"

DEFAULT_RUNS = (
    ("v19", "data/runs/grimmsnarl/20260812_grimmsnarl_ml_v19_sub55445763"),
    ("v20", "data/runs/grimmsnarl/20260812_grimmsnarl_ml_v20_sub55445769"),
)


def _decks(steps: list[Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[seat] = [int(value) for value in action]
    return decks


def _late_current(steps: list[Any], seat: int) -> dict[str, Any] | None:
    """Latest board state from *either* seat's view.

    Reading only our own seat stops at the last step we were ACTIVE, which on a
    board-out loss is several turns before the end - episode 92187003 looked
    like 6 prizes left on both sides when the opponent had already taken two
    and swept our board.  Both prize counts and both benches are public, so the
    opponent's later view is the correct source; ``yourIndex`` says which way
    ``players`` is oriented.
    """
    best: dict[str, Any] | None = None
    best_key = (-1, -1)
    for index, step in enumerate(steps):
        for actor in (0, 1):
            if actor >= len(step):
                continue
            current = ((step[actor] or {}).get("observation") or {}).get("current")
            if not (isinstance(current, dict) and current.get("players")):
                continue
            # An INACTIVE seat keeps a stale observation, so rank by the turn
            # the view reports and only then by step order.
            key = (int(current.get("turn", -1)), index)
            if key > best_key:
                best, best_key = current, key
    return best


def _late_sides(
    current: dict[str, Any] | None, seat: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not current:
        return {}, {}
    players = current.get("players") or []
    if len(players) < 2:
        return {}, {}
    # ``players`` is indexed by absolute seat (``yourIndex`` == the viewer's
    # seat), so no re-mapping is needed - only the *step* had to be fixed.
    return players[seat], players[1 - seat]


def _ready_grim(player: dict[str, Any]) -> bool:
    for card in mf._cards(player, "active") + mf._cards(player, "bench"):
        if (
            int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf._dark_energy_count(card) >= mf.SHADOW_BULLET_COST
        ):
            return True
    return False


def _bodies(player: dict[str, Any]) -> int:
    return len(mf._cards(player, "active")) + len(mf._cards(player, "bench"))


def _prize_left(player: dict[str, Any]) -> int | None:
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    if isinstance(prize, int):
        return prize
    return None


def _own_turn(turn: int, went_first: bool | None) -> int:
    """Turn ordinal for our own seat; ``current.turn`` is shared."""
    if went_first is None:
        return (turn + 1) // 2
    return (turn + 1) // 2 if went_first else turn // 2


def walk_episode(replay: dict[str, Any], seat: int) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    decks = _decks(steps)
    if decks[seat] is None or deck_hash(decks[seat]) != OUR_DECK_HASH:
        return None

    current_late = _late_current(steps, seat)
    went_first: bool | None = None
    if current_late is not None:
        first = int(current_late.get("firstPlayer", -1))
        went_first = (first == seat) if first >= 0 else None

    rewards = replay.get("rewards") or [None, None]
    ours, theirs = rewards[seat], rewards[1 - seat]
    if ours is None:
        return None
    won = bool(ours > (theirs if theirs is not None else 0))

    first_ready_turn: int | None = None
    first_shadow_turn: int | None = None
    first_any_attack_turn: int | None = None
    opp_first_attack_turn: int | None = None
    our_main_turns: set[int] = set()
    shadow_count = 0
    attacks: Counter[int] = Counter()
    max_turn = 0
    grim_evolutions = 0
    rare_candies = 0
    adrenas = 0
    froslass_evolves = 0
    stamps = 0
    bosses = 0

    for index, step in enumerate(steps[:-1]):
        for actor in (0, 1):
            if actor >= len(step) or actor >= len(steps[index + 1]):
                continue
            record = step[actor] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            current = observation.get("current") or {}
            players = current.get("players") or []
            if len(players) < 2 or not options:
                continue
            turn = int(current.get("turn", -1))
            max_turn = max(max_turn, turn)
            action = (steps[index + 1][actor] or {}).get("action")
            picked = [
                int(value) for value in action
                if isinstance(value, int) and 0 <= int(value) < len(options)
            ] if isinstance(action, list) else []

            mine = actor == seat
            if mine:
                if int(select.get("context", -1)) == mf.MAIN_CONTEXT:
                    our_main_turns.add(turn)
                if first_ready_turn is None and _ready_grim(players[actor]):
                    first_ready_turn = turn

            for choice in picked:
                option = options[choice]
                try:
                    kind = mf.action_type(current, option, select)
                except Exception:  # noqa: BLE001
                    kind = "?"
                attack_id = mf._int(option.get("attackId"))
                if kind == "attack":
                    if mine:
                        attacks[attack_id] += 1
                        if first_any_attack_turn is None:
                            first_any_attack_turn = turn
                        if attack_id == mf.SHADOW_BULLET_ID:
                            shadow_count += 1
                            if first_shadow_turn is None:
                                first_shadow_turn = turn
                    elif opp_first_attack_turn is None:
                        opp_first_attack_turn = turn
                if not mine:
                    continue
                card = mf.candidate_card(current, option, select) or {}
                card_id = int(card.get("id", -1))
                if kind == "ability":
                    # Adrena-Brain is Munkidori's activated ability.  Punk Up
                    # is *not* activated - it triggers when Grimmsnarl ex is
                    # played from hand to evolve, so it is counted below as an
                    # evolution, and an ability option holding 648 never
                    # appears.  Spikemuth Gym resolves to area 7 and is
                    # deliberately not counted as either.
                    if card_id == mf.MUNKIDORI_ID:
                        adrenas += 1
                if kind == "evolve" and card_id == mf.GRIMMSNARL_EX_ID:
                    grim_evolutions += 1
                if kind not in {"attack", "ability", "end", "retreat"}:
                    if card_id == mf.RARE_CANDY_ID:
                        rare_candies += 1
                    if card_id == mf.FROSLASS_ID:
                        froslass_evolves += 1
                    elif card_id == mf.UNFAIR_STAMP_ID:
                        stamps += 1
                    elif card_id == mf.BOSS_ID:
                        bosses += 1

    us_late, them_late = _late_sides(current_late, seat)

    return {
        "won": won,
        "went_first": went_first,
        "opponent_family": family(decks[1 - seat]),
        "opponent_deck_hash": deck_hash(decks[1 - seat]) if decks[1 - seat] else "",
        "turns": max_turn,
        "our_turns": len(our_main_turns),
        "our_prize_left": _prize_left(us_late),
        "opp_prize_left": _prize_left(them_late),
        "our_deck_left": us_late.get("deckCount"),
        "opp_deck_left": them_late.get("deckCount"),
        "our_bodies_left": _bodies(us_late) if us_late else None,
        "first_ready_turn": first_ready_turn,
        "first_shadow_turn": first_shadow_turn,
        "first_attack_turn": first_any_attack_turn,
        "opp_first_attack_turn": opp_first_attack_turn,
        "own_first_shadow_turn": (
            _own_turn(first_shadow_turn, went_first)
            if first_shadow_turn is not None else None
        ),
        "own_first_ready_turn": (
            _own_turn(first_ready_turn, went_first)
            if first_ready_turn is not None else None
        ),
        "shadow_attacks": shadow_count,
        "attacks": sum(attacks.values()),
        "grim_evolutions": grim_evolutions,
        "rare_candies": rare_candies,
        "adrena_brains": adrenas,
        "froslass_evolves": froslass_evolves,
        "stamps": stamps,
        "bosses": bosses,
    }


def load_run(label: str, run_dir: Path) -> list[dict[str, Any]]:
    manifest = {
        row["episode_id"]: row
        for row in csv.DictReader(
            (run_dir / "manifest.csv").open(encoding="utf-8-sig")
        )
    }
    episodes = {
        row["episode_id"]: row
        for row in csv.DictReader(
            (run_dir / "episodes.csv").open(encoding="utf-8-sig")
        )
    }

    rows: list[dict[str, Any]] = []
    for episode_id, entry in manifest.items():
        seat_text = entry.get("detected_submission_agent_index", "")
        if seat_text not in {"0", "1"}:
            continue
        seat = int(seat_text)
        path = run_dir / "episodes" / episode_id / "replay" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        row = walk_episode(replay, seat)
        if row is None:
            continue
        meta = episodes.get(episode_id, {})
        opponent_rating = meta.get(f"agent_{1 - seat}_initial_score", "")
        our_rating = meta.get(f"agent_{seat}_initial_score", "")
        row.update(
            {
                "version": label,
                "episode_id": int(episode_id),
                "seat": seat,
                "create_time": meta.get("create_time", ""),
                "opponent_submission": meta.get(
                    f"agent_{1 - seat}_submission_id", ""
                ),
                "opponent_rating": float(opponent_rating) if opponent_rating else None,
                "our_rating": float(our_rating) if our_rating else None,
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: item["episode_id"])
    return rows


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for row in rows if row["won"])
    low, high = wilson(wins, len(rows)) if rows else (0.0, 0.0)
    return {
        "games": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": round(wins / len(rows), 4) if rows else None,
        "wilson95": [low, high],
    }


def group(rows: list[dict[str, Any]], key) -> dict[str, dict[str, Any]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return {
        str(name): block(items)
        for name, items in sorted(
            buckets.items(), key=lambda item: -len(item[1])
        )
    }


def rating_band(row: dict[str, Any]) -> str:
    rating = row.get("opponent_rating")
    if rating is None:
        return "unknown"
    for edge in (700, 800, 900, 1000, 1100):
        if rating < edge:
            return f"<{edge}"
    return ">=1100"


def mean(values: list[float]) -> float | None:
    numbers = [value for value in values if value is not None]
    return round(sum(numbers) / len(numbers), 3) if numbers else None


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": block(rows),
        "by_turn_order": group(
            rows,
            lambda row: {True: "first", False: "second", None: "unknown"}[
                row["went_first"]
            ],
        ),
        "by_family": group(rows, lambda row: row["opponent_family"]),
        "by_family_turn_order": group(
            rows,
            lambda row: (
                f"{row['opponent_family']} | "
                f"{ {True: 'first', False: 'second', None: 'unknown'}[row['went_first']] }"
            ),
        ),
        "by_rating_band": group(rows, rating_band),
        "tempo": {
            "own_first_shadow_turn": mean(
                [row["own_first_shadow_turn"] for row in rows]
            ),
            "own_first_ready_turn": mean(
                [row["own_first_ready_turn"] for row in rows]
            ),
            "gate_violations": sum(
                1 for row in rows
                if row["first_ready_turn"] is not None
                and row["first_shadow_turn"] is not None
                and row["first_shadow_turn"] > row["first_ready_turn"] + 1
            ),
            "never_shadowed": sum(
                1 for row in rows if row["first_shadow_turn"] is None
            ),
            "never_ready": sum(
                1 for row in rows if row["first_ready_turn"] is None
            ),
            "shadow_attacks_per_game": mean(
                [float(row["shadow_attacks"]) for row in rows]
            ),
            "grim_evolutions_per_game": mean(
                [float(row["grim_evolutions"]) for row in rows]
            ),
            "rare_candies_per_game": mean(
                [float(row["rare_candies"]) for row in rows]
            ),
            "adrena_brains_per_game": mean(
                [float(row["adrena_brains"]) for row in rows]
            ),
            "stamps_per_game": mean([float(row["stamps"]) for row in rows]),
            "bosses_per_game": mean([float(row["bosses"]) for row in rows]),
            "froslass_per_game": mean(
                [float(row["froslass_evolves"]) for row in rows]
            ),
        },
        "loss_anatomy": {
            "mean_our_prize_left_on_loss": mean(
                [
                    float(row["our_prize_left"])
                    for row in rows
                    if not row["won"] and row["our_prize_left"] is not None
                ]
            ),
            "mean_opp_prize_left_on_win": mean(
                [
                    float(row["opp_prize_left"])
                    for row in rows
                    if row["won"] and row["opp_prize_left"] is not None
                ]
            ),
            "losses_with_5plus_prizes_left": sum(
                1 for row in rows
                if not row["won"]
                and (row["our_prize_left"] or 0) >= 5
            ),
            "losses_with_1_prize_left": sum(
                1 for row in rows
                if not row["won"] and row["our_prize_left"] == 1
            ),
            "losses_board_out": sum(
                1 for row in rows
                if not row["won"] and (row["our_bodies_left"] or 0) == 0
            ),
            "mean_turns_on_loss": mean(
                [float(row["turns"]) for row in rows if not row["won"]]
            ),
            "mean_turns_on_win": mean(
                [float(row["turns"]) for row in rows if row["won"]]
            ),
            "prizes_taken_on_loss": dict(
                sorted(
                    Counter(
                        6 - (row["our_prize_left"] or 6)
                        for row in rows if not row["won"]
                    ).items()
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Extra run to include; defaults to the v19/v20 ladder pair.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20" / "ladder_v19_v20.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20" / "ladder_v19_v20_games.csv",
    )
    args = parser.parse_args()

    specs = list(DEFAULT_RUNS)
    for item in args.run:
        label, _, path = item.partition("=")
        specs.append((label, path))

    all_rows: list[dict[str, Any]] = []
    per_version: dict[str, Any] = {}
    for label, path in specs:
        run_dir = Path(path)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        rows = load_run(label, run_dir)
        per_version[label] = report(rows)
        all_rows.extend(rows)
        print(f"{label}: {len(rows)} games from {run_dir}")

    payload = {
        "runs": {label: path for label, path in specs},
        "per_version": per_version,
        "pooled": report(all_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = sorted({key for row in all_rows for key in row})
    with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps(payload["per_version"], ensure_ascii=False, indent=2))
    print("\n=== pooled ===")
    print(json.dumps(payload["pooled"], ensure_ascii=False, indent=2))
    print(f"\nJSON: {args.output}\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
