"""Where the second seat's games are decided, on the pooled v22-equivalent corpus.

The v27 footprint probe reproduces v22's action on 2747 of 2755 stored
decisions, so v22, v26 and v27 are one policy sampled at three times and can
be pooled to 264 games.  On that pool the second seat is the largest single
cell that moved: 0.585 against 0.688, and inside the 08-15 window 0.467
against 0.785 while the field's first attack moved from own turn 2.3 to 2.0.

This script asks what separates a won second-seat game from a lost one, and -
because [[grimmsnarl-v24-froslass-lever-was-a-confound]] retired reading
levers off win-rate splits - it separates two things that a raw split cannot:

* *offer side*: on how many of our own turns was the accelerating play even
  legal (Rare Candy in hand with a Morgrem/Impidimp to jump, a ready
  Grimmsnarl ex to attack with);
* *take side*: given that it was offered, did we take it.

A gap that is entirely offer-side is a deck/draw fact and no policy change can
reach it.  Only a take-side gap is a lever.
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
for path in (ROOT, ROOT / "scripts", ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
RUN_ROOT = ROOT / "data/runs/grimmsnarl"
POOL = (
    ("v22_a", 55479857, "20260813_grimmsnarl_ml_v22_sub55479857"),
    ("v22_b", 55483874, "20260814_grimmsnarl_ml_v22_b_sub55483874"),
    ("v22_c", 55486680, "20260814_grimmsnarl_ml_v22_c_sub55486680"),
    ("v22_d", 55486691, "20260814_grimmsnarl_ml_v22_d_sub55486691"),
    ("v26", 55520389, "20260815_grimmsnarl_ml_v26_sub55520389"),
    ("v27", 55521760, "20260815_grimmsnarl_ml_v27_sub55521760"),
)


def cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [c for c in (player.get(area) or []) if isinstance(c, dict)]


def hand_ids(player: dict[str, Any]) -> Counter:
    return Counter(
        int(c.get("id", -1)) for c in cards(player, "hand")
    )


def walk(replay: dict[str, Any], seat: int) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    if len(steps) < 2:
        return None
    decks: list[list[int] | None] = [None, None]
    for side in (0, 1):
        action = (steps[1][side] or {}).get("action")
        if isinstance(action, list) and len(action) == 60:
            decks[side] = [int(v) for v in action]
    if decks[seat] is None or deck_hash(decks[seat]) != OUR_DECK_HASH:
        return None
    rewards = replay.get("rewards") or [None, None]
    if rewards[seat] is None:
        return None
    won = int(rewards[seat] > (rewards[1 - seat] or 0))

    went_first: bool | None = None
    ready_turn = shadow_turn = None
    candy_offered_turns: set[int] = set()
    candy_used_turns: set[int] = set()
    attack_offered_turns: set[int] = set()
    attack_taken_turns: set[int] = set()
    main_turns: set[int] = set()
    grim_on_board_turn = None

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
        options = list(select.get("option") or [])
        if len(players) < 2 or not options:
            continue
        turn = int(current.get("turn", -1))
        if went_first is None:
            first = int(current.get("firstPlayer", -1))
            if first >= 0:
                went_first = first == seat
        me = players[seat]
        if int(select.get("context", -1)) == mf.MAIN_CONTEXT:
            main_turns.add(turn)
            board = cards(me, "active") + cards(me, "bench")
            if grim_on_board_turn is None and any(
                int(c.get("id", -1)) == mf.GRIMMSNARL_EX_ID for c in board
            ):
                grim_on_board_turn = turn
            if ready_turn is None and any(
                int(c.get("id", -1)) == mf.GRIMMSNARL_EX_ID
                and mf._dark_energy_count(c) >= mf.SHADOW_BULLET_COST
                for c in board
            ):
                ready_turn = turn
            hand = hand_ids(me)
            # Rare Candy is only an accelerant when a basic Impidimp is on the
            # board and a Grimmsnarl ex is in hand to land on it.
            has_target = any(
                int(c.get("id", -1)) == mf.IMPIDIMP_ID for c in board
            )
            if (
                hand.get(mf.RARE_CANDY_ID)
                and hand.get(mf.GRIMMSNARL_EX_ID)
                and has_target
            ):
                candy_offered_turns.add(turn)
        action = (steps[index + 1][seat] or {}).get("action")
        picked = [
            int(v) for v in action
            if isinstance(v, int) and 0 <= int(v) < len(options)
        ] if isinstance(action, list) else []
        for choice in picked:
            option = options[choice]
            try:
                kind = mf.action_type(current, option, select)
            except Exception:  # noqa: BLE001
                continue
            card = mf.candidate_card(current, option, select) or {}
            if kind == "attack" and mf._int(option.get("attackId")) == mf.SHADOW_BULLET_ID:
                attack_taken_turns.add(turn)
                if shadow_turn is None:
                    shadow_turn = turn
            if int(card.get("id", -1)) == mf.RARE_CANDY_ID and kind not in (
                "attack", "ability", "end", "retreat"
            ):
                candy_used_turns.add(turn)
        # Was a Shadow Bullet even on the menu this decision?
        for option in options:
            if mf._int(option.get("attackId")) == mf.SHADOW_BULLET_ID:
                attack_offered_turns.add(turn)
                break

    def own(turn: int | None) -> int | None:
        if turn is None:
            return None
        if went_first is None:
            return (turn + 1) // 2
        return (turn + 1) // 2 if went_first else turn // 2

    return {
        "won": won,
        "went_first": (
            "" if went_first is None else ("first" if went_first else "second")
        ),
        "opponent_family": "",
        "own_ready_turn": own(ready_turn),
        "own_shadow_turn": own(shadow_turn),
        "own_grim_turn": own(grim_on_board_turn),
        "candy_offer_turns": len(candy_offered_turns),
        "candy_used_turns": len(candy_used_turns),
        "candy_offered_not_used": len(candy_offered_turns - candy_used_turns),
        "attack_offer_turns": len(attack_offered_turns),
        "attack_taken_turns": len(attack_taken_turns),
        "attack_offered_not_taken": len(
            attack_offered_turns - attack_taken_turns
        ),
        "our_main_turns": len(main_turns),
    }


def block(rows, label: str, width: int = 34) -> str:
    if not rows:
        return f"{label:<{width}} n=  0"
    wins = sum(r["won"] for r in rows)
    low, high = wilson(wins, len(rows))
    return (f"{label:<{width}} n={len(rows):>3}  {wins:>3}-{len(rows) - wins:<3} "
            f"{wins / len(rows):.3f} [{low:.3f},{high:.3f}]")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/second_seat_clock.json",
    )
    args = parser.parse_args()

    families = {}
    for raw in csv.DictReader(
        (ROOT / "experiments/grimmsnarl_ml_v27/version_games.csv").open(
            encoding="utf-8-sig"
        )
    ):
        families[int(raw["episode_id"])] = (
            raw["opponent_family"], raw["create_time"][:10],
            float(raw["opponent_rating"]) if raw["opponent_rating"] else None,
        )

    rows: list[dict[str, Any]] = []
    for label, submission, folder in POOL:
        run = RUN_ROOT / folder
        for meta in csv.DictReader(
            (run / "episodes.csv").open(encoding="utf-8-sig")
        ):
            if meta.get("state") != "COMPLETED":
                continue
            if meta.get("episode_type") == "EPISODE_TYPE_VALIDATION":
                continue
            if meta.get("agent_0_submission_id") == str(submission):
                seat = 0
            elif meta.get("agent_1_submission_id") == str(submission):
                seat = 1
            else:
                continue
            episode_id = int(meta["episode_id"])
            path = (
                run / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not path.exists():
                continue
            row = walk(json.loads(path.read_text(encoding="utf-8")), seat)
            if row is None:
                continue
            family, day, rating = families.get(episode_id, ("", "", None))
            row.update({
                "version": label, "episode_id": episode_id,
                "opponent_family": family, "day": day,
                "opponent_rating": rating,
                "late": day >= "2026-08-15",
            })
            rows.append(row)
    print(f"pooled v22-equivalent games: {len(rows)}")

    second = [r for r in rows if r["went_first"] == "second"]
    first = [r for r in rows if r["went_first"] == "first"]
    print("\n" + block(first, "going first"))
    print(block(second, "going second"))

    print("\n=== win rate by our own first Shadow Bullet turn ===")
    print("This is a contrast, not an intervention.  Read it as where the "
          "games are, not as a proven cause.\n")
    for order, subset in (("first", first), ("second", second)):
        print(f"-- going {order}")
        buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            key = row["own_shadow_turn"]
            buckets["never" if key is None else min(key, 5)].append(row)
        for key in sorted(buckets, key=lambda k: (k == "never", k)):
            print("   " + block(buckets[key], f"own shadow turn {key}", 30))

    print("\n=== win rate by our own first ready-Grimmsnarl turn ===")
    for order, subset in (("first", first), ("second", second)):
        print(f"-- going {order}")
        buckets = defaultdict(list)
        for row in subset:
            key = row["own_ready_turn"]
            buckets["never" if key is None else min(key, 5)].append(row)
        for key in sorted(buckets, key=lambda k: (k == "never", k)):
            print("   " + block(buckets[key], f"own ready turn {key}", 30))

    print("\n=== offer side vs take side ===")
    print("Rare Candy 'offered' = a turn where Rare Candy and Grimmsnarl ex "
          "were both in hand with an Impidimp on board.\n")
    print(f"{'cell':<28}{'games':>7}{'candy offers':>14}{'candy used':>12}"
          f"{'offered-unused':>16}{'shadow offers':>15}{'shadow unused':>15}")
    for label, subset in (
        ("first, won", [r for r in first if r["won"]]),
        ("first, lost", [r for r in first if not r["won"]]),
        ("second, won", [r for r in second if r["won"]]),
        ("second, lost", [r for r in second if not r["won"]]),
        ("second, 08-15", [r for r in second if r["late"]]),
        ("second, earlier", [r for r in second if not r["late"]]),
        ("first, 08-15", [r for r in first if r["late"]]),
        ("first, earlier", [r for r in first if not r["late"]]),
    ):
        if not subset:
            continue
        def avg(key: str) -> float:
            return sum(r[key] for r in subset) / len(subset)
        print(
            f"{label:<28}{len(subset):>7}{avg('candy_offer_turns'):>14.2f}"
            f"{avg('candy_used_turns'):>12.2f}"
            f"{avg('candy_offered_not_used'):>16.2f}"
            f"{avg('attack_offer_turns'):>15.2f}"
            f"{avg('attack_offered_not_taken'):>15.2f}"
        )

    print("\n=== second seat by matchup, 08-15 vs earlier ===")
    for family in sorted(
        {r["opponent_family"] for r in second},
        key=lambda f: -sum(1 for r in second if r["opponent_family"] == f),
    )[:8]:
        early = [r for r in second
                 if r["opponent_family"] == family and not r["late"]]
        late = [r for r in second if r["opponent_family"] == family and r["late"]]
        print(
            f"  {family:<28} earlier {block(early, '', 0)}   "
            f"08-15 {block(late, '', 0)}"
        )

    payload = {"games": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
