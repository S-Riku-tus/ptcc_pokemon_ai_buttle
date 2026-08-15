"""Build one per-game table across every stored Grimmsnarl ladder submission.

Every column is read from the stored replay or from ``episodes.csv``; nothing
is inferred from a model.  The table is the input for
``analyze_grimmsnarl_v27_vs_champions.py`` and for any later version
comparison, so all of the run-specific knowledge lives here once:

* identity - version label, submission id, episode id, seat, timestamps;
* pairing  - opponent submission, opponent rating carried into the pairing,
  our rating before and after, opponent deck hash and archetype family;
* outcome  - win/loss, prizes left on both sides, deck counts, bodies left,
  turn count, deck-out and board-out flags;
* tempo    - the v15 attack-access gate, first ready/shadow turns, per-game
  counts of the actions previous verdicts argued about;
* runtime  - Kaggle ``remainingOverageTime`` consumed by each seat and any
  non ACTIVE/INACTIVE/DONE status, which is how a search budget overrun or an
  agent crash would show up.

The runtime block is new for v26/v27: those are the first versions to spend a
real per-episode search budget, so "did the agent simply run out of clock"
has to be a measured column and not an assumption.
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
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"

RUN_ROOT = "data/runs/grimmsnarl"

# label, submission id, run directory under RUN_ROOT.  Ordered oldest first.
RUNS: tuple[tuple[str, int, str], ...] = (
    ("v22_a", 55479857, "20260813_grimmsnarl_ml_v22_sub55479857"),
    ("v22_b", 55483874, "20260814_grimmsnarl_ml_v22_b_sub55483874"),
    ("v22_c", 55486680, "20260814_grimmsnarl_ml_v22_c_sub55486680"),
    ("v22_d", 55486691, "20260814_grimmsnarl_ml_v22_d_sub55486691"),
    ("v23", 55485982, "20260814_grimmsnarl_ml_v23_sub55485982"),
    ("v24_a", 55496021, "20260814_grimmsnarl_ml_v24_a_sub55496021"),
    ("v24_b", 55496665, "20260814_grimmsnarl_ml_v24_b_sub55496665"),
    ("v25_a", 55507909, "20260815_grimmsnarl_ml_v25_sub55507909"),
    ("v25_b", 55517142, "20260815_grimmsnarl_ml_v25_b_sub55517142"),
    ("v26", 55520389, "20260815_grimmsnarl_ml_v26_sub55520389"),
    ("v27", 55521760, "20260815_grimmsnarl_ml_v27_sub55521760"),
)

# Families whose Active can be immune to Grimmsnarl ex's Shadow Bullet; the
# v25 verdict located the entire deficit here.
WALL_FAMILIES = {"Kangaskhan / Crustle", "Ogerpon"}


def _decks(steps: list[Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[seat] = [int(value) for value in action]
    return decks


def _late_current(steps: list[Any]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_key = (-1, -1)
    for index, step in enumerate(steps):
        for actor in (0, 1):
            if actor >= len(step):
                continue
            current = ((step[actor] or {}).get("observation") or {}).get("current")
            if not (isinstance(current, dict) and current.get("players")):
                continue
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
    if went_first is None:
        return (turn + 1) // 2
    return (turn + 1) // 2 if went_first else turn // 2


def _overage(steps: list[Any], seat: int) -> tuple[float | None, float | None]:
    """First and last ``remainingOverageTime`` seen for one seat.

    Kaggle decrements this only while the agent is thinking, so
    ``first - last`` is the wall-clock the agent actually spent on top of the
    per-act allowance.  ``actTimeout`` is 0 on this competition, so the overage
    bank *is* the whole clock.
    """
    first = last = None
    for step in steps:
        if seat >= len(step):
            continue
        value = ((step[seat] or {}).get("observation") or {}).get(
            "remainingOverageTime"
        )
        if not isinstance(value, (int, float)):
            continue
        if first is None:
            first = float(value)
        last = float(value)
    return first, last


def walk_episode(replay: dict[str, Any], seat: int) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    decks = _decks(steps)
    if decks[seat] is None or deck_hash(decks[seat]) != OUR_DECK_HASH:
        return None

    current_late = _late_current(steps)
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
    froslass_actions = 0
    froslass_true_evolutions = 0
    stamps = 0
    bosses = 0
    lillies = 0
    our_decisions = 0
    our_multi_pick = 0
    our_ends = 0

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
                our_decisions += 1
                if int(select.get("maxCount", 1) or 1) > 1:
                    our_multi_pick += 1
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
                if kind == "end":
                    our_ends += 1
                card = mf.candidate_card(current, option, select) or {}
                card_id = int(card.get("id", -1))
                if kind == "ability" and card_id == mf.MUNKIDORI_ID:
                    adrenas += 1
                if kind == "evolve":
                    if card_id == mf.GRIMMSNARL_EX_ID:
                        grim_evolutions += 1
                    if card_id == mf.FROSLASS_ID:
                        froslass_true_evolutions += 1
                if kind not in {"attack", "ability", "end", "retreat"}:
                    if card_id == mf.RARE_CANDY_ID:
                        rare_candies += 1
                    if card_id == mf.FROSLASS_ID:
                        froslass_actions += 1
                    elif card_id == mf.UNFAIR_STAMP_ID:
                        stamps += 1
                    elif card_id == mf.BOSS_ID:
                        bosses += 1
                    elif card_id == getattr(mf, "LILLIE_ID", -999):
                        lillies += 1

    us_late, them_late = _late_sides(current_late, seat)
    our_first, our_last = _overage(steps, seat)
    opp_first, opp_last = _overage(steps, 1 - seat)
    statuses = Counter(
        (record or {}).get("status") for step in steps for record in step
    )
    odd_status = {
        str(name): count
        for name, count in statuses.items()
        if name not in {"ACTIVE", "INACTIVE", "DONE", None}
    }

    opponent_family = family(decks[1 - seat])
    opponent_hash = deck_hash(decks[1 - seat]) if decks[1 - seat] else ""

    return {
        "won": int(won),
        "went_first": (
            "" if went_first is None else ("first" if went_first else "second")
        ),
        "opponent_family": opponent_family,
        "opponent_deck_hash": opponent_hash,
        "exact_mirror": int(opponent_hash == OUR_DECK_HASH),
        "wall_family": int(opponent_family in WALL_FAMILIES),
        "turns": max_turn,
        "our_turns": len(our_main_turns),
        "our_prize_left": _prize_left(us_late),
        "opp_prize_left": _prize_left(them_late),
        "our_deck_left": us_late.get("deckCount"),
        "opp_deck_left": them_late.get("deckCount"),
        "our_bodies_left": _bodies(us_late) if us_late else None,
        "board_out": int(bool(us_late) and _bodies(us_late) == 0),
        "deck_out": int(us_late.get("deckCount") == 0) if us_late else 0,
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
        "gate_violation": int(
            first_ready_turn is not None
            and first_shadow_turn is not None
            and first_shadow_turn > first_ready_turn + 1
        ),
        "shadow_attacks": shadow_count,
        "attacks": sum(attacks.values()),
        "grim_evolutions": grim_evolutions,
        "rare_candies": rare_candies,
        "adrena_brains": adrenas,
        "froslass_actions": froslass_actions,
        "froslass_true_evolutions": froslass_true_evolutions,
        "stamps": stamps,
        "bosses": bosses,
        "lillies": lillies,
        "our_decisions": our_decisions,
        "our_multi_pick": our_multi_pick,
        "our_ends": our_ends,
        "our_overage_start": our_first,
        "our_overage_end": our_last,
        "our_overage_used": (
            round(our_first - our_last, 3)
            if our_first is not None and our_last is not None else None
        ),
        "opp_overage_used": (
            round(opp_first - opp_last, 3)
            if opp_first is not None and opp_last is not None else None
        ),
        "odd_status": json.dumps(odd_status) if odd_status else "",
        "steps": len(steps),
    }


def load_run(
    label: str, submission_id: int, run_dir: Path
) -> list[dict[str, Any]]:
    episodes = {
        row["episode_id"]: row
        for row in csv.DictReader(
            (run_dir / "episodes.csv").open(encoding="utf-8-sig")
        )
    }
    rows: list[dict[str, Any]] = []
    for episode_id, meta in episodes.items():
        if meta.get("state") != "COMPLETED":
            continue
        # Every submission is paired against itself once as Kaggle's
        # validation episode.  It carries no rating and is a mirror by
        # construction, so leaving it in inflates both the mirror cell
        # and the denominator of the overall win rate.
        if meta.get("episode_type") == "EPISODE_TYPE_VALIDATION":
            continue
        if meta.get("agent_0_submission_id") == str(submission_id):
            seat = 0
        elif meta.get("agent_1_submission_id") == str(submission_id):
            seat = 1
        else:
            continue
        path = (
            run_dir / "episodes" / episode_id / "replay" / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        row = walk_episode(replay, seat)
        if row is None:
            continue

        def number(key: str) -> float | None:
            text = meta.get(key, "")
            try:
                return float(text)
            except (TypeError, ValueError):
                return None

        row.update(
            {
                "version": label,
                "submission_id": submission_id,
                "episode_id": int(episode_id),
                "seat": seat,
                "create_time": meta.get("create_time", ""),
                "opponent_submission": meta.get(
                    f"agent_{1 - seat}_submission_id", ""
                ),
                "opponent_rating": number(f"agent_{1 - seat}_initial_score"),
                "our_rating_before": number(f"agent_{seat}_initial_score"),
                "our_rating_after": number(f"agent_{seat}_updated_score"),
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: item["create_time"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v27" / "version_games.csv",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to these version labels (repeatable).",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for label, submission_id, path in RUNS:
        if args.only and label not in args.only:
            continue
        run_dir = ROOT / RUN_ROOT / path
        if not run_dir.exists():
            print(f"{label}: MISSING {run_dir}")
            continue
        rows = load_run(label, submission_id, run_dir)
        wins = sum(row["won"] for row in rows)
        print(
            f"{label:6s} sub={submission_id}  games={len(rows):3d}  "
            f"{wins}-{len(rows) - wins}  "
            f"final={rows[-1]['our_rating_after'] if rows else None}"
        )
        all_rows.extend(rows)

    fields = [
        "version", "submission_id", "episode_id", "create_time", "seat",
        "opponent_submission", "opponent_rating", "our_rating_before",
        "our_rating_after", "won", "went_first", "opponent_family",
        "opponent_deck_hash", "exact_mirror", "wall_family", "turns",
        "our_turns", "our_prize_left", "opp_prize_left", "our_deck_left",
        "opp_deck_left", "our_bodies_left", "board_out", "deck_out",
        "first_ready_turn", "first_shadow_turn", "first_attack_turn",
        "opp_first_attack_turn", "own_first_shadow_turn",
        "own_first_ready_turn", "gate_violation", "shadow_attacks", "attacks",
        "grim_evolutions", "rare_candies", "adrena_brains", "froslass_actions",
        "froslass_true_evolutions", "stamps", "bosses", "lillies",
        "our_decisions", "our_multi_pick", "our_ends", "our_overage_start",
        "our_overage_end", "our_overage_used", "opp_overage_used",
        "odd_status", "steps",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n{len(all_rows)} games -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
