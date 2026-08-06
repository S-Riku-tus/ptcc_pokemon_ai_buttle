"""What losing looks like in the matchups that populate the current top 20.

matchup_gap.json says *which* opponents beat this 60-card list. This says *how*,
from the Grimmsnarl seat, by comparing the same measurements in that
matchup's wins and losses:

* hand size at the end of our own turn - the hand the opponent attacks into.
  a7ee29914c1dce64 (ranks 3, 4, 9, 14, 19) plays Mega Froslass ex, whose
  Resentful Refrain costs one {W} and does 50 damage per card in our hand, so
  a 7-card hand kills a 320 HP Grimmsnarl ex with no weakness involved;
* the turn Grimmsnarl ex first reaches play, and whether it ever does;
* the prize race: prizes each side had left at the end;
* Boss / Petrel / Unfair Stamp plays per game.

Usage:
    python experiments/v6_meta_gap_analysis/measure_matchup_shape.py \
        --opponent a7ee29914c1dce64 --opponent 0dede7cb8026e473 \
        --out matchup_shape.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v5"))

import ml_features as mf  # noqa: E402

from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402

DECK_HASH = "9714ab5c3996f6cc"
CORPUS = ROOT / "data" / "kaggle_grimmsnarl_top50"
RUNS = ROOT / "data" / "runs" / "grimmsnarl"
GRIMMSNARL_EX = 648
STAMP = 1080
BOSS = 1182
PETREL = 1219
MAIN = 0


def in_play_ids(player: dict) -> set[int]:
    ids = set()
    for card in mf._in_play(player):
        try:
            ids.add(int(card.get("id", -1)))
        except (TypeError, ValueError):
            pass
    return ids


def scan(job: tuple[str, int, tuple[str, ...]]) -> dict | None:
    """One game, measured from the Grimmsnarl seat."""
    path, seat, wanted = job
    try:
        head = extract_fast_header_from_file(path)
    except Exception:
        return None
    hashes = head.get("deck_hashes") or ["", ""]
    rewards = head.get("rewards") or [None, None]
    if len(hashes) < 2 or hashes[seat] != DECK_HASH:
        return None
    opponent_deck = hashes[1 - seat]
    if wanted and opponent_deck not in wanted:
        return None
    try:
        won = int(int(rewards[seat]) > int(rewards[1 - seat]))
    except (TypeError, ValueError):
        return None
    try:
        replay = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

    steps = replay.get("steps") or []
    hand_by_turn: dict[int, int] = {}
    opp_prize_by_turn: dict[int, int] = {}
    my_prize_by_turn: dict[int, int] = {}
    ex_first_turn: int | None = None
    plays: Counter = Counter()
    first_player = -1
    last_turn = -1

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
        your = int(current.get("yourIndex", seat))
        if len(players) < 2 or your >= len(players):
            continue
        me, opponent = players[your], players[1 - your]
        turn = int(current.get("turn", -1))
        last_turn = max(last_turn, turn)
        if first_player < 0 and int(current.get("firstPlayer", -1)) >= 0:
            first_player = int(current.get("firstPlayer", -1))
        opp_prize_by_turn[turn] = len(opponent.get("prize") or [])
        my_prize_by_turn[turn] = len(me.get("prize") or [])
        if ex_first_turn is None and GRIMMSNARL_EX in in_play_ids(me):
            ex_first_turn = turn
        if not select or int(select.get("context", -1)) != MAIN:
            continue
        # The last MAIN hand size in a turn is the hand the opponent sees.
        hand_by_turn[turn] = len(me.get("hand") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        if not isinstance(action, list) or not action:
            continue
        options = list(select.get("option") or [])
        slot = action[0]
        if not isinstance(slot, int) or not 0 <= slot < len(options):
            continue
        card = mf.candidate_card(current, options[slot], select) or {}
        try:
            picked = int(card.get("id", -1))
        except (TypeError, ValueError):
            picked = -1
        if picked == STAMP:
            plays["stamp"] += 1
        elif picked == BOSS:
            plays["boss"] += 1
        elif picked == PETREL:
            plays["petrel"] += 1

    if not hand_by_turn:
        return None
    hands = [hand_by_turn[turn] for turn in sorted(hand_by_turn)]
    turns = sorted(opp_prize_by_turn)
    # Turns after which the opponent took a prize, i.e. we lost a Pokemon.
    hand_when_ko: list[int] = []
    for earlier, later in zip(turns, turns[1:]):
        if opp_prize_by_turn[later] < opp_prize_by_turn[earlier]:
            if earlier in hand_by_turn:
                hand_when_ko.append(hand_by_turn[earlier])
    return {
        "opponent_deck": opponent_deck,
        "won": won,
        "turns": last_turn,
        "went_first": int(first_player == seat) if first_player >= 0 else None,
        "my_prizes_left": my_prize_by_turn[turns[-1]] if turns else None,
        "opp_prizes_left": opp_prize_by_turn[turns[-1]] if turns else None,
        "hand_mean": round(statistics.fmean(hands), 3),
        "hand_max": max(hands),
        "hand_ge6_share": round(sum(1 for h in hands if h >= 6) / len(hands), 4),
        "hand_ge7_share": round(sum(1 for h in hands if h >= 7) / len(hands), 4),
        "hand_when_ko_mean": (
            round(statistics.fmean(hand_when_ko), 3) if hand_when_ko else None
        ),
        "grimmsnarl_ex_turn": ex_first_turn,
        "grimmsnarl_ex_ever": int(ex_first_turn is not None),
        "stamp_plays": plays["stamp"],
        "boss_plays": plays["boss"],
        "petrel_plays": plays["petrel"],
    }


def corpus_jobs(wanted: tuple[str, ...]) -> list[tuple[str, int, tuple]]:
    seen: set[tuple[str, str]] = set()
    jobs: list[tuple[str, int, tuple]] = []
    for row in csv.DictReader(
        open(CORPUS / "indexes" / "replay_index.csv", encoding="utf-8-sig")
    ):
        if row["deck_hash"] != DECK_HASH:
            continue
        if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
            continue
        key = (row["episode_id"], row["seat_index"])
        if key in seen:
            continue
        seen.add(key)
        path = CORPUS / Path(row["replay_path"].replace(chr(92), "/"))
        if path.exists():
            jobs.append((str(path), int(row["seat_index"]), wanted))
    return jobs


def run_jobs(run_dir: Path, wanted: tuple[str, ...]) -> list[tuple[str, int, tuple]]:
    seats = {
        row["episode_id"]: row["detected_submission_agent_index"]
        for row in csv.DictReader(
            open(run_dir / "manifest.csv", encoding="utf-8-sig")
        )
    }
    jobs: list[tuple[str, int, tuple]] = []
    for row in csv.DictReader(
        open(run_dir / "episodes.csv", encoding="utf-8-sig")
    ):
        episode = row["episode_id"]
        if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
            continue
        seat = seats.get(episode, "")
        if seat not in ("0", "1"):
            continue
        path = (
            run_dir / "episodes" / episode / "replay" / f"episode_{episode}.json"
        )
        if path.exists():
            jobs.append((str(path), int(seat), wanted))
    return jobs


NUMERIC = (
    "turns", "my_prizes_left", "opp_prizes_left", "hand_mean", "hand_max",
    "hand_ge6_share", "hand_ge7_share", "hand_when_ko_mean",
    "grimmsnarl_ex_turn", "grimmsnarl_ex_ever", "stamp_plays", "boss_plays",
    "petrel_plays", "went_first",
)


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"games": len(rows)}
    for key in NUMERIC:
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = round(statistics.fmean(values), 3) if values else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument(
        "--our-runs", action="store_true",
        help="Measure our own ladder runs instead of the corpus pilots.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    wanted = tuple(args.opponent)
    if args.our_runs:
        jobs = [
            job for run in sorted(RUNS.glob("2026080*")) if run.is_dir()
            for job in run_jobs(run, wanted)
        ]
    else:
        jobs = corpus_jobs(wanted)
    print(f"replays={len(jobs)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for row in pool.map(scan, jobs, chunksize=16) if row]
    print(f"parsed={len(rows)}", flush=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["opponent_deck"]].append(row)

    report = {
        "deck_hash": DECK_HASH,
        "source": "runs" if args.our_runs else "corpus",
    }
    for opponent, group in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        report[opponent] = {
            "all": aggregate(group),
            "wins": aggregate([row for row in group if row["won"]]),
            "losses": aggregate([row for row in group if not row["won"]]),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    keys = (
        "games", "hand_mean", "hand_ge7_share", "hand_when_ko_mean",
        "grimmsnarl_ex_turn", "grimmsnarl_ex_ever", "turns",
        "opp_prizes_left", "stamp_plays", "boss_plays",
    )
    header = f"{'opponent':<17}{'split':<8}"
    print(header + "".join(f"{key[:11]:>12}" for key in keys))
    for opponent, row in report.items():
        if not isinstance(row, dict) or "all" not in row:
            continue
        for split in ("wins", "losses"):
            cells = "".join(
                f"{row[split][key]:>12.3f}"
                if row[split].get(key) is not None else " " * 12
                for key in keys
            )
            print(f"{opponent:<17}{split:<8}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
