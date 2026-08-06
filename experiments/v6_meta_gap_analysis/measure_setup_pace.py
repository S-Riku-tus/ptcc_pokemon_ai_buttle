"""How fast each pilot gets Marnie's Grimmsnarl ex into play, and how fast we do.

matchup_shape_corpus.json found one metric that separates wins from losses in
*every* current matchup, including ones where the direction of everything else
flips: the turn Grimmsnarl ex first reaches play (mirror 4.83 vs 5.12, Alakazam
4.77 vs 5.78, Ogerpon 4.20 vs 5.57, Dragapult 5.96 vs 7.60) and whether it
arrives at all. Unlike a MAIN preference rate, this is a *pace* statistic, so it
can be compared pilot to pilot and against our own ladder games directly.

`turn` in a replay is a global player-turn counter, so seat parity shifts every
number by one; going first and going second are therefore reported separately.

Usage:
    python experiments/v6_meta_gap_analysis/measure_setup_pace.py \
        --out setup_pace.json
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
IMPIDIMP, MORGREM, GRIMMSNARL_EX = 646, 647, 648
SNORUNT, FROSLASS = 860, 104
DARK = 7
MAIN = 0


def ids(cards: list[dict]) -> set[int]:
    out = set()
    for card in cards:
        try:
            out.add(int(card.get("id", -1)))
        except (TypeError, ValueError):
            pass
    return out


def scan(job: tuple[str, int, str]) -> dict | None:
    path, seat, label = job
    try:
        head = extract_fast_header_from_file(path)
    except Exception:
        return None
    hashes = head.get("deck_hashes") or ["", ""]
    rewards = head.get("rewards") or [None, None]
    if len(hashes) < 2 or hashes[seat] != DECK_HASH:
        return None
    try:
        won = int(int(rewards[seat]) > int(rewards[1 - seat]))
    except (TypeError, ValueError):
        return None
    try:
        replay = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

    first_seen: dict[int, int] = {}
    first_player = -1
    own_turns: set[int] = set()
    energy_on_line: dict[int, int] = {}
    for index, step in enumerate(replay.get("steps") or []):
        if seat >= len(step):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        current = (record.get("observation") or {}).get("current") or {}
        players = current.get("players") or []
        your = int(current.get("yourIndex", seat))
        if len(players) < 2 or your >= len(players):
            continue
        me = players[your]
        turn = int(current.get("turn", -1))
        own_turns.add(turn)
        if first_player < 0 and int(current.get("firstPlayer", -1)) >= 0:
            first_player = int(current.get("firstPlayer", -1))
        in_play = mf._in_play(me)
        present = ids(in_play)
        for card_id in (IMPIDIMP, MORGREM, GRIMMSNARL_EX, SNORUNT, FROSLASS):
            if card_id in present and card_id not in first_seen:
                first_seen[card_id] = turn
        if GRIMMSNARL_EX in present and turn not in energy_on_line:
            attached = 0
            for card in in_play:
                if int(card.get("id", -1)) != GRIMMSNARL_EX:
                    continue
                for energy in card.get("energy") or []:
                    if isinstance(energy, dict):
                        attached += 1
            energy_on_line[turn] = attached

    if first_player < 0 or not own_turns:
        return None
    ex_turn = first_seen.get(GRIMMSNARL_EX)
    ordered = sorted(own_turns)
    # Our own nth turn, so the pace is comparable across seat parity.
    def nth(turn: int | None) -> int | None:
        if turn is None:
            return None
        return sum(1 for t in ordered if t <= turn)

    return {
        "label": label,
        "won": won,
        "went_first": int(first_player == seat),
        "ex_turn": ex_turn,
        "ex_own_turn": nth(ex_turn),
        "ex_ever": int(ex_turn is not None),
        "morgrem_own_turn": nth(first_seen.get(MORGREM)),
        "impidimp_own_turn": nth(first_seen.get(IMPIDIMP)),
        "froslass_own_turn": nth(first_seen.get(FROSLASS)),
        "energy_on_ex_first": (
            energy_on_line[min(energy_on_line)] if energy_on_line else None
        ),
        "own_turns": len(ordered),
    }


def corpus_jobs() -> list[tuple[str, int, str]]:
    seen: set[tuple[str, str]] = set()
    jobs: list[tuple[str, int, str]] = []
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
            jobs.append((str(path), int(row["seat_index"]), row["team_id"]))
    return jobs


def run_jobs() -> list[tuple[str, int, str]]:
    jobs: list[tuple[str, int, str]] = []
    for run_dir in sorted(RUNS.glob("2026080*")):
        if not run_dir.is_dir():
            continue
        seats = {
            row["episode_id"]: row["detected_submission_agent_index"]
            for row in csv.DictReader(
                open(run_dir / "manifest.csv", encoding="utf-8-sig")
            )
        }
        label = "ours:" + run_dir.name.split("_grimmsnarl_")[-1].split("_sub")[0]
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
                run_dir / "episodes" / episode / "replay"
                / f"episode_{episode}.json"
            )
            if path.exists():
                jobs.append((str(path), int(seat), label))
    return jobs


def summarise(rows: list[dict]) -> dict:
    def mean(key: str, subset: list[dict]) -> float | None:
        values = [row[key] for row in subset if row.get(key) is not None]
        return round(statistics.fmean(values), 3) if values else None

    def share(subset: list[dict], limit: int) -> float | None:
        if not subset:
            return None
        hit = sum(
            1 for row in subset
            if row["ex_own_turn"] is not None and row["ex_own_turn"] <= limit
        )
        return round(hit / len(subset), 4)

    out = {
        "games": len(rows),
        "win_rate": round(statistics.fmean([r["won"] for r in rows]), 4),
        "went_first_share": round(
            statistics.fmean([r["went_first"] for r in rows]), 4
        ),
        "ex_ever": mean("ex_ever", rows),
        "ex_own_turn": mean("ex_own_turn", rows),
        "ex_by_own_turn_2": share(rows, 2),
        "ex_by_own_turn_3": share(rows, 3),
        "morgrem_own_turn": mean("morgrem_own_turn", rows),
        "froslass_own_turn": mean("froslass_own_turn", rows),
        "energy_on_ex_first": mean("energy_on_ex_first", rows),
    }
    for seat_label, subset in (
        ("first", [r for r in rows if r["went_first"]]),
        ("second", [r for r in rows if not r["went_first"]]),
    ):
        out[f"{seat_label}_games"] = len(subset)
        out[f"{seat_label}_win_rate"] = (
            round(statistics.fmean([r["won"] for r in subset]), 4)
            if subset else None
        )
        out[f"{seat_label}_ex_own_turn"] = mean("ex_own_turn", subset)
        out[f"{seat_label}_ex_by_own_turn_2"] = share(subset, 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    jobs = corpus_jobs() + run_jobs()
    print(f"replays={len(jobs)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for row in pool.map(scan, jobs, chunksize=16) if row]
    print(f"parsed={len(rows)}", flush=True)

    scores: dict[str, float] = {}
    for row in csv.DictReader(
        open(CORPUS / "indexes" / "replay_index.csv", encoding="utf-8-sig")
    ):
        scores.setdefault(row["team_id"], float(row["submission_score"]))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    grouped["field:all_pilots"] = [
        row for row in rows if not row["label"].startswith("ours:")
    ]

    report = {}
    for label, group in grouped.items():
        summary = summarise(group)
        summary["corpus_submission_score"] = scores.get(label)
        report[label] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    keys = (
        "games", "corpus_submission_score", "ex_own_turn", "ex_by_own_turn_2",
        "ex_by_own_turn_3", "ex_ever", "first_ex_own_turn",
        "second_ex_own_turn", "morgrem_own_turn", "energy_on_ex_first",
        "win_rate",
    )
    print(f"{'label':<22}" + "".join(f"{key[:12]:>13}" for key in keys))
    ordered = sorted(
        report.items(),
        key=lambda kv: (
            kv[0].startswith("ours:"), -(kv[1]["corpus_submission_score"] or 0)
        ),
    )
    for label, row in ordered:
        cells = "".join(
            f"{row[key]:>13.3f}" if isinstance(row.get(key), (int, float))
            else " " * 13
            for key in keys
        )
        print(f"{label:<22}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
