"""Build the per-episode ladder table across every run under data/runs/grimmsnarl/.

One row per (run, episode).  Win/loss is read from the replay's ``rewards``
array and cross-checked against the final prize counts recovered from the last
board state; the cached episodes.csv is used only for the opponent's
``initialScore`` (a rating, not an outcome) and for the seat mapping, and the
seat mapping is itself re-verified against the 60-card deck list in step 1.

Writes episodes_all.csv + runs.json into this directory.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
RUNS_DIR = ROOT / "data" / "runs" / "grimmsnarl"
OUT = Path(__file__).resolve().parent

# user-reported final ladder rating, from the assignment brief / run_meta notes
REPORTED = {
    55185513: ("v1", 1, 871.0),
    55205556: ("v2", 2, 967.4),
    55216787: ("v3a", 3, 996.6),
    55217233: ("v3b", 4, 907.6),
    55253296: ("v4", 5, 1031.2),
    55275464: ("v4.5", 6, 979.1),
    55275642: ("v5", 7, 963.6),
    55290882: ("v6", 8, 996.6),
    55302846: ("v7", 9, 943.3),
    55317804: ("v8", 10, 1035.8),
    55325029: ("v9", 11, None),
    55346539: ("v11a", 12, None),
    55346548: ("v11b", 13, None),
    55353978: ("v11", 14, 950.3),
    55373676: ("v12a", 15, 914.8),
    55374240: ("v12b", 16, 894.6),
    55380882: ("v13a", 17, 942.5),
    55380958: ("v13b", 18, 964.8),
    55395386: ("v14", 19, 927.6),
    55404196: ("v15", 20, 1007.7),
    55409394: ("v15b", 21, 862.0),
    55422280: ("v16", 22, 955.8),
    55423572: ("v17", 23, 896.8),
    55428191: ("v18", 24, 779.5),
    55428196: ("v19a", 25, 978.3),
    55445763: ("v19b", 26, 904.6),
    55445769: ("v20", 27, 982.0),
    55456713: ("v21", 28, 948.2),
}
# version-index for the monotone-trend test: the code lineage number, so the
# a/b pairs of one version share an index.
LINEAGE = {
    "v1": 1, "v2": 2, "v3a": 3, "v3b": 3, "v4": 4, "v4.5": 4.5, "v5": 5,
    "v6": 6, "v7": 7, "v8": 8, "v9": 9, "v11a": 11, "v11b": 11, "v11": 11,
    "v12a": 12, "v12b": 12, "v13a": 13, "v13b": 13, "v14": 14, "v15": 15,
    "v15b": 15, "v16": 16, "v17": 17, "v18": 18, "v19a": 19, "v19b": 19,
    "v20": 20, "v21": 21,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def decks(steps: list[Any]) -> list[list[int] | None]:
    out: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                out[seat] = [int(v) for v in action]
    return out


def late_current(steps: list[Any]) -> dict[str, Any] | None:
    best, best_key = None, (-1, -1)
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


def prize_left(player: dict[str, Any]) -> int | None:
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    if isinstance(prize, int):
        return prize
    return None


def main() -> int:
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sub_id = int(meta["submission_id"])
        else:
            # v9 was fetched without a run_meta.json; the id is in the dir name
            tail = run_dir.name.rsplit("sub", 1)[-1].lstrip("_")
            if not tail.isdigit():
                print(f"skip (no meta, no id): {run_dir.name}", file=sys.stderr)
                continue
            sub_id = int(tail)
        label, order, rating = REPORTED.get(sub_id, (run_dir.name, 999, None))

        csv_rows = read_csv(run_dir / "episodes.csv")
        by_episode = {int(r["episode_id"]): r for r in csv_rows}

        # rating trajectory straight out of episodes.csv, for cross-check
        traj: list[tuple[str, float]] = []
        for r in csv_rows:
            seat = 0 if int(r["agent_0_submission_id"]) == sub_id else 1
            try:
                traj.append((r["end_time"], float(r[f"agent_{seat}_updated_score"])))
            except (ValueError, KeyError):
                pass
        traj.sort()

        n_replay = 0
        seat_mismatch = 0
        reward_prize_disagree = 0
        run_rows: list[dict[str, Any]] = []
        ep_dir = run_dir / "episodes"
        if not ep_dir.exists():
            continue
        for episode in sorted(ep_dir.iterdir()):
            replay_files = list((episode / "replay").glob("*.json")) if (episode / "replay").exists() else []
            if not replay_files:
                continue
            replay = json.loads(replay_files[0].read_text(encoding="utf-8"))
            steps = replay.get("steps") or []
            our_decks = decks(steps)
            hashes = [deck_hash(d) if d else None for d in our_decks]
            episode_id = int(episode.name)
            meta_row = by_episode.get(episode_id)

            csv_seat = None
            if meta_row is not None:
                if int(meta_row["agent_0_submission_id"]) == sub_id:
                    csv_seat = 0
                elif int(meta_row["agent_1_submission_id"]) == sub_id:
                    csv_seat = 1

            # seat from the deck hash, when it is unambiguous
            deck_seats = [s for s in (0, 1) if hashes[s] == OUR_DECK_HASH]
            seat = csv_seat
            if seat is None:
                if len(deck_seats) == 1:
                    seat = deck_seats[0]
                else:
                    continue
            elif len(deck_seats) == 1 and deck_seats[0] != seat:
                seat_mismatch += 1

            rewards = replay.get("rewards") or [None, None]
            ours, theirs = rewards[seat], rewards[1 - seat]
            if ours is None or theirs is None:
                continue
            won = int(ours > theirs)
            drew = int(ours == theirs)

            current = late_current(steps)
            our_prize = their_prize = None
            went_first = None
            if current:
                players = current.get("players") or []
                if len(players) >= 2:
                    our_prize = prize_left(players[seat])
                    their_prize = prize_left(players[1 - seat])
                first = int(current.get("firstPlayer", -1))
                went_first = (first == seat) if first >= 0 else None
            if our_prize is not None and their_prize is not None:
                prize_won = None
                if their_prize == 0 and our_prize > 0:
                    prize_won = 1
                elif our_prize == 0 and their_prize > 0:
                    prize_won = 0
                if prize_won is not None and prize_won != won:
                    reward_prize_disagree += 1

            opp_deck = our_decks[1 - seat]
            opp_sub = None
            opp_initial = None
            our_initial = None
            our_updated = None
            if meta_row is not None:
                opp_sub = int(meta_row[f"agent_{1 - seat}_submission_id"])
                try:
                    opp_initial = float(meta_row[f"agent_{1 - seat}_initial_score"])
                except (ValueError, KeyError):
                    opp_initial = None
                try:
                    our_initial = float(meta_row[f"agent_{seat}_initial_score"])
                    our_updated = float(meta_row[f"agent_{seat}_updated_score"])
                except (ValueError, KeyError):
                    pass

            run_rows.append({
                "run": run_dir.name, "label": label, "order": order,
                "lineage": LINEAGE.get(label, order), "sub_id": sub_id,
                "reported_rating": rating, "episode_id": episode_id,
                "seat": seat, "won": won, "drew": drew,
                "reward_ours": ours, "reward_theirs": theirs,
                "our_prize_left": our_prize, "their_prize_left": their_prize,
                "went_first": went_first, "opp_sub": opp_sub,
                "opp_initial": opp_initial, "our_initial": our_initial,
                "our_updated": our_updated,
                "opp_family": family(opp_deck) if opp_deck else "unknown",
                "opp_deck_hash": hashes[1 - seat],
                "end_time": meta_row["end_time"] if meta_row else None,
            })
            n_replay += 1

        rows.extend(run_rows)
        wins = sum(r["won"] for r in run_rows)
        runs.append({
            "label": label, "order": order, "lineage": LINEAGE.get(label, order),
            "sub_id": sub_id, "run": run_dir.name,
            "reported_rating": rating,
            "csv_episodes": len(csv_rows), "replay_episodes": n_replay,
            "wins": wins, "losses": n_replay - wins,
            "draws": sum(r["drew"] for r in run_rows),
            "seat_mismatch": seat_mismatch,
            "reward_prize_disagree": reward_prize_disagree,
            "last_updated_score": traj[-1][1] if traj else None,
            "first_initial_score": (
                min((r["our_initial"] for r in run_rows if r["our_initial"] is not None),
                    default=None)
            ),
        })

    runs.sort(key=lambda r: r["order"])
    with (OUT / "episodes_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    print(f"{'label':6} {'sub':>9} {'rep':>7} {'csvN':>5} {'repN':>5} "
          f"{'W':>3} {'L':>3} {'D':>2} {'seatX':>5} {'rzX':>4} {'lastUpd':>8}")
    for r in runs:
        print(f"{r['label']:6} {r['sub_id']:>9} "
              f"{(r['reported_rating'] if r['reported_rating'] is not None else float('nan')):>7.1f} "
              f"{r['csv_episodes']:>5} {r['replay_episodes']:>5} {r['wins']:>3} "
              f"{r['losses']:>3} {r['draws']:>2} {r['seat_mismatch']:>5} "
              f"{r['reward_prize_disagree']:>4} "
              f"{(r['last_updated_score'] or float('nan')):>8.1f}")
    print(f"total episode rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
