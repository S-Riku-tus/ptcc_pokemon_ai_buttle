"""Where v28 sits in the live field, and how it plays against same-deck peers.

Three joins the per-game table cannot make on its own:

1. *the board* - every opponent submission is joined to the current public
   leaderboard, so "opponent rated 990 at pairing" can be checked against
   where that submission actually ended up, and our own rank is read from the
   same file rather than from a screenshot;
2. *the field* - unique opponents grouped by archetype family with their
   current scores, which says which archetypes carry the top of the board and
   which of them we actually beat;
3. *the peers* - in an exact mirror both seats play our 60, so the opponent's
   side of the replay is a same-deck pilot playing the same game.  Walking
   both seats gives a controlled behavioural comparison against pilots rated
   above and below us, which no aggregate leaderboard number can give.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_grimmsnarl_v27_vs_champions as champ  # noqa: E402
import build_grimmsnarl_version_games as builder  # noqa: E402

champ.GROUPS["v28"] = ("v28",)

OUR_SUBMISSIONS = {
    55479857: "v22_a", 55483874: "v22_b", 55486680: "v22_c", 55486691: "v22_d",
    55485982: "v23", 55496021: "v24_a", 55496665: "v24_b", 55507909: "v25_a",
    55517142: "v25_b", 55520389: "v26", 55521760: "v27", 55526859: "v28",
}

PEER_METRICS = (
    "attacks", "shadow_attacks", "adrena_brains", "grim_evolutions",
    "rare_candies", "froslass_true_evolutions", "stamps", "bosses", "lillies",
    "our_multi_pick", "our_ends",
)


def load_board(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    teams = {team["teamId"]: team for team in data["teams"]}
    rows = data["publicLeaderboard"]
    by_submission = {}
    for row in rows:
        by_submission[int(row["submissionId"])] = {
            "rank": int(row["rank"]),
            "score": float(row["displayScore"]),
            "team_id": int(row["teamId"]),
            "team": teams.get(row["teamId"], {}).get("teamName", ""),
        }
    return by_submission, rows


def per_turn(rows: Sequence[dict[str, Any]], column: str) -> float | None:
    total = sum(r[column] for r in rows if r.get(column) is not None)
    turns = sum(r["our_turns"] for r in rows if r.get("our_turns"))
    return total / turns if turns else None


def peer_table(
    label: str, ours: Sequence[dict[str, Any]], theirs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    print(f"\n{label}  (n={len(ours)} games)")
    print(f"  {'per own turn':<26}{'us':>10}{'peer':>10}{'delta':>10}")
    result: dict[str, Any] = {"games": len(ours)}
    for column in PEER_METRICS:
        mine = per_turn(ours, column)
        peer = per_turn(theirs, column)
        delta = None if mine is None or peer is None else mine - peer
        print(
            f"  {column:<26}{champ.fmt(mine, 3):>10}{champ.fmt(peer, 3):>10}"
            f"{champ.fmt(delta, 3):>10}"
        )
        result[column] = {"us": mine, "peer": peer}
    for column in ("own_first_shadow_turn", "own_first_ready_turn", "our_turns",
                   "our_prize_left", "our_bodies_left", "our_decisions"):
        mine = champ.mean(r[column] for r in ours)
        peer = champ.mean(r[column] for r in theirs)
        delta = None if mine is None or peer is None else mine - peer
        print(
            f"  {column + ' (mean)':<26}{champ.fmt(mine, 2):>10}"
            f"{champ.fmt(peer, 2):>10}{champ.fmt(delta, 2):>10}"
        )
        result[column] = {"us": mine, "peer": peer}
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/version_games.csv",
    )
    parser.add_argument(
        "--leaderboard", type=Path,
        default=ROOT / "data/kaggle_top100/latest/raw/api/leaderboard_full.json",
    )
    parser.add_argument("--target", default="v28")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/field.json",
    )
    args = parser.parse_args()

    rows = champ.load(args.games)
    target = [r for r in rows if r["group"] == args.target]
    board, board_rows = load_board(args.leaderboard)
    report: dict[str, Any] = {}

    champ.section("1. The board right now")
    manifest = json.loads(
        (args.leaderboard.parents[2] / "manifest.json").read_text(encoding="utf-8")
    )
    print(f"snapshot: {manifest['competition']['name']} "
          f"retrieved {manifest['retrievedAtJst']}, {len(board_rows)} teams")
    for rank in (1, 3, 5, 10, 20, 30, 50, 100, 150, 200, 300, 500, 1000):
        if rank <= len(board_rows):
            print(f"  rank {rank:>5}: {board_rows[rank - 1]['displayScore']:>8}")
    print("\nour submissions on the board:")
    ours_on_board = []
    for submission, label in OUR_SUBMISSIONS.items():
        entry = board.get(submission)
        if entry:
            print(f"  {label:<6} sub {submission}  rank {entry['rank']:>5} "
                  f"score {entry['score']:>8}  team {entry['team']}")
            ours_on_board.append({"label": label, **entry})
    report["board"] = {
        "retrieved": manifest["retrievedAtJst"],
        "teams": len(board_rows),
        "ours": ours_on_board,
    }

    champ.section(f"2. Who {args.target} played, and where those opponents stand now")
    print(f"{'ep':>9} {'res':<3} {'seat':<6} {'pairing':>8} {'now':>8} "
          f"{'rank':>6}  {'family':<24} team")
    detail = []
    for row in target:
        submission = int(row["opponent_submission"] or 0)
        entry = board.get(submission)
        print(
            f"{row['episode_id']:>9} {'W' if row['won'] else 'L':<3} "
            f"{row['went_first']:<6} {row['opponent_rating'] or 0:8.1f} "
            f"{(entry['score'] if entry else float('nan')):8.1f} "
            f"{(entry['rank'] if entry else 0):6d}  {row['opponent_family']:<24} "
            f"{entry['team'] if entry else '(not on board)'}"
        )
        detail.append({
            "episode_id": row["episode_id"], "won": row["won"],
            "pairing_rating": row["opponent_rating"],
            "current_score": entry["score"] if entry else None,
            "current_rank": entry["rank"] if entry else None,
            "family": row["opponent_family"],
            "team": entry["team"] if entry else None,
        })
    report["target_opponents"] = detail

    covered = [d for d in detail if d["current_rank"]]
    print(f"\n{len(covered)}/{len(detail)} opponents still active on the board")
    for cutoff in (100, 200, 500, 1000):
        subset = [d for d in covered if d["current_rank"] <= cutoff]
        if subset:
            wins = sum(d["won"] for d in subset)
            print(f"  vs teams currently ranked <= {cutoff:>4}: "
                  f"{wins}-{len(subset) - wins} ({wins / len(subset):.3f})")

    champ.section("3. The field by archetype: who is strong, and do we beat them")
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        submission = row["opponent_submission"]
        if not submission or int(submission) in OUR_SUBMISSIONS:
            continue
        entry = board.get(int(submission))
        current = unique.setdefault(submission, {
            "family": row["opponent_family"],
            "score": entry["score"] if entry else None,
            "rank": entry["rank"] if entry else None,
            "games": 0,
        })
        current["games"] += 1
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in unique.values():
        by_family[value["family"]].append(value)
    print(f"{'family':<26}{'opps':>6}{'on board':>9}{'median now':>11}"
          f"{'top rank':>9}   our record (all versions / v28)")
    family_rows = []
    for family, values in sorted(
        by_family.items(), key=lambda item: -len(item[1])
    ):
        scored = sorted(v["score"] for v in values if v["score"] is not None)
        ranks = [v["rank"] for v in values if v["rank"] is not None]
        median = scored[len(scored) // 2] if scored else None
        allv = [r for r in rows if r["opponent_family"] == family]
        tgt = [r for r in target if r["opponent_family"] == family]
        allw = sum(r["won"] for r in allv)
        tw = sum(r["won"] for r in tgt)
        print(
            f"{family:<26}{len(values):>6}{len(scored):>9}"
            f"{(f'{median:.1f}' if median else '-'):>11}"
            f"{(min(ranks) if ranks else 0):>9}   "
            f"{allw}-{len(allv) - allw}"
            + (f" / {tw}-{len(tgt) - tw}" if tgt else "")
        )
        family_rows.append({
            "family": family, "opponents": len(values),
            "median_current_score": median,
            "best_rank": min(ranks) if ranks else None,
            "all_versions": [allw, len(allv)],
            "target": [tw, len(tgt)],
        })
    report["families"] = family_rows

    print("\nthe same table weighted by how strong those opponents are now:")
    strong = {
        family: [v for v in values if v["score"] is not None and v["score"] >= 1000]
        for family, values in by_family.items()
    }
    ranked = sorted(strong.items(), key=lambda item: -len(item[1]))
    for family, values in ranked:
        if values:
            print(f"  {family:<26} {len(values):>3} opponents currently >= 1000")

    champ.section("4. Same-deck peers: both seats of every exact mirror")
    mirror_rows = [r for r in rows if r["exact_mirror"]]
    print(f"{len(mirror_rows)} mirror games across all versions, "
          f"{len([r for r in mirror_rows if r['group'] == args.target])} for {args.target}")
    peers: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing = 0
    for row in mirror_rows:
        run_dir = None
        for label, submission, path in builder.RUNS:
            if label == row["version"]:
                run_dir = ROOT / builder.RUN_ROOT / path
                break
        if run_dir is None:
            continue
        replay_path = (
            run_dir / "episodes" / str(row["episode_id"]) / "replay"
            / f"episode_{row['episode_id']}.json"
        )
        if not replay_path.exists():
            missing += 1
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        opponent = builder.walk_episode(replay, 1 - row["seat"])
        if opponent is None:
            missing += 1
            continue
        peers.append((row, opponent))
    print(f"walked {len(peers)} mirror games from both seats ({missing} skipped)")

    peer_report: dict[str, Any] = {}
    for label, predicate in (
        (f"{args.target}: all mirrors", lambda r: r["group"] == args.target),
        (
            f"{args.target}: peer rated above us at pairing",
            lambda r: r["group"] == args.target
            and (r["opponent_rating"] or 0) > (r["our_rating_before"] or 0),
        ),
        (
            f"{args.target}: peer rated below us at pairing",
            lambda r: r["group"] == args.target
            and (r["opponent_rating"] or 0) <= (r["our_rating_before"] or 0),
        ),
        ("v22: all mirrors", lambda r: r["group"] == "v22"),
        (
            "all versions: peer rated 950+",
            lambda r: (r["opponent_rating"] or 0) >= 950,
        ),
    ):
        selected = [(o, p) for o, p in peers if predicate(o)]
        if not selected:
            continue
        peer_report[label] = peer_table(
            label, [o for o, _ in selected], [p for _, p in selected]
        )
        wins = sum(o["won"] for o, _ in selected)
        print(f"  record {wins}-{len(selected) - wins}")
        peer_report[label]["record"] = [wins, len(selected)]
    report["peers"] = peer_report

    champ.section("5. Mirror peers by their current board position")
    for row in [r for r in mirror_rows if r["group"] == args.target]:
        entry = board.get(int(row["opponent_submission"] or 0))
        print(
            f"  ep {row['episode_id']} {'W' if row['won'] else 'L'}  "
            f"pairing {row['opponent_rating'] or 0:7.1f}  "
            f"now {(entry['score'] if entry else float('nan')):7.1f} "
            f"rank {(entry['rank'] if entry else 0):>5}  "
            f"{entry['team'] if entry else '(not on board)'}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
