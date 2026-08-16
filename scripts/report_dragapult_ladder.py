"""Per-episode ladder trajectory for one of our own submissions.

The displayed leaderboard rating is a single number at one instant.  What
decides whether a version is better than another is the sequence: who it was
paired against, at what rating, and what the result did to ours.  The
EpisodeService returns both agents' initial and updated ratings per episode, so
the whole trajectory is reconstructable without joining a leaderboard snapshot.

Also prints the Elo fixed point (opponent mean + Elo(win rate)), which is where
a rating converges to, and the opponent-rating buckets that are the only honest
way to compare two versions that met different fields.

Usage:
  python scripts/report_dragapult_ladder.py \
      --run data/submissions/submission_55550682_dragapult_v2 --label v2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def elo(win_rate: float) -> float:
    win_rate = min(max(win_rate, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / win_rate - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    summaries = []
    for position, run in enumerate(args.run):
        label = args.label[position] if position < len(args.label) else run.name
        episodes = {
            row["episode_id"]: row
            for row in csv.DictReader(
                (run / "episodes.csv").read_text(encoding="utf-8-sig").splitlines()
            )
        }
        manifest = {
            row["episode_id"]: row
            for row in csv.DictReader(
                (run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
            )
        }
        games = []
        for episode_id in sorted(episodes, key=int):
            row = episodes[episode_id]
            seat = manifest.get(episode_id, {}).get(
                "detected_submission_agent_index", ""
            )
            if seat not in ("0", "1") or row["state"] != "COMPLETED":
                continue
            other = "1" if seat == "0" else "0"

            def score(agent: str, field: str) -> float | None:
                raw = row[f"agent_{agent}_{field}_score"]
                return float(raw) if raw else None

            ours_before = score(seat, "initial")
            ours_after = score(seat, "updated")
            theirs_before = score(other, "initial")
            if ours_before is None or ours_after is None:
                continue
            delta = ours_after - ours_before
            games.append({
                "episode_id": int(episode_id),
                "end_time": row["end_time"][:19],
                "seat": int(seat),
                "ours_before": ours_before,
                "ours_after": ours_after,
                "opponent_before": theirs_before,
                "delta": delta,
                # A win always raises the rating and a loss always lowers it;
                # the replay's own reward field is not in episodes.csv.
                "result": "W" if delta > 0 else "L" if delta < 0 else "D",
            })

        wins = sum(1 for game in games if game["result"] == "W")
        losses = sum(1 for game in games if game["result"] == "L")
        played = wins + losses
        opponents = [
            game["opponent_before"] for game in games
            if game["opponent_before"] is not None
        ]
        opponent_mean = sum(opponents) / len(opponents) if opponents else float("nan")
        win_rate = wins / played if played else float("nan")

        print(f"\n=== {label}  ({run.name}) ===")
        print(f"{'episode':>10} {'end':<19} {'seat':>4} {'opp':>7} "
              f"{'ours':>7} {'->':>7} {'d':>7}  r")
        for game in games:
            opponent = (
                f"{game['opponent_before']:.1f}"
                if game["opponent_before"] is not None else "-"
            )
            print(f"{game['episode_id']:>10} {game['end_time']:<19} "
                  f"{game['seat']:>4} {opponent:>7} {game['ours_before']:>7.1f} "
                  f"{game['ours_after']:>7.1f} {game['delta']:>+7.1f}  "
                  f"{game['result']}")

        print(f"record {wins}-{losses}  win rate {win_rate:.4f}")
        print(f"final displayed rating {games[-1]['ours_after']:.1f}"
              if games else "no completed games")
        print(f"opponent mean initial {opponent_mean:.1f}")
        print(f"Elo fixed point {opponent_mean + elo(win_rate):.1f}")

        buckets = [(0, 400), (400, 500), (500, 600), (600, 700),
                   (700, 800), (800, 10000)]
        print(f"\n{'opponent band':>16} {'n':>4} {'W':>3} {'L':>3} {'wr':>7}")
        by_bucket = []
        for low, high in buckets:
            inside = [
                game for game in games
                if game["opponent_before"] is not None
                and low <= game["opponent_before"] < high
            ]
            if not inside:
                continue
            bucket_wins = sum(1 for game in inside if game["result"] == "W")
            print(f"{f'{low}-{high}':>16} {len(inside):>4} {bucket_wins:>3} "
                  f"{len(inside) - bucket_wins:>3} "
                  f"{bucket_wins / len(inside):>7.3f}")
            by_bucket.append({
                "low": low, "high": high, "n": len(inside),
                "wins": bucket_wins,
                "win_rate": round(bucket_wins / len(inside), 4),
            })

        # Halves, to see whether the later games look like the earlier ones.
        if played >= 8:
            half = len(games) // 2
            for name, block in (("first half", games[:half]),
                                ("second half", games[half:])):
                block_wins = sum(1 for game in block if game["result"] == "W")
                block_opponents = [
                    game["opponent_before"] for game in block
                    if game["opponent_before"] is not None
                ]
                mean = (sum(block_opponents) / len(block_opponents)
                        if block_opponents else float("nan"))
                print(f"{name:>16}: {block_wins}-{len(block) - block_wins}"
                      f"  opp mean {mean:.1f}")

        summaries.append({
            "label": label, "run": str(run), "games": games,
            "wins": wins, "losses": losses, "win_rate": win_rate,
            "opponent_mean": opponent_mean,
            "elo_fixed_point": opponent_mean + elo(win_rate) if played else None,
            "displayed": games[-1]["ours_after"] if games else None,
            "buckets": by_bucket,
        })

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
