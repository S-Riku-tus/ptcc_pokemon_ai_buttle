"""Compare Alakazam ladder runs by win rate bucketed on opponent rating.

Headline rating cannot rank versions: the same agent has scored 842.8 and
804.0 on two runs because Kaggle hands each run a different opponent pool.
Raw win rate has the same problem. What survives is win rate conditioned on
how strong the opponent was when the game started.

``fetch_submission_logs.py`` does not persist per-episode scores, so this
re-queries EpisodeService/ListEpisodes and reads ``initialScore`` from the
``agents[]`` entries.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_submission_logs import (  # noqa: E402
    LIST_EPISODES_URL,
    post_json,
)


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return (centre - spread, centre + spread)


def episode_rows(submission_id: int) -> list[dict]:
    data = post_json(LIST_EPISODES_URL, {"submissionId": submission_id})
    rows = []
    for episode in data.get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        if str(episode.get("type") or "") != "EPISODE_TYPE_PUBLIC":
            continue
        if str(episode.get("state") or "") != "COMPLETED":
            continue
        agents = episode.get("agents")
        if not isinstance(agents, list) or len(agents) != 2:
            continue
        mine = other = None
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            if str(agent.get("submissionId")) == str(submission_id):
                mine = agent
            else:
                other = agent
        if mine is None or other is None:
            continue
        reward = mine.get("reward")
        opponent_reward = other.get("reward")
        if reward is None or opponent_reward is None:
            continue
        opponent_initial = other.get("initialScore")
        if opponent_initial is None:
            continue
        rows.append({
            "episode_id": int(episode["id"]),
            "won": float(reward) > float(opponent_reward),
            "drawn": float(reward) == float(opponent_reward),
            "own_initial": mine.get("initialScore"),
            "own_updated": mine.get("updatedScore"),
            "opponent_initial": float(opponent_initial),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="NAME:SUBMISSION_ID",
    )
    parser.add_argument(
        "--edges", type=float, nargs="+",
        default=[0, 700, 800, 900, 10000],
        help="Opponent initialScore bucket edges.",
    )
    parser.add_argument(
        "--pooled-floor", type=float, default=800.0,
        help="Only opponents at or above this initialScore are pooled.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        name, _, submission = spec.partition(":")
        rows = episode_rows(int(submission))
        runs[name] = {"submission_id": int(submission), "episodes": rows}
        opponents = [row["opponent_initial"] for row in rows]
        finals = [
            row["own_updated"] for row in rows
            if row["own_updated"] is not None
        ]
        decided = [row for row in rows if not row["drawn"]]
        wins = sum(row["won"] for row in decided)
        low, high = wilson(wins, len(decided))
        runs[name]["summary"] = {
            "games": len(rows),
            "decided": len(decided),
            "wins": int(wins),
            "win_rate": wins / max(len(decided), 1),
            "win_rate_95ci": [low, high],
            "opponent_initial_mean": (
                statistics.fmean(opponents) if opponents else None
            ),
            "opponent_initial_median": (
                statistics.median(opponents) if opponents else None
            ),
            "final_score": max(finals) if finals else None,
        }
        print(f"{name}: {json.dumps(runs[name]['summary'])}", flush=True)

    edges = args.edges
    labels = [
        f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)
    ]
    buckets: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for name, run in runs.items():
        counts = {label: [0, 0] for label in labels}
        for row in run["episodes"]:
            if row["drawn"]:
                continue
            score = row["opponent_initial"]
            for index in range(len(edges) - 1):
                if edges[index] <= score < edges[index + 1]:
                    counts[labels[index]][0] += int(row["won"])
                    counts[labels[index]][1] += 1
                    break
        buckets[name] = counts

    print()
    header = f"{'bucket':>12s}" + "".join(f"{name:>18s}" for name in runs)
    print(header)
    for label in labels:
        line = f"{label:>12s}"
        for name in runs:
            wins, total = buckets[name][label]
            cell = (
                f"{wins}/{total} {wins / total:.0%}" if total else "-"
            )
            line += f"{cell:>18s}"
        print(line)

    # Weak opponents are close to free wins for every version, so the
    # discriminating evidence is the pool the ladder actually gates on.
    pooled = {}
    for name, run in runs.items():
        wins = total = 0
        for row in run["episodes"]:
            if row["drawn"] or row["opponent_initial"] < args.pooled_floor:
                continue
            wins += int(row["won"])
            total += 1
        low, high = wilson(wins, total)
        pooled[name] = {
            "wins": wins, "games": total,
            "win_rate": wins / max(total, 1),
            "win_rate_95ci": [low, high],
        }
    print()
    print(f"pooled versus opponents rated >= {args.pooled_floor:.0f}")
    for name, value in pooled.items():
        print(f"  {name}: {value['wins']}/{value['games']} = "
              f"{value['win_rate']:.1%} "
              f"(95% CI {value['win_rate_95ci'][0]:.1%}"
              f"-{value['win_rate_95ci'][1]:.1%})")

    report = {
        "pooled_versus_strong_opponents": {
            "floor": args.pooled_floor,
            "runs": pooled,
        },
        "runs": {
            name: {
                "submission_id": run["submission_id"],
                "summary": run["summary"],
                "buckets": {
                    label: {
                        "wins": buckets[name][label][0],
                        "games": buckets[name][label][1],
                        "win_rate": (
                            buckets[name][label][0] / buckets[name][label][1]
                            if buckets[name][label][1] else None
                        ),
                    }
                    for label in labels
                },
            }
            for name, run in runs.items()
        },
        "bucket_edges": edges,
        "method": (
            "Win rate conditioned on the opponent's initialScore at match "
            "time. Headline rating and raw win rate both track the opponent "
            "pool rather than agent quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
