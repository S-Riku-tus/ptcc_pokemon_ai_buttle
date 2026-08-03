"""Separate rule-based pilots from search-based pilots by compute spent.

v1 found that agreement with a pilot is inversely related to that pilot's
rating, and pinned the most imitable (weakest) pilot as a result. Before
re-targeting a stronger pilot, we need to know whether the strong pilots are
merely *harder* to fit or structurally *unfittable* by a per-option ranker.

``remainingOverageTime`` starts at 600 s and drops as an agent thinks, so the
per-decision delta is that agent's wall-clock compute. A table-driven rule bot
spends microseconds; a search agent spends visible time and its choice depends
on rollouts no static feature vector can express.

Also measures opening determinism: the first MAIN decision of a game, keyed by
(hand multiset, active, going first). Those collide across games, so per-pilot
self-consistency there is a usable determinism probe even though full-state
signatures never repeat.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_teacher_corpus import (  # noqa: E402
    MAIN_CONTEXT, _cards, option_sig,
)


def _opening_key(current: dict[str, Any], menu: tuple) -> tuple:
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    me = players[your] if your < len(players) else {}
    return (
        int(current.get("turn", -1)),
        int(current.get("turnActionCount", -1)),
        int(current.get("firstPlayer", -1)) == your,
        tuple(sorted(int(c.get("id", -1)) for c in _cards(me, "hand"))),
        tuple(sorted(int(c.get("id", -1)) for c in _cards(me, "active"))),
        tuple(sorted(int(c.get("id", -1)) for c in _cards(me, "bench"))),
        menu,
    )


def _worker(payload: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    replay_root, rows = payload
    per_team_times: dict[int, list[float]] = defaultdict(list)
    per_team_game_time: dict[int, list[float]] = defaultdict(list)
    openings: list[tuple] = []
    stats: Counter[str] = Counter()

    for row in rows:
        path = Path(replay_root) / f"episode_{row['episode_id']}.json"
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats["replay_unreadable"] += 1
            continue
        seat = int(row["seat_index"])
        team = int(row["team_id"])
        steps = replay.get("steps") or []

        previous_overage: float | None = None
        first_main_seen = False
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            overage = observation.get("remainingOverageTime")
            if record.get("status") != "ACTIVE":
                if isinstance(overage, (int, float)):
                    previous_overage = float(overage)
                continue
            if isinstance(overage, (int, float)):
                if previous_overage is not None:
                    delta = previous_overage - float(overage)
                    if 0.0 <= delta < 60.0:
                        per_team_times[team].append(delta)
                previous_overage = float(overage)

            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            action = (steps[index + 1][seat] or {}).get("action")
            if (
                first_main_seen
                or int(select.get("context", -1)) != MAIN_CONTEXT
                or len(options) < 2
                or not isinstance(action, list)
                or len(action) != 1
                or not isinstance(action[0], int)
                or not 0 <= action[0] < len(options)
            ):
                continue
            first_main_seen = True
            current = observation.get("current") or {}
            sigs = [option_sig(current, option) for option in options]
            openings.append((
                team,
                repr(_opening_key(current, tuple(sorted(sigs)))),
                repr(sigs[action[0]]),
            ))

        final = steps[-1] if steps else []
        last = (final[seat] or {}).get("observation") if seat < len(final) else None
        if isinstance(last, dict):
            remaining = last.get("remainingOverageTime")
            if isinstance(remaining, (int, float)):
                per_team_game_time[team].append(600.0 - float(remaining))
        stats["episodes"] += 1

    return {
        "times": {k: v for k, v in per_team_times.items()},
        "game_times": {k: v for k, v in per_team_game_time.items()},
        "openings": openings,
        "stats": dict(stats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--deck-hash", default="9714ab5c3996f6cc")
    parser.add_argument("--limit-per-team", type=int, default=120)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v2" / "pilot_compute.json",
    )
    args = parser.parse_args()

    index = pd.read_csv(args.data_root / "indexes" / "episodes.csv")
    index = index[index["download_status"] == "success"]
    if args.deck_hash:
        index = index[index["deck_hash"] == args.deck_hash]
    index = index.drop_duplicates(subset=["episode_id", "seat_index"])
    if args.limit_per_team:
        index = index.groupby("team_id", group_keys=False).head(
            args.limit_per_team
        )
    meta = index.drop_duplicates("team_id").set_index("team_id")
    rows = index[["team_id", "episode_id", "seat_index"]].to_dict("records")
    print(f"trajectories={len(rows)}", flush=True)

    replay_root = str((args.data_root / "replays").resolve())
    workers = max(1, min(args.workers, len(rows)))
    chunks = [rows[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        parts = list(executor.map(
            _worker, [(replay_root, chunk) for chunk in chunks]
        ))

    times: dict[int, list[float]] = defaultdict(list)
    game_times: dict[int, list[float]] = defaultdict(list)
    openings: list[tuple] = []
    for part in parts:
        for team, values in part["times"].items():
            times[int(team)].extend(values)
        for team, values in part["game_times"].items():
            game_times[int(team)].extend(values)
        openings.extend(part["openings"])

    # Opening determinism: same team, same opening key, twice.
    by_team_key: dict[tuple, list[str]] = defaultdict(list)
    for team, key, chosen in openings:
        by_team_key[(team, key)].append(chosen)
    per_team_pairs: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for (team, _key), choices in by_team_key.items():
        if len(choices) < 2:
            continue
        sample = choices[:40]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                per_team_pairs[team][0] += 1
                per_team_pairs[team][1] += int(sample[i] == sample[j])

    report = []
    for team in sorted(times, key=lambda t: -float(np.mean(times[t]))):
        deltas = np.asarray(times[team])
        pairs, agree = per_team_pairs.get(team, [0, 0])
        report.append({
            "team_id": team,
            "leaderboard_rank": int(meta.loc[team, "leaderboard_rank"]),
            "submission_score": float(meta.loc[team, "submission_score"]),
            "decisions_timed": int(len(deltas)),
            "mean_ms_per_decision": round(float(deltas.mean()) * 1000, 2),
            "p95_ms_per_decision": round(
                float(np.percentile(deltas, 95)) * 1000, 2
            ),
            "max_ms_per_decision": round(float(deltas.max()) * 1000, 2),
            "mean_seconds_per_game": round(
                float(np.mean(game_times[team])), 2
            ) if game_times.get(team) else None,
            "opening_pairs": pairs,
            "opening_self_consistency": (
                round(agree / pairs, 4) if pairs else None
            ),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"teams": report}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    header = (
        f"{'team':<11}{'rank':>5}{'rating':>9}{'ms/dec':>9}"
        f"{'p95ms':>9}{'s/game':>9}{'openN':>8}{'openCons':>10}"
    )
    print(header)
    for row in report:
        print(
            f"{row['team_id']:<11}{row['leaderboard_rank']:>5}"
            f"{row['submission_score']:>9.1f}"
            f"{row['mean_ms_per_decision']:>9.2f}"
            f"{row['p95_ms_per_decision']:>9.2f}"
            f"{row['mean_seconds_per_game'] or -1:>9.2f}"
            f"{row['opening_pairs']:>8}"
            f"{(row['opening_self_consistency'] if row['opening_self_consistency'] is not None else -1):>10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
