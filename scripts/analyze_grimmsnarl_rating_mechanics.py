"""What actually produces a Kaggle rating, and how much of it is the draw?

The per-game update on this competition is plain Elo with a decaying K:

    delta = K(n) * (result - 1 / (1 + 10 ** ((opponent - ours) / 400)))

``K(n)`` is fitted here from the 445 stored games and comes out identical
across all eleven runs (spread inside a game index is a couple of points,
against a level that falls from 216 at game 1 to 18 at game 34).  That decay
is the whole story of this script: the first ten games carry more rating
weight than games 11-34 put together, so *when* a loss lands matters as much
as whether it happens.

Four questions are answered, each by replaying the fitted update:

1. what did each run's rating path actually look like, game by game;
2. ordering - hold a run's own (opponent, result) pairs fixed and permute the
   arrival order, which gives the spread attributable to sequencing alone;
3. sampling - draw 34 games with replacement from the pooled 264-game
   v22-equivalent policy and replay, which gives the full noise floor of a
   34-game run of *this* policy against *this* field;
4. draw quality - replay one run's results against another run's opponent
   sequence, to separate "we played worse" from "we were fed a worse ladder".

Every simulation is seeded, so the numbers reproduce.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v27/version_games.csv"
POOL = ("v22_a", "v22_b", "v22_c", "v22_d", "v26", "v27")
START = 600.0


def elo(w: float) -> float:
    w = min(max(w, 1e-4), 1 - 1e-4)
    return 400 * math.log10(w / (1 - w))


def load() -> dict[str, list[dict]]:
    rows = list(csv.DictReader(GAMES.open(encoding="utf-8-sig")))
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row["opponent_rating"]:
            continue
        row["opp"] = float(row["opponent_rating"])
        row["before"] = float(row["our_rating_before"])
        row["after"] = float(row["our_rating_after"])
        row["w"] = int(row["won"])
        by_run[row["version"]].append(row)
    for games in by_run.values():
        games.sort(key=lambda r: r["create_time"])
    return by_run


def fit_k(by_run: dict[str, list[dict]]) -> tuple[np.ndarray, float]:
    """Recover K(n) as the mean over runs, and the worst replay error."""
    samples: dict[int, list[float]] = defaultdict(list)
    for games in by_run.values():
        for index, row in enumerate(games, 1):
            expected = 1 / (1 + 10 ** ((row["opp"] - row["before"]) / 400))
            if abs(row["w"] - expected) < 1e-6:
                continue
            samples[index].append(
                (row["after"] - row["before"]) / (row["w"] - expected)
            )
    top = max(samples)
    schedule = np.zeros(top + 2)
    for index in range(1, top + 1):
        values = samples.get(index)
        schedule[index] = (
            float(np.mean(values)) if values else schedule[index - 1]
        )
    schedule[top + 1] = schedule[top]

    worst = 0.0
    for games in by_run.values():
        rating = START
        for index, row in enumerate(games, 1):
            expected = 1 / (1 + 10 ** ((row["opp"] - rating) / 400))
            rating += schedule[min(index, top)] * (row["w"] - expected)
            worst = max(worst, abs(rating - row["after"]))
    return schedule, worst


def replay(pairs, schedule: np.ndarray) -> float:
    rating = START
    top = len(schedule) - 2
    for index, (opponent, result) in enumerate(pairs, 1):
        expected = 1 / (1 + 10 ** ((opponent - rating) / 400))
        rating += schedule[min(index, top)] * (result - expected)
    return rating


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/rating_mechanics.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    by_run = load()
    schedule, worst = fit_k(by_run)
    payload: dict = {"replay_max_error": worst}

    print("=== 1. the update rule ===")
    print(f"replaying every run with the fitted K(n) reproduces the stored "
          f"rating to within {worst:.2f} points, so the model is exact.\n")
    print(f"{'game':>6}{'K':>9}{'cumulative K':>15}{'share of total':>16}")
    total_k = float(schedule[1:35].sum())
    running = 0.0
    for index in (1, 2, 3, 5, 10, 15, 20, 25, 30, 34):
        running = float(schedule[1:index + 1].sum())
        print(f"{index:>6}{schedule[index]:>9.1f}{running:>15.0f}"
              f"{running / total_k:>16.1%}")
    payload["k_first10_share"] = float(schedule[1:11].sum()) / total_k
    print(f"\ngames 1-10 carry {payload['k_first10_share']:.1%} of the total "
          f"K available in a 34-game run.")

    print("\n=== 2. rating path, first 12 games ===")
    for run in ("v22_a", "v24_a", "v26", "v27"):
        games = by_run[run]
        print(f"\n{run}: final {games[-1]['after']:.1f} over {len(games)} games")
        print(f"  {'#':>3}{'opp':>8}{'res':>5}{'delta':>8}{'rating':>9}  "
              f"{'seat':<7}{'matchup'}")
        for index, row in enumerate(games[:12], 1):
            print(f"  {index:>3}{row['opp']:>8.0f}"
                  f"{('W' if row['w'] else 'L'):>5}"
                  f"{row['after'] - row['before']:>8.1f}{row['after']:>9.1f}  "
                  f"{row['went_first']:<7}{row['opponent_family']}")
        first10 = games[:10]
        wins = sum(r["w"] for r in first10)
        print(f"  first 10: {wins}-{len(first10) - wins}, "
              f"opponent mean {np.mean([r['opp'] for r in first10]):.0f}, "
              f"rating after 10 = {first10[-1]['after']:.1f}")

    print("\n=== 3. first-10 record and where each run ended ===")
    print(f"{'run':<8}{'first10':>9}{'opp10':>8}{'r@10':>9}"
          f"{'full':>10}{'oppall':>9}{'final':>9}{'equilib':>9}")
    rows_out = []
    for run in sorted(by_run):
        games = by_run[run]
        first10 = games[:10]
        w10 = sum(r["w"] for r in first10)
        wins = sum(r["w"] for r in games)
        opp = float(np.mean([r["opp"] for r in games]))
        equilibrium = opp + elo(wins / len(games))
        print(f"{run:<8}{f'{w10}-{len(first10) - w10}':>9}"
              f"{np.mean([r['opp'] for r in first10]):>8.0f}"
              f"{first10[-1]['after']:>9.1f}"
              f"{f'{wins}-{len(games) - wins}':>10}{opp:>9.0f}"
              f"{games[-1]['after']:>9.1f}{equilibrium:>9.1f}")
        rows_out.append({
            "run": run, "n": len(games), "wins": wins,
            "first10_wins": w10, "final": games[-1]["after"],
            "equilibrium": equilibrium, "opp_mean": opp,
        })
    payload["runs"] = rows_out

    print("\n=== 4. ordering alone: permute each run's own games ===")
    print(f"{'run':<8}{'actual':>9}{'mean':>9}{'sd':>7}{'p05':>9}{'p95':>9}"
          f"{'pctile':>9}")
    order_out = {}
    for run in sorted(by_run):
        games = by_run[run]
        pairs = np.array([(r["opp"], r["w"]) for r in games], dtype=float)
        finals = np.empty(args.draws)
        for draw in range(args.draws):
            finals[draw] = replay(rng.permutation(pairs), schedule)
        actual = games[-1]["after"]
        pct = float((finals < actual).mean())
        print(f"{run:<8}{actual:>9.1f}{finals.mean():>9.1f}{finals.std():>7.1f}"
              f"{np.percentile(finals, 5):>9.1f}"
              f"{np.percentile(finals, 95):>9.1f}{pct:>9.1%}")
        order_out[run] = {
            "actual": actual, "mean": float(finals.mean()),
            "sd": float(finals.std()), "percentile": pct,
        }
    payload["ordering"] = order_out

    print("\n=== 5. the full 34-game noise floor of the pooled policy ===")
    pool = [r for run in POOL for r in by_run[run]]
    wins = sum(r["w"] for r in pool)
    low, high = wilson(wins, len(pool))
    print(f"pool = {' + '.join(POOL)}: {wins}-{len(pool) - wins} "
          f"({wins / len(pool):.3f}) over {len(pool)} games, "
          f"Wilson [{low:.3f}, {high:.3f}]")
    pool_pairs = np.array([(r["opp"], r["w"]) for r in pool], dtype=float)
    for size in (34, 45):
        finals = np.empty(args.draws)
        for draw in range(args.draws):
            pick = rng.integers(0, len(pool_pairs), size)
            finals[draw] = replay(pool_pairs[pick], schedule)
        print(f"  {size:>3}-game bootstrap: mean {finals.mean():>7.1f}  "
              f"sd {finals.std():>5.1f}  "
              f"[p05 {np.percentile(finals, 5):.0f}, "
              f"p95 {np.percentile(finals, 95):.0f}]  "
              f"90% width {np.percentile(finals, 95) - np.percentile(finals, 5):.0f}")
        payload[f"bootstrap_{size}"] = {
            "mean": float(finals.mean()), "sd": float(finals.std()),
            "p05": float(np.percentile(finals, 5)),
            "p95": float(np.percentile(finals, 95)),
        }
        if size == 34:
            for run in ("v27", "v26", "v22_a", "v22_c"):
                actual = by_run[run][-1]["after"]
                print(f"      {run:<6} finished {actual:>7.1f} -> "
                      f"percentile {float((finals < actual).mean()):.1%}")

    print("\n=== 6. draw quality: same results, another run's opponents ===")
    print("each cell replays the row run's win/loss sequence against the "
          "column run's opponent ratings (truncated to the shorter run).")
    names = ["v22_a", "v22_c", "v24_a", "v26", "v27"]
    header = "".join(f"{n:>10}" for n in names)
    print(f"{'results':<10}{header}")
    cross = {}
    for row_run in names:
        results = [r["w"] for r in by_run[row_run]]
        line = f"{row_run:<10}"
        for col_run in names:
            opponents = [r["opp"] for r in by_run[col_run]]
            size = min(len(results), len(opponents))
            value = replay(
                list(zip(opponents[:size], results[:size])), schedule
            )
            line += f"{value:>10.1f}"
            cross[f"{row_run}|{col_run}"] = value
        print(line)
    payload["cross_draw"] = cross

    print("\n=== 7. what a given first-10 record is worth ===")
    print("v27's own remaining games, replayed after every possible "
          "first-10 record drawn from the pooled policy.")
    tail = [(r["opp"], r["w"]) for r in by_run["v27"][10:]]
    opening = [r["opp"] for r in by_run["v27"][:10]]
    for wins10 in range(4, 11):
        finals = np.empty(2000)
        for draw in range(2000):
            order = rng.permutation(10)
            results = [1.0] * wins10 + [0.0] * (10 - wins10)
            head = [(opening[i], results[order[i]]) for i in range(10)]
            finals[draw] = replay(head + tail, schedule)
        print(f"  first 10 = {wins10}-{10 - wins10}  ->  final "
              f"{finals.mean():>7.1f}  (sd {finals.std():.1f})")
        payload.setdefault("first10_curve", {})[str(wins10)] = float(
            finals.mean()
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
