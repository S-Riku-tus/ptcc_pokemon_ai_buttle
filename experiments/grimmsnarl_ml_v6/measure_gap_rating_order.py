"""Does each gap actually run with pilot rating, or only with our pin's outlier?

Every version in this line has justified a change by "the behaviour is monotone
in pilot rating and we are off the end of it". That is a claim about 21 points
and it has never been tested as one. It matters here because the escalation
teacher was picked on it, and because the leaderboard churns: of the 21 pilots
in the corpus only 8 are still in the top 60 a day later, and the 1220.2-rated
pilot the analysis calls elite is not one of them.

Spearman rho over the 21 pilots, against the submission score recorded when
their games were collected and against the current leaderboard score for the
subset still ranked. A two-sided permutation test, because n=21 and the rates
are bounded.

Usage:
    python experiments/grimmsnarl_ml_v6/measure_gap_rating_order.py \
        --gaps experiments/grimmsnarl_ml_v6/teacher_by_gap.json \
        --leaderboard .tmp/v6/lb_snapshot/latest/leaderboard_top60.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path

METRICS = (
    ("froslass turn rate", ("froslass", "turn_take_rate")),
    ("froslass turn rate, mirror", ("froslass", "turn_take_rate_mirror")),
    ("froslass turn rate, negative", ("froslass", "turn_take_rate_negative")),
    ("froslass decision rate", ("froslass", "decision_take_rate")),
    ("dead Unfair Stamp taken", ("stamp", "dead_take_rate")),
    ("attachment made when legal", ("attachment", "made_when_legal")),
)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        shared = (position + end) / 2 + 1
        for index in order[position:end + 1]:
            out[index] = shared
        position = end + 1
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    da = [x - mean_a for x in a]
    db = [y - mean_b for y in b]
    denominator = (
        sum(x * x for x in da) ** 0.5 * sum(y * y for y in db) ** 0.5
    )
    return sum(x * y for x, y in zip(da, db)) / denominator if denominator else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def permutation_p(a: list[float], b: list[float], trials: int = 20000) -> float:
    """Two-sided p by shuffling one side deterministically.

    A seeded PRNG is avoided: the shuffles are generated from a fixed linear
    congruential sequence so the number is reproducible without a seed
    argument, which matters because this figure ends up in a report.
    """
    observed = abs(spearman(a, b))
    state = 12345
    extreme = 0
    shuffled = list(b)
    for _ in range(trials):
        for index in range(len(shuffled) - 1, 0, -1):
            state = (1103515245 * state + 12345) % (1 << 31)
            swap = state % (index + 1)
            shuffled[index], shuffled[swap] = shuffled[swap], shuffled[index]
        if abs(spearman(a, shuffled)) >= observed - 1e-12:
            extreme += 1
    return round((extreme + 1) / (trials + 1), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    gaps = json.loads(args.gaps.read_text(encoding="utf-8"))
    current: dict[str, float] = {}
    if args.leaderboard and args.leaderboard.exists():
        for row in csv.DictReader(
            args.leaderboard.open(encoding="utf-8-sig")
        ):
            current[row["team_id"]] = float(row["leaderboard_score"])

    report: dict[str, dict] = {}
    for label, (section, key) in METRICS:
        report[label] = {}
        for basis, scores in (
            ("collected", {t: v["submission_score"] for t, v in gaps.items()}),
            ("current", current),
        ):
            pairs = [
                (scores[team], gaps[team][section][key])
                for team in gaps
                if team in scores and gaps[team][section][key] is not None
            ]
            if len(pairs) < 6:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            report[label][basis] = {
                "n": len(pairs),
                "spearman": round(spearman(xs, ys), 4),
                "permutation_p": permutation_p(xs, ys),
            }

    print(f"{'metric':>30} | {'n':>3} {'rho':>7} {'p':>7} | {'n':>3} {'rho':>7} {'p':>7}")
    print(f"{'':>30} | {'collected score':^19} | {'current score':^19}")
    for label, bases in report.items():
        cells = []
        for basis in ("collected", "current"):
            entry = bases.get(basis)
            cells.append(
                f"{entry['n']:>3} {entry['spearman']:>7.3f} {entry['permutation_p']:>7.4f}"
                if entry else f"{'-':>19}"
            )
        print(f"{label:>30} | {cells[0]} | {cells[1]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
