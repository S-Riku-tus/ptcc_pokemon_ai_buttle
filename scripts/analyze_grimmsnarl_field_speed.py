"""Did the field get faster, and is that why going second collapsed?

v26 and v27 pool to 0.854 going first and 0.424 going second over 74 games
(Fisher p = 0.0002) while v22, whose actions they reproduce, showed no split
at all over 190.  Either the two late runs are unlucky, or the field they met
punishes the second seat harder than the field v22 met.

``opponent first attack turn`` is the cleanest test available: it is a
property of the opponent's deck and policy, not of ours, and it is recorded in
every replay.  If the field's first attack moved earlier between 08-13 and
08-15, the second seat loses a turn of setup against a clock that got shorter,
and that is a meta change rather than a regression in our agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v27/version_games.csv"
GROUPS = {
    "v22": ("v22_a", "v22_b", "v22_c", "v22_d"),
    "v23": ("v23",),
    "v24": ("v24_a", "v24_b"),
    "v25": ("v25_a", "v25_b"),
    "v26": ("v26",),
    "v27": ("v27",),
}
WINDOWS = (
    ("08-13", "2026-08-13", "2026-08-14"),
    ("08-14", "2026-08-14", "2026-08-15"),
    ("08-15", "2026-08-15", "2026-08-16"),
)


def load() -> list[dict[str, Any]]:
    rows = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if raw["went_first"] not in ("first", "second"):
            continue
        row: dict[str, Any] = dict(raw)
        for key in (
            "won", "turns", "our_turns", "our_prize_left", "opp_prize_left",
            "our_bodies_left", "shadow_attacks", "attacks", "exact_mirror",
            "first_ready_turn", "first_shadow_turn", "first_attack_turn",
            "opp_first_attack_turn", "own_first_ready_turn",
            "own_first_shadow_turn", "our_deck_left", "board_out",
        ):
            row[key] = int(raw[key]) if raw[key] not in ("", "None") else None
        row["opponent_rating"] = (
            float(raw["opponent_rating"]) if raw["opponent_rating"] else None
        )
        row["day"] = raw["create_time"][:10]
        row["group"] = next(
            (g for g, labels in GROUPS.items() if raw["version"] in labels),
            raw["version"],
        )
        # ``opp_first_attack_turn`` is the shared turn counter.  Converting it
        # to the opponent's own turn ordinal makes "how fast is this deck"
        # comparable between the games where they moved first and second.
        opp_went_first = row["went_first"] == "second"
        turn = row["opp_first_attack_turn"]
        row["opp_own_first_attack"] = (
            None if turn is None
            else ((turn + 1) // 2 if opp_went_first else turn // 2)
        )
        rows.append(row)
    rows.sort(key=lambda r: r["create_time"])
    return rows


def mean(values) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return sum(numbers) / len(numbers) if numbers else None


def fmt(value, digits: int = 2) -> str:
    return " n/a" if value is None else f"{value:.{digits}f}"


def window_of(row: dict[str, Any]) -> str:
    for name, low, high in WINDOWS:
        if low <= row["day"] < high:
            return name
    return "?"


def rate(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "     -"
    wins = sum(r["won"] for r in rows)
    return f"{wins:>3}-{len(rows) - wins:<3}{wins / len(rows):.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/field_speed.json",
    )
    args = parser.parse_args()
    rows = load()
    payload: dict[str, Any] = {}

    print("=== 1. how fast the opponent's deck attacked, by calendar day ===")
    print("opp_own_first_attack = the opponent's own turn ordinal on which it "
          "made its first attack.  Lower = faster field.\n")
    print(f"{'window':<8}{'n':>5}{'opp 1st atk':>13}{'never atk':>11}"
          f"{'our 1st shadow':>16}{'turns':>8}{'our wr':>9}")
    speed = {}
    for name, _, _ in WINDOWS:
        subset = [r for r in rows if window_of(r) == name]
        if not subset:
            continue
        never = sum(1 for r in subset if r["opp_first_attack_turn"] is None)
        value = mean(r["opp_own_first_attack"] for r in subset)
        speed[name] = {
            "games": len(subset),
            "opp_first_attack_own_turn": value,
            "never_attacked": never,
            "our_first_shadow": mean(r["own_first_shadow_turn"] for r in subset),
            "win_rate": sum(r["won"] for r in subset) / len(subset),
        }
        print(
            f"{name:<8}{len(subset):>5}{fmt(value):>13}{never:>11}"
            f"{fmt(mean(r['own_first_shadow_turn'] for r in subset)):>16}"
            f"{fmt(mean(r['turns'] for r in subset)):>8}"
            f"{sum(r['won'] for r in subset) / len(subset):>9.3f}"
        )
    payload["speed_by_window"] = speed

    early = [
        r["opp_own_first_attack"] for r in rows
        if window_of(r) in ("08-13", "08-14") and r["opp_own_first_attack"]
    ]
    late = [
        r["opp_own_first_attack"] for r in rows
        if window_of(r) == "08-15" and r["opp_own_first_attack"]
    ]
    stat = mannwhitneyu(late, early, alternative="less")
    print(f"\n08-15 vs earlier, opponent first attack turn, Mann-Whitney "
          f"one-sided earlier: U={stat.statistic:.0f} p={stat.pvalue:.4g}  "
          f"(means {mean(late):.2f} vs {mean(early):.2f})")

    print("\nsame, per version:")
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        print(
            f"  {group:<5} n={len(subset):<4} opp 1st atk "
            f"{fmt(mean(r['opp_own_first_attack'] for r in subset))}   "
            f"our 1st shadow "
            f"{fmt(mean(r['own_first_shadow_turn'] for r in subset))}   "
            f"our 1st ready "
            f"{fmt(mean(r['own_first_ready_turn'] for r in subset))}"
        )

    print("\n=== 2. turn order x window ===")
    print(f"{'window':<8}{'first':>14}{'second':>14}{'diff':>8}{'fisher':>9}")
    for name, _, _ in WINDOWS:
        subset = [r for r in rows if window_of(r) == name]
        first = [r for r in subset if r["went_first"] == "first"]
        second = [r for r in subset if r["went_first"] == "second"]
        if not first or not second:
            continue
        table = [
            [sum(r["won"] for r in first), len(first) - sum(r["won"] for r in first)],
            [sum(r["won"] for r in second), len(second) - sum(r["won"] for r in second)],
        ]
        a, b = table[0][0] / len(first), table[1][0] / len(second)
        print(
            f"{name:<8}{rate(first):>14}{rate(second):>14}{a - b:>+8.3f}"
            f"{float(fisher_exact(table).pvalue):>9.4f}"
        )

    print("\n=== 3. what the second seat lost to, 08-15 vs earlier ===")
    for label, predicate in (
        ("earlier, second", lambda r: window_of(r) != "08-15" and r["went_first"] == "second"),
        ("08-15,  second", lambda r: window_of(r) == "08-15" and r["went_first"] == "second"),
        ("earlier, first ", lambda r: window_of(r) != "08-15" and r["went_first"] == "first"),
        ("08-15,  first ", lambda r: window_of(r) == "08-15" and r["went_first"] == "first"),
    ):
        subset = [r for r in rows if predicate(r)]
        losses = [r for r in subset if not r["won"]]
        print(
            f"{label}  n={len(subset):<4} wr "
            f"{sum(r['won'] for r in subset) / len(subset):.3f}   "
            f"opp 1st atk {fmt(mean(r['opp_own_first_attack'] for r in subset))}   "
            f"our 1st shadow {fmt(mean(r['own_first_shadow_turn'] for r in subset))}   "
            f"prizes left on loss {fmt(mean(r['our_prize_left'] for r in losses))}   "
            f"blowout losses (>=5 left) "
            f"{sum(1 for r in losses if (r['our_prize_left'] or 0) >= 5)}/{len(losses)}"
        )

    print("\n=== 4. mirror x turn order ===")
    print(f"{'group':<7}{'mirror first':>16}{'mirror second':>16}"
          f"{'other first':>16}{'other second':>16}")
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        cells = []
        for is_mirror in (1, 0):
            for order in ("first", "second"):
                cells.append(rate([
                    r for r in subset
                    if r["exact_mirror"] == is_mirror and r["went_first"] == order
                ]))
        print(f"{group:<7}" + "".join(f"{c:>16}" for c in cells))

    print("\n=== 5. family x turn order, 08-15 window ===")
    late_rows = [r for r in rows if window_of(r) == "08-15"]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in late_rows:
        families[row["opponent_family"]].append(row)
    for family, subset in sorted(families.items(), key=lambda i: -len(i[1])):
        first = [r for r in subset if r["went_first"] == "first"]
        second = [r for r in subset if r["went_first"] == "second"]
        print(
            f"  {family:<30} first {rate(first):<14} second {rate(second)}"
        )

    print("\n=== 6. our own clock, by turn order ===")
    print("If the second seat is losing because our own setup is slower "
          "there, the gap shows up here; if not, the clock is the field's.\n")
    print(f"{'group':<7}{'order':<8}{'1st ready':>11}{'1st shadow':>12}"
          f"{'shadow/game':>13}{'our turns':>11}{'wr':>8}")
    for group in ("v22", "v24", "v26", "v27"):
        for order in ("first", "second"):
            subset = [
                r for r in rows
                if r["group"] == group and r["went_first"] == order
            ]
            if not subset:
                continue
            print(
                f"{group:<7}{order:<8}"
                f"{fmt(mean(r['own_first_ready_turn'] for r in subset)):>11}"
                f"{fmt(mean(r['own_first_shadow_turn'] for r in subset)):>12}"
                f"{fmt(mean(r['shadow_attacks'] for r in subset)):>13}"
                f"{fmt(mean(r['our_turns'] for r in subset)):>11}"
                f"{sum(r['won'] for r in subset) / len(subset):>8.3f}"
            )

    print("\n=== 7. pooled 'v22-equivalent' policy (v22 + v26 + v27) ===")
    print("v27 reproduces v22's action on 2747 of 2755 stored decisions, so "
          "these three runs are one policy sampled at three times.\n")
    pooled = [r for r in rows if r["group"] in ("v22", "v26", "v27")]
    for label, subset in (
        ("pooled", pooled),
        ("pooled, first", [r for r in pooled if r["went_first"] == "first"]),
        ("pooled, second", [r for r in pooled if r["went_first"] == "second"]),
        ("pooled, opp>=950", [
            r for r in pooled if (r["opponent_rating"] or 0) >= 950
        ]),
        ("pooled, opp>=950 first", [
            r for r in pooled
            if (r["opponent_rating"] or 0) >= 950 and r["went_first"] == "first"
        ]),
        ("pooled, opp>=950 second", [
            r for r in pooled
            if (r["opponent_rating"] or 0) >= 950 and r["went_first"] == "second"
        ]),
    ):
        wins = sum(r["won"] for r in subset)
        low, high = wilson(wins, len(subset)) if subset else (0, 0)
        print(f"  {label:<24} n={len(subset):>3}  {wins}-{len(subset) - wins}  "
              f"{wins / max(len(subset), 1):.3f} [{low:.3f},{high:.3f}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
