"""Is Adrena-Brain a lever, or is it what winning looks like?

Munkidori's Adrena-Brain is the single strongest correlate of winning in the
480 stored games: 5+ uses wins 0.766 against 0.449, and the effect survives the
per-turn denominator.  It is also exactly the shape of correlate that burned
the Alakazam line - Powerful Hand count looked like a lever, was pushed, and
bought nothing - so the count has to be split into the part that precedes the
result and the part that follows it.

The split here is by our own turn index:

* uses inside our first three own turns cannot be caused by a board we have
  already won, so they are a candidate lever;
* uses after that are as likely to be a symptom (a live Munkidori and a live
  board) as a cause.

Both are fitted against winning with opponent rating and turn order held
fixed, and reported per archetype family.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22"))

import ml_features as mf  # noqa: E402

import analyze_grimmsnarl_v27_vs_champions as champ  # noqa: E402
import build_grimmsnarl_version_games as builder  # noqa: E402

champ.GROUPS["v28"] = ("v28",)


def early_counts(replay: dict[str, Any], seat: int) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    current_late = builder._late_current(steps)
    went_first: bool | None = None
    if current_late is not None:
        first = int(current_late.get("firstPlayer", -1))
        went_first = (first == seat) if first >= 0 else None

    by_own_turn: dict[int, int] = defaultdict(int)
    munkidori_by_turn: dict[int, int] = {}
    energised_by_turn: dict[int, int] = {}
    damaged_by_turn: dict[int, int] = {}
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) < 2 or not options:
            continue
        own_turn = builder._own_turn(int(current.get("turn", -1)), went_first)
        us = players[seat]
        bodies = mf._cards(us, "active") + mf._cards(us, "bench")
        munkidori = [
            c for c in bodies if int(c.get("id", -1)) == mf.MUNKIDORI_ID
        ]
        if munkidori:
            munkidori_by_turn.setdefault(own_turn, 1)
        if any(mf._dark_energy_count(c) >= 1 for c in munkidori):
            energised_by_turn.setdefault(own_turn, 1)
        if any(
            int(c.get("hp", 0)) < int(c.get("maxHp", c.get("hp", 0)) or 0)
            for c in bodies
        ):
            damaged_by_turn.setdefault(own_turn, 1)
        action = (steps[index + 1][seat] or {}).get("action")
        picked = [
            int(value) for value in action
            if isinstance(value, int) and 0 <= int(value) < len(options)
        ] if isinstance(action, list) else []
        for choice in picked:
            option = options[choice]
            try:
                kind = mf.action_type(current, option, select)
            except Exception:  # noqa: BLE001
                continue
            card = mf.candidate_card(current, option, select) or {}
            if kind == "ability" and int(card.get("id", -1)) == mf.MUNKIDORI_ID:
                by_own_turn[own_turn] += 1
    return {
        "by_own_turn": dict(by_own_turn),
        "munkidori_turns": sorted(munkidori_by_turn),
        "early": sum(count for turn, count in by_own_turn.items() if turn <= 3),
        "late": sum(count for turn, count in by_own_turn.items() if turn > 3),
        "munkidori_by_own_turn_2": int(
            any(turn <= 2 for turn in munkidori_by_turn)
        ),
        "munkidori_by_own_turn_3": int(
            any(turn <= 3 for turn in munkidori_by_turn)
        ),
        "energised_by_own_turn_2": int(
            any(turn <= 2 for turn in energised_by_turn)
        ),
        "energised_by_own_turn_3": int(
            any(turn <= 3 for turn in energised_by_turn)
        ),
        "damaged_by_own_turn_3": int(
            any(turn <= 3 for turn in damaged_by_turn)
        ),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/version_games.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/adrena_lever.json",
    )
    args = parser.parse_args()

    rows = champ.load(args.games)
    run_dirs = {label: ROOT / builder.RUN_ROOT / path for label, _, path in builder.RUNS}

    enriched = []
    missing = 0
    for row in rows:
        run_dir = run_dirs.get(row["version"])
        if run_dir is None:
            continue
        path = (
            run_dir / "episodes" / str(row["episode_id"]) / "replay"
            / f"episode_{row['episode_id']}.json"
        )
        if not path.exists():
            missing += 1
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        counts = early_counts(replay, row["seat"])
        if counts is None:
            missing += 1
            continue
        enriched.append({**row, **counts})
    print(f"walked {len(enriched)} games ({missing} missing)")

    champ.section("1. Adrena-Brain split by when it happened")
    print("  early = our own turns 1-3, late = own turn 4 onward")
    for label, flag in (
        ("any early Adrena-Brain", lambda r: float(r["early"] >= 1)),
        ("2+ early Adrena-Brain", lambda r: float(r["early"] >= 2)),
        ("3+ early Adrena-Brain", lambda r: float(r["early"] >= 3)),
        ("any late Adrena-Brain", lambda r: float(r["late"] >= 1)),
        ("3+ late Adrena-Brain", lambda r: float(r["late"] >= 3)),
        ("Munkidori in play by own turn 2", lambda r: float(r["munkidori_by_own_turn_2"])),
    ):
        on = [r for r in enriched if flag(r) >= 0.5]
        off = [r for r in enriched if flag(r) < 0.5]
        fit = champ.fit_dummy(enriched, flag)
        print(
            f"  {label:<34} on {sum(r['won'] for r in on):>3}/{len(on):<3} "
            f"{sum(r['won'] for r in on) / max(len(on), 1):.3f}   "
            f"off {sum(r['won'] for r in off):>3}/{len(off):<3} "
            f"{sum(r['won'] for r in off) / max(len(off), 1):.3f}   "
            f"elo {str(fit.get('elo')):>7}  p {fit.get('p')}"
        )

    champ.section("2. Early Adrena-Brain, restricted to games we could still lose")
    print("  the late count is unusable if the game was already decided, so the "
          "early count is repeated inside the games that reached own turn 5+")
    long_games = [r for r in enriched if (r["our_turns"] or 0) >= 5]
    for label, flag in (
        ("any early Adrena-Brain", lambda r: float(r["early"] >= 1)),
        ("2+ early Adrena-Brain", lambda r: float(r["early"] >= 2)),
    ):
        fit = champ.fit_dummy(long_games, flag)
        on = [r for r in long_games if flag(r) >= 0.5]
        off = [r for r in long_games if flag(r) < 0.5]
        print(
            f"  {label:<34} on {sum(r['won'] for r in on):>3}/{len(on):<3} "
            f"{sum(r['won'] for r in on) / max(len(on), 1):.3f}   "
            f"off {sum(r['won'] for r in off):>3}/{len(off):<3} "
            f"{sum(r['won'] for r in off) / max(len(off), 1):.3f}   "
            f"elo {str(fit.get('elo')):>7}  p {fit.get('p')}"
        )

    champ.section("3. Per version and per family")
    groups = ["v22", "v24", "v25", "v26", "v27", "v28"]
    print(f"  {'version':<8}{'games':>6}{'early/game':>12}{'late/game':>11}"
          f"{'munki by t2':>13}")
    per_version = {}
    for name in groups:
        subset = [r for r in enriched if r["group"] == name]
        if not subset:
            continue
        early = champ.mean(r["early"] for r in subset)
        late = champ.mean(r["late"] for r in subset)
        munki = champ.mean(r["munkidori_by_own_turn_2"] for r in subset)
        print(f"  {name:<8}{len(subset):>6}{champ.fmt(early, 2):>12}"
              f"{champ.fmt(late, 2):>11}{champ.fmt(munki, 2):>13}")
        per_version[name] = {"games": len(subset), "early": early, "late": late,
                             "munkidori_by_t2": munki}

    print(f"\n  {'family':<26}{'games':>6}{'wr':>7}{'early W':>9}{'early L':>9}"
          f"{'munki t2 W':>12}{'munki t2 L':>12}")
    per_family = {}
    for family in sorted({r["opponent_family"] for r in enriched}):
        subset = [r for r in enriched if r["opponent_family"] == family]
        if len(subset) < 15:
            continue
        wins = [r for r in subset if r["won"]]
        losses = [r for r in subset if not r["won"]]
        print(
            f"  {family:<26}{len(subset):>6}{len(wins) / len(subset):>7.3f}"
            f"{champ.fmt(champ.mean(r['early'] for r in wins), 2):>9}"
            f"{champ.fmt(champ.mean(r['early'] for r in losses), 2):>9}"
            f"{champ.fmt(champ.mean(r['munkidori_by_own_turn_2'] for r in wins), 2):>12}"
            f"{champ.fmt(champ.mean(r['munkidori_by_own_turn_2'] for r in losses), 2):>12}"
        )
        per_family[family] = {
            "games": len(subset),
            "win_rate": round(len(wins) / len(subset), 4),
            "early_wins": champ.mean(r["early"] for r in wins),
            "early_losses": champ.mean(r["early"] for r in losses),
        }

    champ.section("4. What blocks an early Adrena-Brain")
    print("  the ability needs a {D} Energy on Munkidori and a damage counter "
          "on one of our own Pokemon, so a missing early use is one of three "
          "states")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        if row["early"]:
            buckets["used it"].append(row)
        elif not row["munkidori_by_own_turn_3"]:
            buckets["no munkidori"].append(row)
        elif not row["energised_by_own_turn_3"]:
            buckets["no energy"].append(row)
        elif not row["damaged_by_own_turn_3"]:
            buckets["no damage"].append(row)
        else:
            buckets["had it, skipped"].append(row)
    blocked = {key: len(value) for key, value in buckets.items()}
    total = sum(v for k, v in blocked.items() if k != "used it")
    print(f"  {total} games with no Adrena-Brain in own turns 1-3:")
    for key in ("no damage", "no munkidori", "no energy", "had it, skipped",
                "used it"):
        group = buckets.get(key) or []
        if not group:
            continue
        wins = sum(r["won"] for r in group)
        share = "" if key == "used it" else f" ({len(group) / max(total, 1):.1%})"
        print(
            f"    {key:<16} {len(group):>4}{share:<8} win rate "
            f"{wins / len(group):.3f}"
        )

    print("\n  the same states as win-rate flags:")
    for label, flag in (
        ("Munkidori energised by own t2",
         lambda r: float(r["energised_by_own_turn_2"])),
        ("Munkidori energised by own t3",
         lambda r: float(r["energised_by_own_turn_3"])),
        ("own damage by own t3",
         lambda r: float(r["damaged_by_own_turn_3"])),
    ):
        on = [r for r in enriched if flag(r) >= 0.5]
        off = [r for r in enriched if flag(r) < 0.5]
        fit = champ.fit_dummy(enriched, flag)
        print(
            f"  {label:<32} on {sum(r['won'] for r in on):>3}/{len(on):<3} "
            f"{sum(r['won'] for r in on) / max(len(on), 1):.3f}   "
            f"off {sum(r['won'] for r in off):>3}/{len(off):<3} "
            f"{sum(r['won'] for r in off) / max(len(off), 1):.3f}   "
            f"elo {str(fit.get('elo')):>7}  p {fit.get('p')}"
        )

    print("\n  energised-by-t3 rate per version:")
    for name in groups:
        subset = [r for r in enriched if r["group"] == name]
        if subset:
            print(
                f"    {name:<6} {champ.fmt(champ.mean(r['energised_by_own_turn_3'] for r in subset), 3)}"
                f"   early AB/game {champ.fmt(champ.mean(r['early'] for r in subset), 2)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"per_version": per_version, "per_family": per_family,
                    "blocked": blocked},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
