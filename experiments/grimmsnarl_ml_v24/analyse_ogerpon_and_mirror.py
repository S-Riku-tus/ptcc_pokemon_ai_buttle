"""The two cells that own the >=950 deficit: the mirror (0.452) and Ogerpon (0.000).

Memory splits Ogerpon into two different problems, and the fix is opposite in
each: Teal Mask Ogerpon ex races us through Grass weakness and is ~0.20 for the
whole field, while Cornerstone Mask Ogerpon ex is an Ability-blocker wall that
only Marnie's Morgrem can punch through.  Treating them as one archetype hides
whichever one is actually costing games.

This lists our Ogerpon games by opponent deck hash, and for every game counts
attacks thrown while the opposing Active was immune to Grimmsnarl ex with no
Bench prize available - the dead-swing signature from v15 episode 91663479.

The same walker reports the mirror's terminal states so a mirror loss that is
a race can be told apart from one that is a board-out.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts", ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402
from fallback_policy import EX_ACTIVE_BLOCKERS  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
RUNS = ROOT / "data/runs/grimmsnarl"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/ogerpon_mirror.json"
MORGREM_ID = 647


def replay_index() -> dict[str, tuple[Path, int]]:
    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (
                    run_dir, int(row["detected_submission_agent_index"]))
    return index


def dead_swings(replay: dict[str, Any], seat: int) -> dict[str, int]:
    """Attacks taken into a board where Grimmsnarl ex can score nothing."""
    counts = Counter({
        "our_attacks": 0,
        "attacks_into_immune_active": 0,
        "immune_active_empty_bench": 0,
        "morgrem_on_board_when_immune": 0,
        "morgrem_attacks": 0,
        "turns_with_immune_active": 0,
    })
    seen_turns: set[int] = set()
    steps = replay.get("steps") or []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        obs = record.get("observation") or {}
        select, current = obs.get("select"), obs.get("current")
        if not isinstance(select, dict) or not isinstance(current, dict):
            continue
        players = current.get("players") or []
        if len(players) < 2:
            continue
        opponent = players[1 - seat]
        us = players[seat]
        actives = mf._cards(opponent, "active")
        if not actives:
            continue
        active_id = int(actives[0].get("id", -1))
        immune = active_id in EX_ACTIVE_BLOCKERS
        bench = mf._cards(opponent, "bench")
        our_bodies = mf._cards(us, "active") + mf._cards(us, "bench")
        have_morgrem = any(int(c.get("id", -1)) == MORGREM_ID for c in our_bodies)
        turn = int(current.get("turn", -1))
        if immune and turn not in seen_turns:
            seen_turns.add(turn)
            counts["turns_with_immune_active"] += 1
            counts["morgrem_on_board_when_immune"] += int(have_morgrem)
            counts["immune_active_empty_bench"] += int(not bench)

        options = list(select.get("option") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        chosen = action[0] if isinstance(action, list) and len(action) == 1 else None
        if not isinstance(chosen, int) or not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        try:
            kind = mf.action_type(current, option, select)
            card = mf.candidate_card(current, option, select) or {}
        except Exception:  # noqa: BLE001
            continue
        if kind != "attack":
            continue
        counts["our_attacks"] += 1
        if int(card.get("id", -1)) == MORGREM_ID:
            counts["morgrem_attacks"] += 1
        elif immune:
            counts["attacks_into_immune_active"] += 1
    return dict(counts)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    index = replay_index()
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith(("v22", "v24")):
            continue
        family = raw["opponent_family"] or ""
        if "Ogerpon" not in family and "mirror" not in family:
            continue
        entry = index.get(raw["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (run_dir / "episodes" / raw["episode_id"] / "replay"
                / f"episode_{raw['episode_id']}.json")
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "episode_id": raw["episode_id"],
            "version": raw["version"],
            "family": family,
            "deck_hash": raw["opponent_deck_hash"],
            "won": raw["won"] == "True",
            "opponent_rating": float(raw["opponent_rating"]) if raw["opponent_rating"] else None,
            "turns": int(float(raw["turns"] or 0)),
            "our_prizes_taken": 6 - int(float(raw["our_prize_left"] or 6)),
            "opp_prizes_taken": 6 - int(float(raw["opp_prize_left"] or 6)),
            "our_bodies_left": int(float(raw["our_bodies_left"] or 0)),
            "our_deck_left": int(float(raw["our_deck_left"] or 0)),
            **dead_swings(replay, seat),
        })

    ogerpon = [r for r in rows if "Ogerpon" in r["family"]]
    mirror = [r for r in rows if "mirror" in r["family"]]

    print(f"EX_ACTIVE_BLOCKERS in the shipped v22 policy: {sorted(EX_ACTIVE_BLOCKERS)}\n")

    print("=== Ogerpon games, by opponent deck hash ===")
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in ogerpon:
        by_hash[row["deck_hash"]].append(row)
    for h, items in sorted(by_hash.items(), key=lambda kv: -len(kv[1])):
        wins = sum(1 for r in items if r["won"])
        immune_turns = sum(r["turns_with_immune_active"] for r in items)
        print(f"  {h}  n={len(items):>2}  {wins}-{len(items) - wins}  "
              f"opp mean {sum(r['opponent_rating'] or 0 for r in items) / len(items):.0f}  "
              f"turns-with-immune-active {immune_turns}")
    print()

    print("=== every Ogerpon game ===")
    print(f"{'episode':<10}{'ver':<7}{'W':<2}{'opp':>6}{'turns':>6}"
          f"{'we':>4}{'they':>5}{'atk':>5}{'dead':>6}{'morg':>6}"
          f"{'immT':>6}{'morgB':>7}{'emptyB':>8}  deck")
    for r in sorted(ogerpon, key=lambda r: r["deck_hash"]):
        print(
            f"{r['episode_id']:<10}{r['version']:<7}"
            f"{'W' if r['won'] else 'L':<2}{r['opponent_rating'] or 0:>6.0f}"
            f"{r['turns']:>6}{r['our_prizes_taken']:>4}{r['opp_prizes_taken']:>5}"
            f"{r['our_attacks']:>5}{r['attacks_into_immune_active']:>6}"
            f"{r['morgrem_attacks']:>6}{r['turns_with_immune_active']:>6}"
            f"{r['morgrem_on_board_when_immune']:>7}"
            f"{r['immune_active_empty_bench']:>8}  {r['deck_hash']}")
    print()

    strong_mirror = [r for r in mirror
                     if (r["opponent_rating"] or 0) >= 950]
    print(f"=== mirror, opponents >= 950 (n={len(strong_mirror)}) ===")
    for label, group in (
        ("win", [r for r in strong_mirror if r["won"]]),
        ("loss", [r for r in strong_mirror if not r["won"]]),
    ):
        if not group:
            continue
        n = len(group)
        print(
            f"  {label:<5} n={n:>3}  turns {sum(r['turns'] for r in group) / n:5.2f}  "
            f"prizes we took {sum(r['our_prizes_taken'] for r in group) / n:4.2f}  "
            f"they took {sum(r['opp_prizes_taken'] for r in group) / n:4.2f}  "
            f"bodies left {sum(r['our_bodies_left'] for r in group) / n:4.2f}  "
            f"deck left {sum(r['our_deck_left'] for r in group) / n:5.2f}  "
            f"attacks {sum(r['our_attacks'] for r in group) / n:5.2f}"
        )
    print()
    print("=== mirror losses at >=950, by prizes we took ===")
    counter = Counter(r["our_prizes_taken"] for r in strong_mirror if not r["won"])
    for taken in sorted(counter):
        print(f"  took {taken}: {counter[taken]}")

    OUT.write_text(json.dumps(
        {"ogerpon": ogerpon, "mirror": mirror}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
