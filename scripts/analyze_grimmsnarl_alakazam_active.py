"""Why the attack is not on offer: the Active's state at the start of each turn.

The offer/take split says the whole tempo gap is in the *offer* - going second
into Alakazam the field is offered an attack on 67.4% of its own turns and takes
it 98% of the time, and we are offered one on 50.0% and take it 100% of the
time. So there is nothing to fix at the attack node. The question moves one step
back: an attack is on offer when the Active has the energy for it, so what is
the Active, and where did the energy go?

Three things are measured at the *first* MAIN decision of each own turn, which
is the board we inherited rather than the one we built:

* what the Active is, how much energy is on it, and whether an attack ends up
  offered anywhere in the turn
* how the energy in play is split between the Active and the bench - the same
  attach rate can produce a fuelled attacker or a fuelled bench
* whether the Active changed since our last turn, and why: promoted after a
  knock-out, retreated, or switched by an effect

The knock-out split is what separates the two readings the outcome data cannot.
If our Active is naked because it keeps being promoted after a knock-out, the
attack drought is a *consequence* of losing the race and no attach-side change
fixes it. If our Active is naked while the energy sits on the bench, it is a
targeting bug and it is directly actionable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_alakazam_stage1 import (  # noqa: E402
    OUR_DECK_HASH, cohort_of, replay_meta,
)

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
MAIN = mf.MAIN_CONTEXT


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not n or not r1 or not c1 or r1 == n or c1 == n:
        return 1.0

    def probability(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    reference = probability(a)
    return round(min(1.0, sum(
        probability(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1)
        if probability(x) <= reference + 1e-12
    )), 4)


def energy_count(body: dict[str, Any]) -> int:
    """Attached energy. Bodies carry ``energies`` (types) and ``energyCards``."""
    for key in ("energies", "energyCards"):
        value = body.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def name_of(body: dict[str, Any] | None) -> str:
    if not body:
        return "-"
    return CARDS.get(int(body.get("id", -1)), {}).get("name", "?")


def turn_records(
    replay: dict[str, Any], seat: int, max_turn: int
) -> list[dict[str, Any]]:
    steps = replay.get("steps") or []
    records: dict[int, dict[str, Any]] = {}
    previous_active: tuple[int, int] | None = None
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != MAIN:
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        options = list(select.get("option") or [])
        if not options or len(players) < 2:
            continue
        turn = int(current.get("turn", -1))
        if turn > max_turn:
            break
        me = players[seat]
        bodies = mf._in_play(me)
        if not bodies:
            continue
        active, bench = bodies[0], bodies[1:]
        entry = records.get(turn)
        if entry is None:
            key = (int(active.get("id", -1)), int(active.get("serial", -1)))
            entry = records[turn] = {
                "turn": turn,
                "active": name_of(active),
                "active_id": int(active.get("id", -1)),
                "active_energy": energy_count(active),
                "bench_energy": sum(energy_count(b) for b in bench),
                "bench_bodies": len(bench),
                "hand": len(me.get("hand") or []),
                "deck": len(me.get("deck") or []),
                "prizes_left": len(me.get("prize") or []),
                "opponent_prizes_left": len(
                    players[1 - seat].get("prize") or []
                ),
                "active_changed": (
                    previous_active is not None and key != previous_active
                ),
                "attack_offered": False,
                "attached_to_active": 0,
                "attached_to_bench": 0,
            }
            previous_active = key
        try:
            kinds = [mf.action_type(current, o, select) for o in options]
        except Exception:  # noqa: BLE001
            kinds = []
        if "attack" in kinds:
            entry["attack_offered"] = True
        # Where did this turn's attaches land? Read from the option the replay
        # took, so a bench attach and an Active attach are told apart.
        action = (steps[index + 1][seat] or {}).get("action")
        if (
            kinds and isinstance(action, list) and len(action) == 1
            and isinstance(action[0], int) and 0 <= action[0] < len(options)
            and kinds[action[0]] == "energy"
        ):
            target = mf.candidate_target(current, options[action[0]]) or {}
            serial = int(target.get("serial", -2))
            if serial == int(active.get("serial", -1)):
                entry["attached_to_active"] += 1
            elif serial >= 0:
                entry["attached_to_bench"] += 1
    return [records[t] for t in sorted(records)]


def summarise(games: list[list[dict[str, Any]]]) -> dict[str, Any]:
    flat = [row for game in games for row in game]
    if not flat:
        return {"games": len(games), "own_turns": 0}

    def mean(key: str, rows: list[dict[str, Any]]) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    by_turn = {}
    for turn in sorted({r["turn"] for r in flat}):
        rows = [r for r in flat if r["turn"] == turn]
        by_turn[str(turn)] = {
            "n": len(rows),
            "attack_offered": round(
                sum(int(r["attack_offered"]) for r in rows) / len(rows), 4
            ),
            "active_energy": mean("active_energy", rows),
            "bench_energy": mean("bench_energy", rows),
            "hand": mean("hand", rows),
            "active_changed": round(
                sum(int(r["active_changed"]) for r in rows) / len(rows), 4
            ),
            "active_is_grimmsnarl": round(sum(
                int(r["active_id"] == mf.GRIMMSNARL_EX_ID) for r in rows
            ) / len(rows), 4),
            "attached_to_active": mean("attached_to_active", rows),
            "attached_to_bench": mean("attached_to_bench", rows),
        }
    return {
        "games": len(games),
        "own_turns": len(flat),
        "attack_offered": round(
            sum(int(r["attack_offered"]) for r in flat) / len(flat), 4
        ),
        "no_energy_on_active": round(
            sum(int(r["active_energy"] == 0) for r in flat) / len(flat), 4
        ),
        "active_changed": round(
            sum(int(r["active_changed"]) for r in flat) / len(flat), 4
        ),
        "attaches_to_active": sum(r["attached_to_active"] for r in flat),
        "attaches_to_bench": sum(r["attached_to_bench"] for r in flat),
        "attach_to_active_share": (
            round(
                sum(r["attached_to_active"] for r in flat)
                / max(1, sum(r["attached_to_active"] + r["attached_to_bench"]
                             for r in flat)), 4
            )
        ),
        "by_turn": by_turn,
        "active_when_no_attack": dict(Counter(
            r["active"] for r in flat if not r["attack_offered"]
        ).most_common(8)),
    }


def collect(
    data_root: Path, runs: list[tuple[Path, str]], max_turn: int
) -> dict[str, list[list[dict[str, Any]]]]:
    cohorts: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for raw in csv.DictReader(
        (data_root / "indexes" / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id, seat = int(raw["episode_id"]), int(raw["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        meta = replay_meta(replay, seat)
        if meta is None:
            continue
        cohort = cohort_of(meta)
        if cohort is None:
            continue
        rows = turn_records(replay, seat, max_turn)
        if not rows:
            continue
        cohorts[cohort].append(rows)
        if cohort == "alakazam_second":
            cohorts["alakazam_second_won" if meta["won"]
                    else "alakazam_second_lost"].append(rows)

    for run_dir, submission in runs:
        for raw in csv.DictReader(
            (run_dir / "episodes.csv").open(encoding="utf-8-sig")
        ):
            a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
            if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
                continue
            episode_id = int(raw["episode_id"])
            path = (
                run_dir / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not path.exists():
                continue
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            meta = replay_meta(replay, 0 if a0 == submission else 1)
            if meta is None:
                continue
            cohort = cohort_of(meta)
            if cohort is None:
                continue
            rows = turn_records(
                replay, 0 if a0 == submission else 1, max_turn
            )
            if rows:
                cohorts[f"ours_{cohort}"].append(rows)
    return cohorts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--max-turn", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not args.run:
        base = ROOT / "data" / "submissions"
        args.run = [
            f"{base / 'submission_55317804'}=55317804",
            f"{base / 'submission_55302846'}=55302846",
        ]
    runs = [
        (Path(spec.rsplit("=", 1)[0]), spec.rsplit("=", 1)[1])
        for spec in args.run
    ]

    cohorts = collect(args.data_root, runs, args.max_turn)
    report = {
        "max_turn": args.max_turn,
        "cohorts": {
            name: summarise(games) for name, games in sorted(cohorts.items())
        },
    }

    us = cohorts.get("ours_alakazam_second") or []
    them = cohorts.get("alakazam_second_lost") or []
    flat_us = [r for g in us for r in g]
    flat_them = [r for g in them for r in g]
    if flat_us and flat_them:
        a = sum(int(r["attack_offered"]) for r in flat_us)
        c = sum(int(r["attack_offered"]) for r in flat_them)
        report["ours_vs_field_losses"] = {
            "note": "field cohort restricted to its own losses, so the "
                    "comparison is not confounded by outcome",
            "attack_offered_fisher": fisher(
                a, len(flat_us) - a, c, len(flat_them) - c
            ),
            "ours": round(a / len(flat_us), 4),
            "field_losses": round(c / len(flat_them), 4),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    order = [
        "alakazam_second", "alakazam_second_won", "alakazam_second_lost",
        "other_second", "ours_alakazam_second", "ours_other_second",
    ]
    print(f"{'cohort':24s} {'turns':>6} {'atk off':>8} {'noNRG':>7} "
          f"{'chgAct':>7} {'toActive':>9}")
    for name in order:
        block = report["cohorts"].get(name)
        if not block or not block.get("own_turns"):
            continue
        print(f"{name:24s} {block['own_turns']:6d} "
              f"{block['attack_offered']:8.3f} "
              f"{block['no_energy_on_active']:7.3f} "
              f"{block['active_changed']:7.3f} "
              f"{block['attach_to_active_share']:9.3f}")
    print()
    for name in order:
        block = report["cohorts"].get(name)
        if not block or not block.get("own_turns"):
            continue
        print(f"{name}:")
        for turn, row in block["by_turn"].items():
            print(f"   t{turn:<3s} n={row['n']:5d} atk={row['attack_offered']:.3f} "
                  f"actNRG={row['active_energy']} benchNRG={row['bench_energy']} "
                  f"hand={row['hand']} chg={row['active_changed']:.3f} "
                  f"grimmActive={row['active_is_grimmsnarl']:.3f}")
    if "ours_vs_field_losses" in report:
        print()
        print("attack offered, ours vs the field's own losses: "
              f"{report['ours_vs_field_losses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
