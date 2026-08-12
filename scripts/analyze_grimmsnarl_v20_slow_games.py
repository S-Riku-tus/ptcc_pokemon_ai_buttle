"""Why do 6.8% of games never assemble a Grimmsnarl ex?

The ladder history says the first Shadow Bullet is the whole race: 0.704 when
it lands on our own turn 2, 0.139 when it lands on turn 6 or never, and 0/20
when it never lands at all.  That gradient is only actionable if the slow games
are *refusals* rather than *bricks*, so this splits them:

* **offered and refused** - the Impidimp bench, the Morgrem/Grimmsnarl evolve
  or the Rare Candy was a legal MAIN option on that turn and we played
  something else.  That is a policy defect and a ranker can fix it.
* **never offered** - the line was not in hand on any turn of the game.  That
  is deck variance and no policy change reaches it.

Offers are collapsed per own turn (a card that stays legal across five
decisions is one offer), matching the denominator used by the rest of this
line.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v20"))

import ml_features as mf  # noqa: E402

from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"

TRACKED = {
    mf.IMPIDIMP_ID: "impidimp",
    mf.MORGREM_ID: "morgrem",
    mf.GRIMMSNARL_EX_ID: "grimmsnarl_ex",
    mf.RARE_CANDY_ID: "rare_candy",
    mf.POFFIN_ID: "poffin",
    mf.NIGHT_STRETCHER_ID: "night_stretcher",
    mf.DARK_ENERGY_ID: "dark_energy",
}


def scan(replay: dict[str, Any], seat: int) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    if len(steps) < 2:
        return None
    action = (steps[1][seat] or {}).get("action")
    if not (isinstance(action, list) and len(action) == 60):
        return None
    if deck_hash([int(v) for v in action]) != OUR_DECK_HASH:
        return None

    offers: dict[str, set[int]] = defaultdict(set)
    takes: dict[str, set[int]] = defaultdict(set)
    first_shadow: int | None = None
    grim_on_board_turn: int | None = None
    turns: set[int] = set()

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
            continue
        options = list(select.get("option") or [])
        current = observation.get("current") or {}
        players = current.get("players") or []
        if not options or len(players) < 2:
            continue
        turn = int(current.get("turn", -1))
        turns.add(turn)
        me = players[seat]
        if grim_on_board_turn is None and any(
            int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            for card in mf._cards(me, "active") + mf._cards(me, "bench")
        ):
            grim_on_board_turn = turn

        raw = (steps[index + 1][seat] or {}).get("action")
        picked = {
            int(value) for value in raw
            if isinstance(value, int) and 0 <= int(value) < len(options)
        } if isinstance(raw, list) else set()

        for position, option in enumerate(options):
            try:
                kind = mf.action_type(current, option, select)
            except Exception:  # noqa: BLE001
                continue
            if kind == "attack":
                if (
                    position in picked
                    and mf._int(option.get("attackId")) == mf.SHADOW_BULLET_ID
                    and first_shadow is None
                ):
                    first_shadow = turn
                continue
            card = mf.candidate_card(current, option, select) or {}
            name = TRACKED.get(int(card.get("id", -1)))
            if name is None:
                continue
            key = f"{name}:{kind}"
            offers[key].add(turn)
            if position in picked:
                takes[key].add(turn)

    return {
        "first_shadow_turn": first_shadow,
        "grim_on_board_turn": grim_on_board_turn,
        "own_main_turns": len(turns),
        "offers": {key: sorted(value) for key, value in offers.items()},
        "takes": {key: sorted(value) for key, value in takes.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20"
        / "ladder_history_games.csv",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "data" / "runs" / "grimmsnarl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20" / "slow_games.json",
    )
    args = parser.parse_args()

    run_for_episode: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(args.runs.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not manifest.is_dir() and manifest.exists():
            for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
                seat = row.get("detected_submission_agent_index", "")
                if seat in {"0", "1"}:
                    run_for_episode[row["episode_id"]] = (run_dir, int(seat))

    rows = list(
        csv.DictReader(args.games.open(encoding="utf-8-sig"))
    )
    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        shadow = row["own_first_shadow_turn"]
        slow = shadow in {"", "None"} or int(shadow) >= 5
        cohorts["slow" if slow else "normal"].append(row)

    summary: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    for cohort, members in cohorts.items():
        offered = Counter()
        taken = Counter()
        games = 0
        never_offered = Counter()
        for row in members:
            entry = run_for_episode.get(row["episode_id"])
            if entry is None:
                continue
            run_dir, seat = entry
            path = (
                run_dir / "episodes" / row["episode_id"] / "replay"
                / f"episode_{row['episode_id']}.json"
            )
            if not path.exists():
                continue
            result = scan(json.loads(path.read_text(encoding="utf-8")), seat)
            if result is None:
                continue
            games += 1
            for key, turns in result["offers"].items():
                offered[key] += len(turns)
            for key, turns in result["takes"].items():
                taken[key] += len(turns)
            for key in (
                "grimmsnarl_ex:evolve",
                "rare_candy:item",
                "impidimp:bench",
                "morgrem:evolve",
            ):
                if not result["offers"].get(key):
                    never_offered[key] += 1
            if cohort == "slow":
                details.append(
                    {
                        "version": row["version"],
                        "episode_id": row["episode_id"],
                        "opponent": row["opponent_family"],
                        "won": row["won"],
                        "own_main_turns": result["own_main_turns"],
                        "grim_on_board_turn": result["grim_on_board_turn"],
                        "offers": {
                            key: value
                            for key, value in result["offers"].items()
                            if key.split(":")[0]
                            in {
                                "grimmsnarl_ex", "rare_candy",
                                "impidimp", "morgrem",
                            }
                        },
                        "takes": {
                            key: value
                            for key, value in result["takes"].items()
                            if key.split(":")[0]
                            in {
                                "grimmsnarl_ex", "rare_candy",
                                "impidimp", "morgrem",
                            }
                        },
                    }
                )
        summary[cohort] = {
            "games": games,
            "offer_turns": dict(offered.most_common()),
            "take_turns": dict(taken.most_common()),
            "take_rate": {
                key: round(taken[key] / value, 4)
                for key, value in offered.items()
                if value >= 10
            },
            "games_with_no_offer": dict(never_offered),
        }

    payload = {"summary": summary, "slow_games": details}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for cohort in ("normal", "slow"):
        info = summary.get(cohort)
        if not info:
            continue
        print(f"=== {cohort}: {info['games']} games")
        for key in sorted(info["take_rate"], key=lambda k: -info["offer_turns"][k]):
            print(
                f"  {key:26s} offered {info['offer_turns'][key]:5d} turns  "
                f"taken {info['take_turns'].get(key, 0):5d}  "
                f"rate {info['take_rate'][key]:.4f}"
            )
        print("  games where the line was never offered at all:")
        for key, value in sorted(info["games_with_no_offer"].items()):
            print(
                f"    {key:26s} {value:3d}/{info['games']} "
                f"= {value / info['games']:.3f}"
            )
        print()
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
