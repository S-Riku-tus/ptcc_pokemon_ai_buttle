"""Decision-level audit for Grimmsnarl attack continuity.

The aggregate Alakazam/second-player gap is an Active-position gap, not an
energy-supply or manual-attachment-rate gap.  This script walks the public
same-deck archive and one or more submitted-agent runs and measures the two
decisions that can create that shape:

* where each Punk Up Darkness Energy is allocated; and
* whether a ready Benched Grimmsnarl ex is promoted when the Active cannot
  attack.

No model is involved.  Every metric is read from the action actually stored in
the replay, so the report distinguishes a policy target from an unavailable
option.  The same collector is used for field and submitted-agent cohorts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(
    0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v19")
)

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_alakazam_stage1 import (  # noqa: E402
    OUR_DECK_HASH,
    cohort_of,
    replay_meta,
)


def _chosen(action: Any, size: int) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
        and 0 <= action[0] < size
    ):
        return int(action[0])
    return None


def _sides(current: dict[str, Any], seat: int) -> tuple[dict[str, Any], dict[str, Any]]:
    players = current.get("players") or [{}, {}]
    return players[seat], players[1 - seat]


def _ready(card: dict[str, Any] | None) -> bool:
    return bool(
        card
        and int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
        and mf._dark_energy_count(card) >= mf.SHADOW_BULLET_COST
    )


def _resolved(
    current: dict[str, Any], select: dict[str, Any], option: dict[str, Any]
) -> tuple[dict[str, Any], bool, int]:
    card, own, area = mf.resolve_option(current, select, option)
    return card or {}, bool(own), int(area)


def walk(replay: dict[str, Any], seat: int, max_turn: int) -> dict[str, Any]:
    """Extract route opportunities and stored choices from one trajectory."""
    punk: list[dict[str, Any]] = []
    route: list[dict[str, Any]] = []
    evolves: list[dict[str, Any]] = []
    opening: list[dict[str, Any]] = []
    pending_own_retreat = False
    steps = replay.get("steps") or []

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        turn = int(current.get("turn", -1))
        if max_turn and turn > max_turn:
            break
        options = list(select.get("option") or [])
        picked = _chosen((steps[index + 1][seat] or {}).get("action"), len(options))
        if picked is None:
            continue
        me, _opponent = _sides(current, seat)
        active = (mf._cards(me, "active") or [{}])[0]
        bench = mf._cards(me, "bench")
        context = int(select.get("context", -1))
        effect = select.get("effect") or {}
        effect_id = mf._first_nested_id(effect)

        if turn == 0 and context == 1:
            chosen_card, own, area = _resolved(current, select, options[picked])
            opening.append({
                "chosen_id": int(chosen_card.get("id", -1)),
                "owner_is_self": own,
                "area": area,
            })

        if context == mf.MAIN_CONTEXT:
            kinds = [mf.action_type(current, option, select) for option in options]
            chosen_kind = kinds[picked]
            ready_bench = [card for card in bench if _ready(card)]
            active_ready = _ready(active)
            retreat_slots = [slot for slot, kind in enumerate(kinds) if kind == "retreat"]
            attack_slots = [slot for slot, kind in enumerate(kinds) if kind == "attack"]
            if ready_bench and not active_ready:
                route.append({
                    "turn": turn,
                    "kind": "ready_bench_main",
                    "retreat_offered": bool(retreat_slots),
                    "retreat_taken": chosen_kind == "retreat",
                    "attack_offered": bool(attack_slots),
                    "chosen_kind": chosen_kind,
                    "active_id": int(active.get("id", -1)),
                    "active_energy": mf._dark_energy_count(active),
                    "ready_bench": len(ready_bench),
                })
            pending_own_retreat = chosen_kind == "retreat"

            grim_evolve_slots = []
            for slot, option in enumerate(options):
                if kinds[slot] != "evolve":
                    continue
                card = mf.candidate_card(current, option, select) or {}
                if int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID:
                    grim_evolve_slots.append(slot)
            if picked in grim_evolve_slots:
                target = mf.candidate_target(current, options[picked]) or {}
                area = mf._int(options[picked].get("inPlayArea"))
                offered_areas = {
                    mf._int(options[slot].get("inPlayArea"))
                    for slot in grim_evolve_slots
                }
                evolves.append({
                    "turn": turn,
                    "target_area": area,
                    "target_is_active": area == mf.AREA_ACTIVE,
                    "active_target_offered": mf.AREA_ACTIVE in offered_areas,
                    "bench_target_offered": mf.AREA_BENCH in offered_areas,
                    "both_areas_offered": {
                        mf.AREA_ACTIVE, mf.AREA_BENCH
                    }.issubset(offered_areas),
                    "target_id": int(target.get("id", -1)),
                    "active_id": int(active.get("id", -1)),
                    "active_energy": mf._dark_energy_count(active),
                })

        if context in (mf.CTX_SWITCH, mf.CTX_TO_ACTIVE):
            chosen_card, own, area = _resolved(current, select, options[picked])
            # A self-owned TO_ACTIVE immediately after our retreat is the
            # promotion decision controlled by the attack-access route.
            if own and area == mf.AREA_BENCH:
                offered = [
                    _resolved(current, select, option)
                    for option in options
                ]
                ready_slots = [
                    slot for slot, (card, is_own, offered_area) in enumerate(offered)
                    if is_own and offered_area == mf.AREA_BENCH and _ready(card)
                ]
                if ready_slots:
                    route.append({
                        "turn": turn,
                        "kind": "promotion",
                        "after_own_retreat": pending_own_retreat,
                        "ready_offered": len(ready_slots),
                        "ready_taken": picked in ready_slots,
                        "chosen_id": int(chosen_card.get("id", -1)),
                        "chosen_energy": mf._dark_energy_count(chosen_card),
                    })
                pending_own_retreat = False

        if context == mf.CTX_ATTACH_FROM and effect_id == mf.GRIMMSNARL_EX_ID:
            chosen_card, own, area = _resolved(current, select, options[picked])
            trigger_serial = mf._int(effect.get("serial"), -3)
            offered = [
                _resolved(current, select, option)
                for option in options
            ]
            trigger_slots = [
                slot for slot, (card, is_own, _area) in enumerate(offered)
                if is_own and mf._int(card.get("serial"), -2) == trigger_serial
            ]
            active_slots = [
                slot for slot, (_card, is_own, offered_area) in enumerate(offered)
                if is_own and offered_area == mf.AREA_ACTIVE
            ]
            underfunded_slots = [
                slot for slot, (card, is_own, _area) in enumerate(offered)
                if is_own and int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
                and mf._dark_energy_count(card) < mf.SHADOW_BULLET_COST
            ]
            active_ready_slots = [
                slot for slot, (card, is_own, offered_area) in enumerate(offered)
                if is_own and offered_area == mf.AREA_ACTIVE and _ready(card)
            ]
            future_bench_slots = [
                slot for slot, (card, is_own, offered_area) in enumerate(offered)
                if is_own and offered_area == mf.AREA_BENCH
                and int(card.get("id", -1)) in (
                    mf.IMPIDIMP_ID, mf.MORGREM_ID, mf.GRIMMSNARL_EX_ID
                )
                and not _ready(card)
            ]
            punk.append({
                "turn": turn,
                "step": index,
                "chosen_id": int(chosen_card.get("id", -1)),
                "chosen_area": area,
                "chosen_is_active": area == mf.AREA_ACTIVE,
                "chosen_is_trigger": picked in trigger_slots,
                "chosen_energy_before": mf._dark_energy_count(chosen_card),
                "trigger_offered": bool(trigger_slots),
                "trigger_taken": picked in trigger_slots,
                "active_offered": bool(active_slots),
                "active_taken": picked in active_slots,
                "underfunded_grim_offered": bool(underfunded_slots),
                "underfunded_grim_taken": picked in underfunded_slots,
                "active_ready_offered": bool(active_ready_slots),
                "active_ready_taken": picked in active_ready_slots,
                "future_bench_line_offered": bool(future_bench_slots),
                "future_bench_line_taken": picked in future_bench_slots,
                "ready_active_vs_future_prompt": bool(
                    active_ready_slots and future_bench_slots
                ),
            })

    return {
        "punk": punk,
        "route": route,
        "evolves": evolves,
        "opening": opening,
    }


def _load_replay(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def field_games(
    data_root: Path,
    allow_episodes: set[int] | None = None,
) -> Iterable[tuple[str, dict[str, Any], int, dict[str, Any], float | None]]:
    ratings: dict[int, float] = {}
    for row in csv.DictReader(
        (data_root / "indexes" / "submissions.csv").open(encoding="utf-8-sig")
    ):
        try:
            ratings[int(row["team_id"])] = float(row["submission_score"])
        except (KeyError, TypeError, ValueError):
            continue
    seen: set[tuple[int, int]] = set()
    for row in csv.DictReader(
        (data_root / "indexes" / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if (
            row.get("download_status") != "success"
            or row.get("deck_hash") != OUR_DECK_HASH
            or row.get("episode_type") != "EPISODE_TYPE_PUBLIC"
        ):
            continue
        episode, seat = int(row["episode_id"]), int(row["seat_index"])
        if allow_episodes is not None and episode not in allow_episodes:
            continue
        if (episode, seat) in seen:
            continue
        seen.add((episode, seat))
        replay = _load_replay(data_root / "replays" / f"episode_{episode}.json")
        if replay is None:
            continue
        meta = replay_meta(replay, seat)
        if meta is None:
            continue
        meta = dict(meta)
        meta["episode_id"] = episode
        cohort = cohort_of(meta)
        if cohort is None:
            continue
        team = int(row["team_id"])
        yield cohort, replay, seat, meta, ratings.get(team)


def run_games(run_dir: Path, submission: str) -> Iterable[tuple[str, dict[str, Any], int, dict[str, Any], None]]:
    for row in csv.DictReader((run_dir / "episodes.csv").open(encoding="utf-8-sig")):
        a0, a1 = row["agent_0_submission_id"], row["agent_1_submission_id"]
        if row.get("episode_type") != "EPISODE_TYPE_PUBLIC" or a0 == a1:
            continue
        episode = int(row["episode_id"])
        seat = 0 if a0 == submission else 1
        replay = _load_replay(
            run_dir / "episodes" / str(episode) / "replay" / f"episode_{episode}.json"
        )
        if replay is None:
            continue
        meta = replay_meta(replay, seat)
        if meta is None:
            continue
        meta = dict(meta)
        meta["episode_id"] = episode
        cohort = cohort_of(meta)
        if cohort is None:
            continue
        yield cohort, replay, seat, meta, None


def summarise(games: list[dict[str, Any]]) -> dict[str, Any]:
    punk = [row for game in games for row in game["events"]["punk"]]
    route = [row for game in games for row in game["events"]["route"]]
    evolves = [row for game in games for row in game["events"]["evolves"]]
    opening = [row for game in games for row in game["events"]["opening"]]
    ready_main = [row for row in route if row["kind"] == "ready_bench_main"]
    promotion = [row for row in route if row["kind"] == "promotion"]

    def rate(rows: list[dict[str, Any]], key: str) -> float | None:
        return round(sum(int(row[key]) for row in rows) / len(rows), 4) if rows else None

    return {
        "games": len(games),
        "wins": sum(int(game["won"]) for game in games),
        "opening_active_ids": dict(Counter(row["chosen_id"] for row in opening)),
        "punk_targets": len(punk),
        "punk_target_ids": dict(Counter(row["chosen_id"] for row in punk)),
        "punk_trigger_offers": sum(int(row["trigger_offered"]) for row in punk),
        "punk_trigger_take_rate": rate(
            [row for row in punk if row["trigger_offered"]], "trigger_taken"
        ),
        "punk_active_offers": sum(int(row["active_offered"]) for row in punk),
        "punk_active_take_rate": rate(
            [row for row in punk if row["active_offered"]], "active_taken"
        ),
        "punk_underfunded_grim_offers": sum(
            int(row["underfunded_grim_offered"]) for row in punk
        ),
        "punk_underfunded_grim_take_rate": rate(
            [row for row in punk if row["underfunded_grim_offered"]],
            "underfunded_grim_taken",
        ),
        "punk_ready_active_vs_future_prompts": sum(
            int(row["ready_active_vs_future_prompt"]) for row in punk
        ),
        "punk_future_line_take_when_ready_active": rate(
            [row for row in punk if row["ready_active_vs_future_prompt"]],
            "future_bench_line_taken",
        ),
        "ready_bench_main_opportunities": len(ready_main),
        "retreat_offered_rate": rate(ready_main, "retreat_offered"),
        "retreat_taken_when_offered": rate(
            [row for row in ready_main if row["retreat_offered"]], "retreat_taken"
        ),
        "ready_promotion_prompts": len(promotion),
        "ready_promotion_take_rate": rate(promotion, "ready_taken"),
        "grim_evolves": len(evolves),
        "grim_evolve_active_rate": rate(evolves, "target_is_active"),
        "grim_evolve_bench_offers": sum(
            int(row["bench_target_offered"]) for row in evolves
        ),
        "grim_evolve_active_when_bench_offered": rate(
            [row for row in evolves if row["bench_target_offered"]],
            "target_is_active",
        ),
        "grim_evolve_both_area_prompts": sum(
            int(row["both_areas_offered"]) for row in evolves
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument(
        "--field-episodes-from", type=Path,
        help=(
            "Optional JSONL decision file used only as an episode allow-list. "
            "This avoids opening every replay in a large archive."
        ),
    )
    parser.add_argument("--max-turn", type=int, default=12)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    allow_episodes: set[int] | None = None
    if args.field_episodes_from is not None:
        allow_episodes = set()
        with args.field_episodes_from.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("cohort") == "alakazam_second":
                    allow_episodes.add(int(row["episode_id"]))

    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cohort, replay, seat, meta, rating in field_games(
        args.data_root, allow_episodes
    ):
        if cohort != "alakazam_second":
            continue
        label = "field_alakazam_second"
        payload = {
            "episode_id": int(meta.get("episode_id", -1)),
            "won": bool(meta["won"]),
            "rating": rating,
            "events": walk(replay, seat, args.max_turn),
        }
        cohorts[label].append(payload)
        cohorts[f"{label}_{'won' if meta['won'] else 'lost'}"].append(payload)
        if rating is not None and rating >= 1100:
            cohorts[f"{label}_elite"].append(payload)

    for spec in args.run:
        name, path, submission = spec.split("=", 2)
        for cohort, replay, seat, meta, _rating in run_games(Path(path), submission):
            if cohort != "alakazam_second":
                continue
            payload = {
                "episode_id": int(meta["episode_id"]),
                "won": bool(meta["won"]),
                "rating": None,
                "events": walk(replay, seat, args.max_turn),
            }
            cohorts[f"{name}_alakazam_second"].append(payload)
            cohorts["ours_alakazam_second"].append(payload)
            cohorts[f"ours_alakazam_second_{'won' if meta['won'] else 'lost'}"].append(payload)

    ours_findings = []
    for game in cohorts.get("ours_alakazam_second", []):
        for row in game["events"]["route"]:
            if (
                row["kind"] == "promotion" and not row["ready_taken"]
            ) or (
                row["kind"] == "ready_bench_main"
                and row["retreat_offered"] and not row["retreat_taken"]
            ):
                ours_findings.append({
                    "episode_id": game["episode_id"],
                    "won": game["won"],
                    **row,
                })
    report = {
        "max_turn": args.max_turn,
        "cohorts": {
            name: summarise(games) for name, games in sorted(cohorts.items())
        },
        "ours_route_misses": ours_findings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
