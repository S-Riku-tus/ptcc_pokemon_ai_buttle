"""Compare how the live Dragapult agent and exact-list teachers play a matchup.

Win-rate cells say *where* a deficit is.  This report describes the mechanism:
board width, evolution timing, typed Energy routing, attacks, counter placement,
and the main search/recovery cards used before and after the first Phantom Dive.

The same observation-only walker is used for both cohorts.  It never imports an
agent and never reads a future state to classify a decision.

Example:
  python scripts/analyze_dragapult_matchup_strategy.py \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --run data/submissions/submission_55550682_dragapult_v2 \
      --matchup "Mega Lucario ex" --matchup "Dragapult ex" \
      --report experiments/dragapult_ml_v2/matchup_strategy_v2.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}

FIRE, PSYCHIC = 2, 5
DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
LINE = {DREEPY, DRAKLOAK, DRAGAPULT}
PHANTOM_DIVE = 154
MEGA_LUCARIO = 678

OPT_CARD = 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE = 7, 8, 9
OPT_RETREAT, OPT_ATTACK = 12, 13
AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5
CTX_MAIN, CTX_COUNTER, CTX_COUNTER_ANY = 0, 13, 14

TRACKED_PLAYS = {
    1080: "unfair_stamp",
    1086: "buddy_buddy_poffin",
    1097: "night_stretcher",
    1120: "crushing_hammer",
    1121: "ultra_ball",
    1152: "poke_pad",
    1182: "boss_orders",
    1198: "crispin",
    1213: "judge",
    1227: "lillie_determination",
    1231: "dawn",
}


def _cards(owner: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = owner.get(key) or []
    if isinstance(value, dict):
        value = [value]
    return [card for card in value if isinstance(card, dict)]


def _zone(owner: dict[str, Any], area: int) -> list[dict[str, Any]]:
    if area == AREA_HAND:
        return _cards(owner, "hand")
    if area == AREA_ACTIVE:
        return _cards(owner, "active")
    if area == AREA_BENCH:
        return _cards(owner, "bench")
    return []


def _card_id(card: dict[str, Any] | None) -> int:
    return int((card or {}).get("id", -1))


def _energy(card: dict[str, Any] | None) -> list[int]:
    return [int(value) for value in ((card or {}).get("energies") or [])]


def _card_at(owner: dict[str, Any], area: int, index: int) -> dict[str, Any] | None:
    zone = _zone(owner, area)
    return zone[index] if 0 <= index < len(zone) else None


def _option_card(
    current: dict[str, Any], select: dict[str, Any], option: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    """Resolve the card represented by an option and whether it is ours."""
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    player = int(option.get("playerIndex", your))
    owner = players[player] if player in (0, 1) else {}
    area = int(option.get("area", -1))
    if int(option.get("type", -1)) in (OPT_PLAY, OPT_ATTACH, OPT_EVOLVE):
        owner, area, player = players[your], AREA_HAND, your
    if area == 1:
        zone = select.get("deck") or []
        index = int(option.get("index", -1))
        card = zone[index] if isinstance(zone, list) and 0 <= index < len(zone) else None
    else:
        card = _card_at(owner, area, int(option.get("index", -1)))
    return (card if isinstance(card, dict) else None), player == your


def _evolve_target(
    owner: dict[str, Any], option: dict[str, Any]
) -> tuple[int, dict[str, Any]] | None:
    """Resolve an evolution target, including logs that omit target fields."""
    raw_area = option.get("inPlayArea", option.get("area"))
    raw_index = option.get("inPlayIndex", option.get("targetIndex"))
    if raw_area is not None and raw_index is not None:
        area, index = int(raw_area), int(raw_index)
        target = _card_at(owner, area, index)
        if target is not None:
            return area, target
    candidates = [
        (area, card)
        for area in (AREA_ACTIVE, AREA_BENCH)
        for card in _zone(owner, area)
        if _card_id(card) == DRAKLOAK
    ]
    return candidates[0] if len(candidates) == 1 else None


def archetype(deck: list[int]) -> str:
    pokemon = Counter(
        card_id for card_id in deck if CARDS.get(card_id, {}).get("cardType") == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple[int, int, int, int, int]:
        card_id, count = item
        card = CARDS[card_id]
        return (
            int(bool(card.get("stage2"))),
            int(bool(card.get("megaEx") or card.get("ex"))),
            int(bool(card.get("stage1"))),
            count,
            int(card.get("hp") or 0),
        )

    best = max(pokemon.items(), key=key)[0]
    return str(CARDS.get(best, {}).get("name") or best)


def replay_decks(replay: dict[str, Any]) -> list[list[int]]:
    decks: list[list[int]] = [[], []]
    steps = replay.get("steps") or []
    if len(steps) < 2:
        return decks
    for seat in (0, 1):
        action = steps[1][seat].get("action")
        if isinstance(action, list) and len(action) == 60:
            decks[seat] = [int(value) for value in action]
    return decks


def deck_hash(card_ids: Iterable[int]) -> str:
    counts = Counter(int(card_id) for card_id in card_ids)
    canonical = ";".join(
        f"{card_id}:{counts[card_id]}" for card_id in sorted(counts)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _chosen(
    steps: list[Any], index: int, seat: int, select: dict[str, Any]
) -> dict[str, Any] | None:
    action = steps[index + 1][seat].get("action") if index + 1 < len(steps) else None
    options = select.get("option") or []
    if not isinstance(action, list) or len(action) != 1:
        return None
    picked = int(action[0])
    if not 0 <= picked < len(options):
        return None
    option = options[picked]
    return option if isinstance(option, dict) else None


def game_metrics(replay: dict[str, Any], seat: int) -> dict[str, Any]:
    steps = replay.get("steps") or []
    decks = replay_decks(replay)
    opponent = archetype(decks[1 - seat])
    seen_turns: set[int] = set()
    own_turn = 0
    seen_pult: set[int] = set()
    metrics: Counter[str] = Counter()
    plays: Counter[str] = Counter()
    attacks: Counter[str] = Counter()
    spread_targets: Counter[str] = Counter()
    first: dict[str, int | None] = {
        "first_dreepy_turn": None,
        "first_drakloak_turn": None,
        "first_dragapult_turn": None,
        "first_phantom_dive_turn": None,
        "second_route_body_turn": None,
    }
    final_taken = [0, 0]

    for index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) != 2:
            continue
        your = int(current.get("yourIndex", seat))
        me, opp = players[your], players[1 - your]
        turn = current.get("turn")
        if isinstance(turn, int) and turn not in seen_turns:
            seen_turns.add(turn)
            own_turn += 1
        board = _cards(me, "active") + _cards(me, "bench")
        opp_board = _cards(opp, "active") + _cards(opp, "bench")
        ids = [_card_id(card) for card in board]
        metrics["max_line_bodies"] = max(
            metrics["max_line_bodies"], sum(card_id in LINE for card_id in ids)
        )
        metrics["max_dragapult"] = max(metrics["max_dragapult"], ids.count(DRAGAPULT))
        metrics["max_bench_line"] = max(
            metrics["max_bench_line"],
            sum(_card_id(card) in LINE for card in _cards(me, "bench")),
        )
        if sum(card_id in LINE for card_id in ids) >= 2 and first["second_route_body_turn"] is None:
            first["second_route_body_turn"] = own_turn
        for card_id, key_name in (
            (DREEPY, "first_dreepy_turn"),
            (DRAKLOAK, "first_drakloak_turn"),
            (DRAGAPULT, "first_dragapult_turn"),
        ):
            if card_id in ids and first[key_name] is None:
                first[key_name] = own_turn
        for card in board:
            if _card_id(card) != DRAGAPULT:
                continue
            serial = int(card.get("serial", -1))
            if serial not in seen_pult:
                seen_pult.add(serial)
                metrics["dragapult_created"] += 1
                on_bench = card in _cards(me, "bench")
                metrics["dragapult_created_bench" if on_bench else "dragapult_created_active"] += 1
                colors = _energy(card)
                if not on_bench and not (FIRE in colors and PSYCHIC in colors):
                    metrics["dragapult_created_active_unpowered"] += 1

        metrics["max_opponent_lucario"] = max(
            metrics["max_opponent_lucario"],
            sum(_card_id(card) == MEGA_LUCARIO for card in opp_board),
        )
        final_taken = [
            6 - len(me.get("prize") or []),
            6 - len(opp.get("prize") or []),
        ]

        select = observation.get("select")
        if not isinstance(select, dict):
            continue
        option = _chosen(steps, index, seat, select)
        if option is None:
            continue
        option_type = int(option.get("type", -1))
        context = int(select.get("context", -1))
        card, ours = _option_card(current, select, option)
        card_id = _card_id(card)

        if context == CTX_MAIN and option_type == OPT_PLAY and card_id in TRACKED_PLAYS:
            key = TRACKED_PLAYS[card_id]
            plays[key] += 1
            if first["first_phantom_dive_turn"] is None:
                plays[f"{key}_before_first_dive"] += 1
        if context == CTX_MAIN and option_type == OPT_RETREAT:
            metrics["retreats"] += 1
        if context == CTX_MAIN and option_type == OPT_ATTACH:
            source_id = card_id
            target = _card_at(
                me,
                int(option.get("inPlayArea", -1)),
                int(option.get("inPlayIndex", -1)),
            )
            if source_id in (FIRE, PSYCHIC) and _card_id(target) in LINE:
                colors = _energy(target)
                if source_id in colors:
                    metrics["route_attach_duplicate"] += 1
                elif (PSYCHIC if source_id == FIRE else FIRE) in colors:
                    metrics["route_attach_completes"] += 1
                else:
                    metrics["route_attach_starts"] += 1
                if int(option.get("inPlayArea", -1)) == AREA_BENCH:
                    metrics["route_attach_to_bench"] += 1
        if context == CTX_MAIN and option_type == OPT_EVOLVE and card_id == DRAGAPULT:
            resolved = _evolve_target(me, option)
            if resolved is None:
                continue
            area, target = resolved
            powered = FIRE in _energy(target) and PSYCHIC in _energy(target)
            metrics["evolve_dragapult"] += 1
            metrics["evolve_dragapult_bench" if area == AREA_BENCH else "evolve_dragapult_active"] += 1
            if area == AREA_ACTIVE and not powered:
                metrics["evolve_dragapult_active_unpowered"] += 1
        if option_type == OPT_ATTACK:
            attack_id = int(option.get("attackId", -1))
            attacks[str(attack_id)] += 1
            metrics["attacks"] += 1
            if attack_id == PHANTOM_DIVE:
                metrics["phantom_dives"] += 1
                if first["first_phantom_dive_turn"] is None:
                    first["first_phantom_dive_turn"] = own_turn
                opp_active = _cards(opp, "active")
                active_id = _card_id(opp_active[0]) if opp_active else -1
                attacks[f"phantom_into_{active_id}"] += 1
        if context in (CTX_COUNTER, CTX_COUNTER_ANY) and option_type == OPT_CARD and not ours:
            spread_targets[str(card_id)] += 1

    rewards = replay.get("rewards") or [0, 0]
    result = "win" if rewards[seat] > rewards[1 - seat] else (
        "loss" if rewards[seat] < rewards[1 - seat] else "draw"
    )
    output: dict[str, Any] = {
        "opponent": opponent,
        "opponent_deck_hash": deck_hash(decks[1 - seat]),
        "result": result,
        "own_turns": own_turn,
        "prizes_taken": final_taken[0],
        "prizes_conceded": final_taken[1],
        **dict(metrics),
        **first,
        "plays": dict(plays),
        "attacks_used": dict(attacks),
        "spread_targets": dict(spread_targets),
    }
    return output


def _mean(games: list[dict[str, Any]], key: str) -> float | None:
    # Counters deliberately omit zero-valued keys; treating absence as missing
    # would condition every rate on the event having happened.  Turn-of-first
    # fields are the exception: a game that never reaches the milestone has no
    # meaningful turn number and stays out of that mean.
    if key.startswith("first_") or key == "second_route_body_turn":
        values = [
            float(game[key])
            for game in games if isinstance(game.get(key), (int, float))
        ]
    else:
        values = [float(game.get(key, 0)) for game in games]
    return round(statistics.mean(values), 4) if values else None


def _pooled(games: list[dict[str, Any]], key: str) -> dict[str, float]:
    total: Counter[str] = Counter()
    for game in games:
        total.update(game.get(key) or {})
    count = max(1, len(games))
    return {name: round(value / count, 4) for name, value in sorted(total.items())}


def summarise(games: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "own_turns", "prizes_taken", "prizes_conceded", "phantom_dives", "attacks",
        "max_line_bodies", "max_bench_line", "max_dragapult",
        "dragapult_created", "dragapult_created_bench", "dragapult_created_active",
        "dragapult_created_active_unpowered", "evolve_dragapult",
        "evolve_dragapult_bench", "evolve_dragapult_active",
        "evolve_dragapult_active_unpowered", "route_attach_starts",
        "route_attach_completes", "route_attach_duplicate", "route_attach_to_bench",
        "retreats", "first_dreepy_turn", "first_drakloak_turn",
        "first_dragapult_turn", "first_phantom_dive_turn", "second_route_body_turn",
    )
    wins = sum(game["result"] == "win" for game in games)
    return {
        "games": len(games),
        "wins": wins,
        "win_rate": round(wins / len(games), 4) if games else None,
        "tail_share": round(
            sum(int(game.get("phantom_dives", 0)) <= 1 for game in games) / len(games), 4
        ) if games else None,
        "means": {key: _mean(games, key) for key in scalar_keys},
        "plays_per_game": _pooled(games, "plays"),
        "attacks_per_game": _pooled(games, "attacks_used"),
        "spread_counters_per_game": _pooled(games, "spread_targets"),
    }


def cohort(games: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": summarise(games),
        "wins": summarise([game for game in games if game["result"] == "win"]),
        "losses": summarise([game for game in games if game["result"] == "loss"]),
    }


def deck_variants(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        grouped.setdefault(str(game["opponent_deck_hash"]), []).append(game)
    rows = []
    for hash_value, block in grouped.items():
        wins = sum(game["result"] == "win" for game in block)
        rows.append({
            "deck_hash": hash_value,
            "games": len(block),
            "wins": wins,
            "win_rate": round(wins / len(block), 4),
            "team_ids": sorted({
                int(game["team_id"]) for game in block if "team_id" in game
            }),
        })
    return sorted(rows, key=lambda row: (-int(row["games"]), str(row["deck_hash"])))


def teacher_cohorts(games: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        if "team_id" in game:
            grouped.setdefault(int(game["team_id"]), []).append(game)
    return {
        str(team_id): cohort(block)
        for team_id, block in sorted(grouped.items())
    }


def _teacher_games(index: Path, matchups: set[str]) -> Iterable[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    for row in csv.DictReader(index.read_text(encoding="utf-8-sig").splitlines()):
        key = (str(row["episode_id"]), int(row["seat_index"]))
        if key in seen:
            continue
        seen.add(key)
        path = Path(row["replay_path"])
        if not path.is_absolute():
            path = index.parent.parent / path
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        decks = replay_decks(replay)
        if archetype(decks[1 - key[1]]) not in matchups:
            continue
        game = game_metrics(replay, key[1])
        game["episode_id"] = int(key[0])
        game["team_id"] = int(row["team_id"])
        yield game


def _live_games(run: Path, matchups: set[str]) -> Iterable[dict[str, Any]]:
    for row in csv.DictReader(
        (run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
    ):
        seat = row.get("detected_submission_agent_index", "")
        if seat not in ("0", "1"):
            continue
        episode = str(row["episode_id"])
        path = run / "episodes" / episode / "replay" / f"episode_{episode}.json"
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        if archetype(replay_decks(replay)[1 - int(seat)]) not in matchups:
            continue
        game = game_metrics(replay, int(seat))
        game["episode_id"] = int(episode)
        yield game


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--matchup", action="append", default=[])
    parser.add_argument(
        "--reference-deck", type=Path,
        help="Optional 60-line deck.csv whose exact opponent-list cell is reported.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    matchups = set(args.matchup or ["Mega Lucario ex", "Dragapult ex"])
    teachers = list(_teacher_games(args.teacher_index, matchups))
    live = list(_live_games(args.run, matchups))
    reference_hash = None
    if args.reference_deck:
        reference_hash = deck_hash(
            int(line.strip())
            for line in args.reference_deck.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    report: dict[str, Any] = {
        "matchups": sorted(matchups),
        "reference_deck_hash": reference_hash,
        "rows": {},
    }
    keys = (
        "phantom_dives", "max_line_bodies", "max_dragapult", "dragapult_created",
        "dragapult_created_bench", "evolve_dragapult_active_unpowered",
        "route_attach_to_bench", "first_dragapult_turn", "first_phantom_dive_turn",
        "prizes_taken", "prizes_conceded",
    )
    for matchup in sorted(matchups):
        teacher_block = [game for game in teachers if game["opponent"] == matchup]
        live_block = [game for game in live if game["opponent"] == matchup]
        reference_block = [
            game for game in teacher_block
            if game["opponent_deck_hash"] == reference_hash
        ] if reference_hash else []
        report["rows"][matchup] = {
            "teachers": cohort(teacher_block),
            "live": cohort(live_block),
            "teacher_opponent_deck_variants": deck_variants(teacher_block),
            "reference_deck_teachers": (
                cohort(reference_block) if reference_hash else None
            ),
            "reference_deck_teachers_by_team": (
                teacher_cohorts(reference_block) if reference_hash else None
            ),
            "live_games": live_block,
        }
        print(f"\n=== {matchup}: teachers {len(teacher_block)}, live {len(live_block)}")
        print(f"{'metric':38} {'teachers':>10} {'live':>10} {'delta':>10}")
        for key in keys:
            left = _mean(teacher_block, key)
            right = _mean(live_block, key)
            delta = None if left is None or right is None else right - left
            print(f"{key:38} {_fmt(left):>10} {_fmt(right):>10} {_fmt(delta):>10}")
        for group in ("plays", "attacks_used", "spread_targets"):
            left = _pooled(teacher_block, group)
            right = _pooled(live_block, group)
            names = sorted(set(left) | set(right))
            print(f"  {group}:")
            for name in names:
                if left.get(name, 0.0) or right.get(name, 0.0):
                    print(f"    {name:34} {left.get(name, 0.0):>8.3f} "
                          f"{right.get(name, 0.0):>8.3f}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
