"""Audit Alakazam ladder replays for Boss, evolution, and spread-damage choices."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile


BOSS_ORDERS = 1182
ABRA = 741
KADABRA = 742
ALAKAZAM = 743
FROSLASS = 104
GRIMMSNARL_EX = 648

PLAY = 7
EVOLVE = 9
ATTACK = 13
END = 14
MAIN_SELECT_TYPE = 0
MAIN_SELECT_CONTEXT = 0
ACTIVE = 4
BENCH = 5


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [card for card in (player.get(area) or []) if isinstance(card, dict)]


def _card_at(player: dict[str, Any], area: int, index: int) -> dict[str, Any] | None:
    key = {1: "deck", 2: "hand", 3: "discard", ACTIVE: "active", BENCH: "bench"}.get(area)
    if key is None:
        return None
    cards = player.get(key) or []
    if not isinstance(index, int) or not 0 <= index < len(cards):
        return None
    card = cards[index]
    return card if isinstance(card, dict) else None


def _selected_option(
    steps: list[list[dict[str, Any]]], step_index: int, seat: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    state = steps[step_index][seat]
    observation = state.get("observation") or {}
    options = ((observation.get("select") or {}).get("option") or [])
    if step_index + 1 >= len(steps) or seat >= len(steps[step_index + 1]):
        return None, options
    action = steps[step_index + 1][seat].get("action")
    if not isinstance(action, list) or len(action) != 1:
        return None, options
    index = action[0]
    if not isinstance(index, int) or not 0 <= index < len(options):
        return None, options
    return options[index], options


def _team_names(replay: dict[str, Any]) -> list[str]:
    info = replay.get("info") or {}
    names = info.get("TeamNames") or [agent.get("Name") for agent in info.get("Agents") or []]
    return [str(name or "") for name in names]


def _seat_for_team(replay: dict[str, Any], team_name: str) -> int | None:
    folded = team_name.strip().casefold()
    matches = [index for index, name in enumerate(_team_names(replay)) if name.strip().casefold() == folded]
    return matches[0] if len(matches) == 1 else None


def _initial_decks(replay: dict[str, Any]) -> list[list[dict[str, Any]]]:
    try:
        visual = replay["steps"][0][0]["visualize"][0]
        return [list(player.get("deck") or []) for player in visual["current"]["players"]]
    except (KeyError, IndexError, TypeError):
        return [[], []]


def _name_map(replay: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for deck in _initial_decks(replay):
        for card in deck:
            if isinstance(card, dict) and card.get("id") is not None:
                result[int(card["id"])] = str(card.get("name") or card["id"])
    return result


def _archetype(deck: list[dict[str, Any]]) -> str:
    names = {str(card.get("name") or "") for card in deck if isinstance(card, dict)}
    if any("Grimmsnarl" in name for name in names):
        return "Marnie's Grimmsnarl ex"
    if any("Alakazam" in name for name in names):
        return "Alakazam"
    for marker in ("Team Rocket's Mewtwo", "Mega Kangaskhan", "Archaludon", "Dragapult"):
        if any(marker in name for name in names):
            return marker
    pokemon = [name for name in names if name and not any(token in name for token in ("Energy", "Orders", "Gym"))]
    return sorted(pokemon)[0] if pokemon else "unknown"


def _prize_value(card: dict[str, Any] | None, names: dict[int, str]) -> int:
    if not card:
        return 0
    name = names.get(int(card.get("id", -1)), "")
    if "Mega " in name and " ex" in name:
        return 3
    return 2 if " ex" in name else 1


def _energy_count(card: dict[str, Any] | None) -> int:
    return len((card or {}).get("energies") or [])


def _action_card(current: dict[str, Any], seat: int, option: dict[str, Any]) -> dict[str, Any] | None:
    players = current.get("players") or [{}, {}]
    if not 0 <= seat < len(players):
        return None
    return _card_at(players[seat], 2, int(option.get("index", -1)))


def _target_card(current: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    players = current.get("players") or [{}, {}]
    player_index = int(option.get("playerIndex", current.get("yourIndex", 0)))
    if not 0 <= player_index < len(players):
        return None
    area = int(option.get("inPlayArea", option.get("area", -1)))
    index = int(option.get("inPlayIndex", option.get("index", -1)))
    return _card_at(players[player_index], area, index)


def _effect_card_id(select: dict[str, Any]) -> int:
    effect = select.get("effect") or select.get("contextCard") or {}
    return int(effect.get("id", -1)) if isinstance(effect, dict) else -1


def _two_turn_boss_candidates(
    current: dict[str, Any], seat: int, names: dict[int, str]
) -> list[dict[str, Any]]:
    players = current.get("players") or [{}, {}]
    if len(players) != 2:
        return []
    me, opp = players[seat], players[1 - seat]
    active = (_cards(me, "active") or [None])[0]
    opp_active = (_cards(opp, "active") or [None])[0]
    if not active or int(active.get("id", -1)) != ALAKAZAM or _energy_count(active) < 1:
        return []
    hand_count = int(me.get("handCount") or len(me.get("hand") or []))
    damage = 20 * max(0, hand_count - 1)
    if damage <= 0:
        return []
    active_prizes = _prize_value(opp_active, names)
    my_prizes = len(me.get("prize") or [])
    rows = []
    for target in _cards(opp, "bench"):
        hp = int(target.get("hp") or 0)
        prizes = _prize_value(target, names)
        same_turn = damage >= hp > 0
        two_turn = damage < hp <= 2 * damage
        closes_game = prizes >= my_prizes > 0
        prize_upgrade = prizes > active_prizes
        if two_turn and prizes >= 2 and (closes_game or prize_upgrade):
            rows.append({
                "target_id": int(target.get("id", -1)),
                "target_name": names.get(int(target.get("id", -1)), ""),
                "target_hp": hp,
                "target_prizes": prizes,
                "target_energy": _energy_count(target),
                "damage_after_boss": damage,
                "same_turn_ko": same_turn,
                "closes_game_in_two_hits": closes_game,
                "prize_upgrade": prize_upgrade,
            })
    return rows


def analyze_replay(replay: dict[str, Any], seat: int) -> dict[str, Any]:
    steps = replay.get("steps") or []
    names = _name_map(replay)
    decks = _initial_decks(replay)
    opponent_deck = decks[1 - seat] if len(decks) == 2 else []
    target_deck = decks[seat] if len(decks) == 2 else []
    rewards = replay.get("rewards") or [None, None]
    reward = rewards[seat] if seat < len(rewards) else None
    episode_id = int((replay.get("info") or {}).get("EpisodeId", 0))

    acting_turns: set[int] = set()
    main_turns: set[int] = set()
    attack_turns: set[int] = set()
    opportunity_turns: set[int] = set()
    boss_plays = 0
    boss_play_turns: set[int] = set()
    boss_targets: list[dict[str, Any]] = []
    boss_two_turn_opportunities: list[dict[str, Any]] = []
    evolution_choices: list[dict[str, Any]] = []
    dual_kadabra_choices: list[dict[str, Any]] = []
    selected_attacks: list[dict[str, Any]] = []
    hp_events: list[dict[str, Any]] = []
    last_hp: dict[int, tuple[int, int, int]] = {}
    seen_state_keys: set[tuple[int, int, int]] = set()

    for step_index in range(max(0, len(steps) - 1)):
        if seat >= len(steps[step_index]):
            continue
        state = steps[step_index][seat]
        observation = state.get("observation") or {}
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) != 2:
            continue
        turn = int(current.get("turn", -1))
        action_count = int(current.get("turnActionCount", -1))
        state_key = (turn, action_count, int(current.get("yourIndex", -1)))
        if state_key not in seen_state_keys:
            seen_state_keys.add(state_key)
            now: dict[int, tuple[int, int, int]] = {}
            for area, key in ((ACTIVE, "active"), (BENCH, "bench")):
                for card in _cards(players[seat], key):
                    serial = int(card.get("serial", -1))
                    now[serial] = (int(card.get("hp") or 0), int(card.get("id", -1)), area)
                    if serial in last_hp:
                        old_hp, old_id, old_area = last_hp[serial]
                        if int(card.get("hp") or 0) < old_hp:
                            hp_events.append({
                                "turn": turn,
                                "card_id": old_id,
                                "area": old_area,
                                "damage": old_hp - int(card.get("hp") or 0),
                            })
            last_hp.update(now)

        if state.get("status") != "ACTIVE":
            continue
        acting_turns.add(turn)
        select = observation.get("select") or {}
        selected, options = _selected_option(steps, step_index, seat)
        if selected is None:
            continue

        if _effect_card_id(select) == BOSS_ORDERS:
            target = _target_card(current, selected)
            if target:
                boss_targets.append({
                    "turn": turn,
                    "target_id": int(target.get("id", -1)),
                    "target_name": names.get(int(target.get("id", -1)), ""),
                    "target_hp": int(target.get("hp") or 0),
                    "target_prizes": _prize_value(target, names),
                    "target_energy": _energy_count(target),
                })

        if int(select.get("type", -1)) != MAIN_SELECT_TYPE or int(select.get("context", -1)) != MAIN_SELECT_CONTEXT:
            continue
        main_turns.add(turn)
        offered_attack = any(int(option.get("type", -1)) == ATTACK for option in options)
        if offered_attack:
            opportunity_turns.add(turn)
        selected_type = int(selected.get("type", -1))
        if selected_type == ATTACK:
            attack_turns.add(turn)
            active = (_cards(players[seat], "active") or [None])[0]
            opponent_active = (_cards(players[1 - seat], "active") or [None])[0]
            selected_attacks.append({
                "turn": turn,
                "attacker_id": int((active or {}).get("id", -1)),
                "attack_id": int(selected.get("attackId", -1)),
                "target_id": int((opponent_active or {}).get("id", -1)),
                "target_hp": int((opponent_active or {}).get("hp", 0)),
            })
        elif selected_type == PLAY:
            card = _action_card(current, seat, selected)
            if card and int(card.get("id", -1)) == BOSS_ORDERS:
                boss_plays += 1
                boss_play_turns.add(turn)
        elif selected_type == EVOLVE:
            card = _action_card(current, seat, selected)
            target = _target_card(current, selected)
            if card and target:
                row = {
                    "turn": turn,
                    "evolution_id": int(card.get("id", -1)),
                    "target_id": int(target.get("id", -1)),
                    "target_area": int(selected.get("inPlayArea", -1)),
                    "target_hp": int(target.get("hp") or 0),
                    "target_max_hp": int(target.get("maxHp") or 0),
                    "opponent_active_id": int(((_cards(players[1 - seat], "active") or [{}])[0]).get("id", -1)),
                    "opponent_active_energy": _energy_count((_cards(players[1 - seat], "active") or [None])[0]),
                }
                evolution_choices.append(row)
                same_evolution = []
                for option in options:
                    if int(option.get("type", -1)) != EVOLVE:
                        continue
                    option_card = _action_card(current, seat, option)
                    if option_card and int(option_card.get("id", -1)) == int(card.get("id", -1)):
                        same_evolution.append(int(option.get("inPlayArea", -1)))
                if int(card.get("id", -1)) == KADABRA and ACTIVE in same_evolution and BENCH in same_evolution:
                    dual_kadabra_choices.append(row)

        legal_boss = any(
            int(option.get("type", -1)) == PLAY
            and int((_action_card(current, seat, option) or {}).get("id", -1)) == BOSS_ORDERS
            for option in options
        )
        if legal_boss:
            for candidate in _two_turn_boss_candidates(current, seat, names):
                boss_two_turn_opportunities.append({
                    "turn": turn,
                    "selected_boss_now": bool(
                        selected_type == PLAY
                        and int((_action_card(current, seat, selected) or {}).get("id", -1)) == BOSS_ORDERS
                    ),
                    **candidate,
                })

    boss_target_by_turn = {event["turn"]: event["target_id"] for event in boss_targets}
    unique_two_turn: dict[tuple[int, int], dict[str, Any]] = {}
    for event in boss_two_turn_opportunities:
        key = (event["turn"], event["target_id"])
        old = unique_two_turn.get(key)
        if old is None or event["selected_boss_now"]:
            unique_two_turn[key] = event
    boss_two_turn_opportunities = []
    for event in unique_two_turn.values():
        event["boss_played_this_turn"] = event["turn"] in boss_play_turns
        event["boss_target_this_turn"] = boss_target_by_turn.get(event["turn"])
        event["candidate_was_boss_target"] = event["target_id"] == boss_target_by_turn.get(event["turn"])
        boss_two_turn_opportunities.append(event)

    first_attack = min(attack_turns) if attack_turns else None
    post_first_main = {turn for turn in main_turns if first_attack is not None and turn >= first_attack}
    ten_damage = [event for event in hp_events if event["damage"] == 10]
    return {
        "episode_id": episode_id,
        "team": _team_names(replay)[seat],
        "opponent": _team_names(replay)[1 - seat],
        "opponent_archetype": _archetype(opponent_deck),
        "target_deck": [
            {"id": int(card.get("id", -1)), "name": str(card.get("name") or "")}
            for card in target_deck if isinstance(card, dict)
        ],
        "won": bool(reward is not None and reward > 0),
        "reward": reward,
        "acting_turns": len(acting_turns),
        "main_turns": len(main_turns),
        "attack_turns": len(attack_turns),
        "attack_opportunity_turns": len(opportunity_turns),
        "missed_attack_opportunity_turns": len(opportunity_turns - attack_turns),
        "post_first_main_turns": len(post_first_main),
        "post_first_attack_turns": len(post_first_main & attack_turns),
        "boss_plays": boss_plays,
        "boss_targets": boss_targets,
        "boss_two_turn_opportunities": boss_two_turn_opportunities,
        "evolution_choices": evolution_choices,
        "dual_kadabra_choices": dual_kadabra_choices,
        "selected_attacks": selected_attacks,
        "hp_damage_events": hp_events,
        "ten_damage_events": ten_damage,
        "has_froslass": any(int(card.get("id", -1)) == FROSLASS for card in opponent_deck),
        "has_grimmsnarl_ex": any(int(card.get("id", -1)) == GRIMMSNARL_EX for card in opponent_deck),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["won"] for row in rows)
    sums = Counter()
    for row in rows:
        for key in (
            "acting_turns", "main_turns", "attack_turns", "attack_opportunity_turns",
            "missed_attack_opportunity_turns", "post_first_main_turns", "post_first_attack_turns",
            "boss_plays",
        ):
            sums[key] += int(row[key])
    boss_targets = [event for row in rows for event in row["boss_targets"]]
    two_turn = [event for row in rows for event in row["boss_two_turn_opportunities"]]
    evolutions = [event for row in rows for event in row["evolution_choices"]]
    dual = [event for row in rows for event in row["dual_kadabra_choices"]]
    ten_damage = [event for row in rows for event in row["ten_damage_events"]]
    grim = [row for row in rows if row["has_grimmsnarl_ex"]]
    target_deck = rows[0].get("target_deck", []) if rows else []
    target_deck_counts = Counter(int(card["id"]) for card in target_deck)
    target_deck_names = {int(card["id"]): str(card["name"]) for card in target_deck}
    by_matchup = []
    for archetype, matchup_rows in sorted(
        ((key, list(group)) for key, group in _group_rows(rows, "opponent_archetype")),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        by_matchup.append({
            "opponent_archetype": archetype,
            "games": len(matchup_rows),
            "wins": sum(row["won"] for row in matchup_rows),
            "win_rate": _ratio(sum(row["won"] for row in matchup_rows), len(matchup_rows)),
        })
    return {
        "games": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": _ratio(wins, len(rows)),
        "target_deck": [
            {"id": card_id, "name": target_deck_names.get(card_id, ""), "count": count}
            for card_id, count in sorted(target_deck_counts.items())
        ],
        "all_acting_turn_attack_rate": _ratio(sums["attack_turns"], sums["acting_turns"]),
        "main_turn_attack_rate": _ratio(sums["attack_turns"], sums["main_turns"]),
        "attack_opportunity_conversion_rate": _ratio(sums["attack_turns"], sums["attack_opportunity_turns"]),
        "post_first_main_turn_attack_rate": _ratio(sums["post_first_attack_turns"], sums["post_first_main_turns"]),
        "boss_plays": sums["boss_plays"],
        "boss_targets": len(boss_targets),
        "boss_target_ids": dict(Counter(event["target_id"] for event in boss_targets).most_common()),
        "two_turn_boss_opportunities": len(two_turn),
        "two_turn_boss_opportunity_games": len({event["episode_id"] for event in _with_episode(rows, "boss_two_turn_opportunities")}),
        "two_turn_boss_opportunities_with_boss_play": sum(event["boss_played_this_turn"] for event in two_turn),
        "two_turn_candidates_selected_as_boss_target": sum(event["candidate_was_boss_target"] for event in two_turn),
        "evolution_choices": len(evolutions),
        "kadabra_evolutions_to_active": sum(event["evolution_id"] == KADABRA and event["target_area"] == ACTIVE for event in evolutions),
        "kadabra_evolutions_to_bench": sum(event["evolution_id"] == KADABRA and event["target_area"] == BENCH for event in evolutions),
        "dual_kadabra_choices": len(dual),
        "dual_kadabra_active_choices": sum(event["target_area"] == ACTIVE for event in dual),
        "dual_kadabra_bench_choices": sum(event["target_area"] == BENCH for event in dual),
        "ten_damage_events": len(ten_damage),
        "ten_damage_bench_events": sum(event["area"] == BENCH for event in ten_damage),
        "grimmsnarl_games": len(grim),
        "grimmsnarl_wins": sum(row["won"] for row in grim),
        "grimmsnarl_win_rate": _ratio(sum(row["won"] for row in grim), len(grim)),
        "grimmsnarl_ten_damage_events": sum(len(row["ten_damage_events"]) for row in grim),
        "matchups": by_matchup,
        "totals": dict(sums),
    }


def _group_rows(rows: list[dict[str, Any]], key: str) -> Iterable[tuple[str, Iterable[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return groups.items()


def _with_episode(rows: list[dict[str, Any]], key: str) -> Iterable[dict[str, Any]]:
    for row in rows:
        for event in row[key]:
            yield {"episode_id": row["episode_id"], **event}


def _iter_directory(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(root.glob("episodes/*/replay/episode_*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def _iter_zip(path: Path) -> Iterable[dict[str, Any]]:
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith(".json") and ("/replay/episode_" in member or "/replays/episode_" in member):
                payload = json.loads(archive.read(member))
                if isinstance(payload, dict) and "steps" in payload:
                    yield payload


def _submission_seats(root: Path, submission_id: int) -> dict[int, int]:
    seats: dict[int, int] = {}
    if root.is_dir():
        episodes_path = root / "episodes.csv"
        if not episodes_path.exists():
            return seats
        text = episodes_path.read_text(encoding="utf-8-sig")
    else:
        with ZipFile(root) as archive:
            member = next(
                (name for name in archive.namelist() if name.endswith("episodes.csv")),
                None,
            )
            if member is None:
                return seats
            text = archive.read(member).decode("utf-8-sig")
    for row in csv.DictReader(text.splitlines()):
        episode_id = int(row["episode_id"])
        if int(row["agent_0_submission_id"]) == submission_id:
            seats[episode_id] = 0
        elif int(row["agent_1_submission_id"]) == submission_id:
            seats[episode_id] = 1
    return seats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Fetched run directory or full-replay ZIP")
    parser.add_argument("--submission-id", type=int)
    parser.add_argument("--team-name")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.submission_id and not args.team_name:
        parser.error("one of --submission-id or --team-name is required")

    try:
        replays = list(_iter_directory(args.source) if args.source.is_dir() else _iter_zip(args.source))
    except BadZipFile as exc:
        raise SystemExit(f"invalid replay ZIP: {args.source}: {exc}") from exc
    seats = _submission_seats(args.source, args.submission_id) if args.submission_id else {}
    rows = []
    unresolved = 0
    for replay in replays:
        episode_id = int((replay.get("info") or {}).get("EpisodeId", 0))
        seat = seats.get(episode_id)
        if seat is None and args.team_name:
            seat = _seat_for_team(replay, args.team_name)
        if seat is None:
            unresolved += 1
            continue
        rows.append(analyze_replay(replay, seat))

    report = {
        "source": str(args.source),
        "submission_id": args.submission_id,
        "team_name": args.team_name,
        "replays_found": len(replays),
        "replays_analyzed": len(rows),
        "unresolved_replays": unresolved,
        "aggregate": aggregate(rows),
        "episodes": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
