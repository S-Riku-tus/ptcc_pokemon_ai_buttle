from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .common import HIGH_IMPORTANCE_CARDS, MAJKEL_KEY_CARDS, stable_code


STATE_FEATURES = [
    "turn", "go_first", "your_prizes", "opp_prizes", "your_deck", "opp_deck",
    "your_hand", "opp_hand", "your_bench", "opp_bench", "supporter_played",
    "energy_attached", "retreated", "stadium_played", "stadium_id",
    "your_active_id", "your_active_hp", "your_active_max_hp", "your_active_damage",
    "your_active_energy", "opp_active_id", "opp_active_hp", "opp_active_max_hp",
    "opp_active_damage", "opp_active_energy", "your_alakazam", "your_kadabra",
    "your_abra", "your_dudunsparce", "your_dunsparce", "opp_board_signature",
    "your_board_signature", "low_deck", "hand_alakazam", "hand_kadabra", "hand_abra",
    "hand_candy", "hand_boss", "hand_hammer", "hand_xerosic", "hand_energy",
    "select_type_code", "context_code", "option_count", "min_count", "max_count",
]

CANDIDATE_FEATURES = [
    "candidate_index", "option_type_code", "card_id", "attack_id", "skill_id",
    "target_card_id", "player_index", "source_area", "target_area", "source_index",
    "target_index", "candidate_hp", "candidate_energy", "hand_delta", "board_delta",
    "attack_damage", "ko_possible", "is_end", "is_attack", "is_ability", "is_retreat",
    "is_evolve", "is_energy", "is_boss", "is_hammer", "is_xerosic", "is_candy",
    "is_dudunsparce", "high_importance", "context_option_code",
]

FEATURE_COLUMNS = STATE_FEATURES + CANDIDATE_FEATURES

OPTION_NAMES = {
    0: "number", 1: "yes", 2: "no", 3: "card", 4: "tool_card",
    5: "energy_card", 6: "energy", 7: "play", 8: "attach", 9: "evolve",
    10: "ability", 11: "discard", 12: "retreat", 13: "attack", 14: "end",
    15: "skill", 16: "special_condition",
}


def _cards(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [card for card in value if isinstance(card, dict)]


def _pokemon_cards(player: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(player.get("active")) + _cards(player.get("bench"))


def _card_id(card: Any) -> int:
    if not isinstance(card, dict):
        return 0
    try:
        return int(card.get("id") or card.get("cardId") or 0)
    except (TypeError, ValueError):
        return 0


def _energy_count(card: Any) -> int:
    if not isinstance(card, dict):
        return 0
    values = card.get("energyCards") or card.get("energies") or []
    return len(values) if isinstance(values, list) else 0


def _active(player: dict[str, Any]) -> dict[str, Any]:
    active = _cards(player.get("active"))
    return active[0] if active else {}


def _remaining_hp(card: dict[str, Any]) -> int:
    hp = int(card.get("hp") or card.get("maxHp") or 0)
    if "hp" not in card:
        hp -= int(card.get("damage") or 0)
    return max(0, hp)


def _board_signature(cards: list[dict[str, Any]]) -> int:
    ids = sorted(_card_id(card) for card in cards)
    return stable_code(",".join(map(str, ids)), buckets=8191)


def state_features(observation: dict[str, Any]) -> dict[str, float]:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your_index = int(current.get("yourIndex") or 0)
    if len(players) < 2:
        players = list(players) + [{}] * (2 - len(players))
    you = players[your_index] if your_index < len(players) else {}
    opponent_index = 1 - your_index
    opponent = players[opponent_index] if opponent_index < len(players) else {}
    your_active = _active(you)
    opp_active = _active(opponent)
    your_board = _pokemon_cards(you)
    opp_board = _pokemon_cards(opponent)
    hand = _cards(you.get("hand"))
    hand_counts = Counter(_card_id(card) for card in hand)
    select = observation.get("select") or {}
    stadium = _cards(current.get("stadium"))

    result: dict[str, float] = {
        "turn": float(current.get("turn") or 0),
        "go_first": float(current.get("firstPlayer") == your_index),
        "your_prizes": float(len(you.get("prize") or [])),
        "opp_prizes": float(len(opponent.get("prize") or [])),
        "your_deck": float(you.get("deckCount") or 0),
        "opp_deck": float(opponent.get("deckCount") or 0),
        "your_hand": float(you.get("handCount") or len(hand)),
        "opp_hand": float(opponent.get("handCount") or 0),
        "your_bench": float(len(you.get("bench") or [])),
        "opp_bench": float(len(opponent.get("bench") or [])),
        "supporter_played": float(bool(current.get("supporterPlayed"))),
        "energy_attached": float(bool(current.get("energyAttached"))),
        "retreated": float(bool(current.get("retreated"))),
        "stadium_played": float(bool(current.get("stadiumPlayed"))),
        "stadium_id": float(_card_id(stadium[0]) if stadium else 0),
        "your_active_id": float(_card_id(your_active)),
        "your_active_hp": float(_remaining_hp(your_active)),
        "your_active_max_hp": float(your_active.get("maxHp") or your_active.get("hp") or 0),
        "your_active_damage": float(your_active.get("damage") or 0),
        "your_active_energy": float(_energy_count(your_active)),
        "opp_active_id": float(_card_id(opp_active)),
        "opp_active_hp": float(_remaining_hp(opp_active)),
        "opp_active_max_hp": float(opp_active.get("maxHp") or opp_active.get("hp") or 0),
        "opp_active_damage": float(opp_active.get("damage") or 0),
        "opp_active_energy": float(_energy_count(opp_active)),
        "your_alakazam": float(sum(_card_id(card) == 743 for card in your_board)),
        "your_kadabra": float(sum(_card_id(card) == 742 for card in your_board)),
        "your_abra": float(sum(_card_id(card) == 741 for card in your_board)),
        "your_dudunsparce": float(sum(_card_id(card) == 66 for card in your_board)),
        "your_dunsparce": float(sum(_card_id(card) == 305 for card in your_board)),
        "opp_board_signature": float(_board_signature(opp_board)),
        "your_board_signature": float(_board_signature(your_board)),
        "low_deck": float((you.get("deckCount") or 0) <= 6),
        "hand_alakazam": float(hand_counts[743]),
        "hand_kadabra": float(hand_counts[742]),
        "hand_abra": float(hand_counts[741]),
        "hand_candy": float(hand_counts[1079]),
        "hand_boss": float(hand_counts[1182]),
        "hand_hammer": float(hand_counts[1081]),
        "hand_xerosic": float(hand_counts[1197]),
        "hand_energy": float(sum(hand_counts[card] for card in (5, 13, 19))),
        "select_type_code": float(stable_code(select.get("type"))),
        "context_code": float(stable_code(select.get("context"))),
        "option_count": float(len(select.get("option") or [])),
        "min_count": float(select.get("minCount") or 0),
        "max_count": float(select.get("maxCount") or 0),
    }
    return result


def _walk_values(value: Any, keys: tuple[str, ...]) -> int:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return int(candidate)
        for nested in value.values():
            found = _walk_values(nested, keys)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _walk_values(nested, keys)
            if found:
                return found
    return 0


def _walk_optional(value: Any, keys: tuple[str, ...]) -> int | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return int(candidate)
        for nested in value.values():
            found = _walk_optional(nested, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _walk_optional(nested, keys)
            if found is not None:
                return found
    return None


def _resolve_area_card(observation: dict[str, Any], area: int, index: int, player_index: int) -> dict[str, Any]:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your_index = int(current.get("yourIndex") or 0)
    if player_index not in (0, 1):
        player_index = your_index
    player = players[player_index] if player_index < len(players) else {}
    select = observation.get("select") or {}
    zones = {
        1: select.get("deck"),
        2: player.get("hand"),
        3: player.get("discard"),
        4: player.get("active"),
        5: player.get("bench"),
        6: player.get("prize"),
        7: current.get("stadium"),
        12: current.get("looking"),
    }
    values = zones.get(area) or []
    try:
        card = values[index]
    except (IndexError, TypeError):
        return {}
    return card if isinstance(card, dict) else {}


def option_identity(
    observation: dict[str, Any], option: dict[str, Any], candidate_index: int
) -> dict[str, int | str]:
    select = observation.get("select") or {}
    raw_option_type = option.get("type")
    option_type_number = int(raw_option_type) if isinstance(raw_option_type, (int, float)) else -1
    option_type = OPTION_NAMES.get(option_type_number, str(raw_option_type or "unknown").lower())
    current = observation.get("current") or {}
    observed_player = _walk_optional(option, ("playerIndex", "player"))
    player_index = int(current.get("yourIndex") or 0) if observed_player is None else observed_player
    source_area = _walk_values(option, ("area", "fromArea", "sourceArea"))
    source_index = _walk_values(option, ("index", "cardIndex", "sourceIndex"))
    target_area = _walk_values(option, ("targetArea", "toArea", "inPlayArea"))
    target_index = _walk_values(option, ("targetIndex", "inPlayIndex"))
    card_id = _walk_values(option, ("cardId", "id"))
    attack_id = _walk_values(option, ("attackId",))
    skill_id = _walk_values(option, ("skillId", "abilityId"))
    if not source_area and option_type_number == 7:
        source_area = 2
    elif not source_area and option_type_number in {12, 13}:
        source_area = 4
    resolved = _resolve_area_card(observation, source_area, source_index, player_index)
    if not card_id:
        card_id = _card_id(resolved)
    target = _resolve_area_card(observation, target_area, target_index, player_index)
    target_card_id = _walk_values(option, ("targetCardId", "cardIdTarget")) or _card_id(target)
    return {
        "option_type": option_type,
        "option_type_number": option_type_number,
        "card_id": card_id,
        "attack_id": attack_id,
        "skill_id": skill_id,
        "target_card_id": target_card_id,
        "player_index": player_index,
        "source_area": source_area,
        "target_area": target_area,
        "source_index": source_index,
        "target_index": target_index,
        "context": str(select.get("context") or ""),
        "select_type": str(select.get("type") or ""),
    }


def candidate_features(
    observation: dict[str, Any], option: dict[str, Any], candidate_index: int,
    cards: dict[int, dict[str, Any]] | None = None,
    attacks: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, int | str]]:
    identity = option_identity(observation, option, candidate_index)
    text = " ".join((identity["option_type"], identity["context"], identity["select_type"])).lower()
    card_id = int(identity["card_id"])
    attack_id = int(identity["attack_id"])
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your_index = int(current.get("yourIndex") or 0)
    opponent = players[1 - your_index] if len(players) > 1 else {}
    opp_active = _active(opponent)
    attack_meta = (attacks or {}).get(attack_id, {})
    damage_raw = attack_meta.get("damage") or 0
    try:
        attack_damage = float(damage_raw)
    except (TypeError, ValueError):
        attack_damage = 0.0
    source_card = _resolve_area_card(
        observation, int(identity["source_area"]), int(identity["source_index"]), int(identity["player_index"])
    )
    source_area = int(identity["source_area"])
    option_type_number = int(identity["option_type_number"])
    is_attack = option_type_number == 13 or attack_id > 0
    is_ability = option_type_number in {10, 15}
    is_end = option_type_number == 14
    is_retreat = option_type_number == 12
    is_evolve = option_type_number == 9
    is_energy = option_type_number == 8 or (card_id in {5, 13, 19} and int(identity["source_area"]) == 2)
    result = {
        "candidate_index": float(candidate_index),
        "option_type_code": float(stable_code(identity["option_type"])),
        "card_id": float(card_id),
        "attack_id": float(attack_id),
        "skill_id": float(identity["skill_id"]),
        "target_card_id": float(identity["target_card_id"]),
        "player_index": float(identity["player_index"]),
        "source_area": float(source_area),
        "target_area": float(identity["target_area"]),
        "source_index": float(identity["source_index"]),
        "target_index": float(identity["target_index"]),
        "candidate_hp": float(_remaining_hp(source_card)),
        "candidate_energy": float(_energy_count(source_card)),
        "hand_delta": float(-1 if source_area == 2 else 1 if "tohand" in text.replace("_", "") else 0),
        "board_delta": float(1 if ("bench" in text or is_evolve) else -1 if is_ability and card_id == 66 else 0),
        "attack_damage": attack_damage,
        "ko_possible": float(is_attack and attack_damage >= _remaining_hp(opp_active) > 0),
        "is_end": float(is_end),
        "is_attack": float(is_attack),
        "is_ability": float(is_ability),
        "is_retreat": float(is_retreat),
        "is_evolve": float(is_evolve),
        "is_energy": float(is_energy),
        "is_boss": float(card_id == 1182),
        "is_hammer": float(card_id == 1081),
        "is_xerosic": float(card_id == 1197),
        "is_candy": float(card_id == 1079),
        "is_dudunsparce": float(card_id == 66),
        "high_importance": float(card_id in HIGH_IMPORTANCE_CARDS or is_attack or is_retreat),
        "context_option_code": float(stable_code(f"{identity['context']}|{identity['option_type']}")),
    }
    return result, identity


def semantic_key(identity: dict[str, int | str]) -> str:
    fields = (
        "select_type", "context", "option_type", "card_id", "attack_id", "skill_id",
        "target_card_id", "source_area", "target_area",
    )
    return "|".join(str(identity.get(field, "")) for field in fields)


def exact_option_key(option: dict[str, Any]) -> str:
    return json.dumps(option, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def action_type(identity: dict[str, int | str]) -> str:
    text = " ".join(str(identity.get(key, "")) for key in ("option_type", "context", "select_type")).lower()
    card = int(identity.get("card_id") or 0)
    option_type_number = int(identity.get("option_type_number") or -1)
    if str(identity.get("option_type") or "") == "__none__":
        return "none"
    if option_type_number == 13 or int(identity.get("attack_id") or 0):
        return "attack"
    if option_type_number in {10, 15}:
        return "ability"
    if option_type_number == 12:
        return "retreat"
    if option_type_number == 14:
        return "end"
    if option_type_number == 9:
        return "evolve"
    if option_type_number == 8:
        return "energy"
    if card == 1182:
        return "boss"
    if card == 1081:
        return "hammer"
    if card == 1197:
        return "xerosic"
    if card:
        return "card"
    return str(identity.get("option_type") or "other").lower()


def tactical_fingerprint(state: dict[str, float], identity: dict[str, int | str]) -> str:
    values = [
        int(state["turn"]), int(state["your_prizes"]), int(state["opp_prizes"]),
        int(state["your_deck"] // 5), int(state["your_hand"] // 3),
        int(state["your_active_id"]), int(state["opp_active_id"]),
        int(state["your_alakazam"]), int(state["your_kadabra"]),
        str(identity["select_type"]), str(identity["context"]),
    ]
    return "|".join(map(str, values))
