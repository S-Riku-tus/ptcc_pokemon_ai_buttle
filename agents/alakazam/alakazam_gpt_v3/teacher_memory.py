"""Compact, serial-aware replay memory used by the v30 policy."""

from __future__ import annotations

import hashlib
import json
from typing import Any


AREA_KEYS = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench"}
UNORDERED_ZONES = {"hand", "discard", "prize"}


def _normalise(
    value: Any,
    *,
    drop_serial: bool,
    parent_key: str = "",
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalise(
                child,
                drop_serial=drop_serial,
                parent_key=str(key),
            )
            for key, child in sorted(value.items())
            if not (drop_serial and key == "serial")
        }
    if isinstance(value, list):
        children = [
            _normalise(child, drop_serial=drop_serial)
            for child in value
        ]
        if parent_key in UNORDERED_ZONES:
            children.sort(
                key=lambda child: json.dumps(
                    child,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return children
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=12).hexdigest()


def _cards(player: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [
        card
        for card in (player.get(key) or [])
        if isinstance(card, dict)
    ]


def _card_at(
    observation: dict[str, Any],
    player_index: int,
    area: int,
    index: int,
) -> dict[str, Any]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    if not 0 <= player_index < len(players):
        return {}
    key = AREA_KEYS.get(area)
    if key == "deck":
        deck = (observation.get("select") or {}).get("deck") or []
        if 0 <= index < len(deck) and isinstance(deck[index], dict):
            return deck[index]
        return {}
    if key is None:
        return {}
    cards = _cards(players[player_index], key)
    if 0 <= index < len(cards):
        return cards[index]
    return {}


def _source_card(
    observation: dict[str, Any],
    option: dict[str, Any],
) -> dict[str, Any]:
    current = observation.get("current") or {}
    player_index = int(
        option.get("playerIndex", current.get("yourIndex", 0))
    )
    option_type = int(option.get("type", -1))
    area = int(option.get("area", -1))
    if option_type in (7, 8, 9):
        area = 2
    index = option.get("index")
    if not isinstance(index, int):
        return {}
    return _card_at(observation, player_index, area, index)


def _target_card(
    observation: dict[str, Any],
    option: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    current = observation.get("current") or {}
    player_index = int(
        option.get("playerIndex", current.get("yourIndex", 0))
    )
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if isinstance(area, int) and isinstance(index, int):
        return _card_at(observation, player_index, area, index)
    if int(option.get("type", -1)) in (3, 10, 12):
        return source
    return {}


def _energy_count(card: dict[str, Any]) -> int:
    return len(card.get("energyCards") or card.get("energies") or [])


def semantic_option_key(
    observation: dict[str, Any],
    option_index: int,
) -> str:
    options = (observation.get("select") or {}).get("option") or []
    if not 0 <= option_index < len(options):
        return "OUT"
    option = options[option_index]
    source = _source_card(observation, option)
    target = _target_card(observation, option, source)
    values = (
        int(option.get("type", -1)),
        int(source.get("id", -1)),
        int(option.get("attackId", -1)),
        int(target.get("id", -1)),
        int(target.get("hp", -1)),
        int(target.get("maxHp", -1)),
        _energy_count(target) if target else -1,
        len(target.get("tools") or []) if target else -1,
    )
    return ",".join(str(value) for value in values)


def semantic_action_key(
    observation: dict[str, Any],
    action: list[int],
) -> str:
    values = [semantic_option_key(observation, index) for index in action]
    if len(values) > 1:
        values.sort()
    return ";".join(values)


def resolve_semantic_action(
    observation: dict[str, Any],
    semantic_action: str,
) -> list[int] | None:
    requested = semantic_action.split(";") if semantic_action else []
    options = (observation.get("select") or {}).get("option") or []
    available: dict[str, list[int]] = {}
    for index in range(len(options)):
        key = semantic_option_key(observation, index)
        available.setdefault(key, []).append(index)
    result = []
    for key in requested:
        candidates = available.get(key)
        if not candidates:
            return None
        result.append(candidates.pop(0))
    return result


def teacher_memory_keys(
    observation: dict[str, Any],
) -> tuple[str, str]:
    exact = {
        key: value
        for key, value in observation.items()
        if key not in {"remainingOverageTime", "step"}
    }
    exact_key = _hash(exact)

    select = observation.get("select") or {}
    options = select.get("option") or []
    canonical_select = {
        "type": int(select.get("type", -1)),
        "context": int(select.get("context", -1)),
        "minCount": int(select.get("minCount") or 0),
        "maxCount": int(select.get("maxCount") or 0),
        "remainDamageCounter": int(
            select.get("remainDamageCounter") or 0
        ),
        "remainEnergyCost": _normalise(
            select.get("remainEnergyCost"),
            drop_serial=True,
        ),
        "contextCard": _normalise(
            select.get("contextCard"),
            drop_serial=True,
        ),
        "effect": _normalise(
            select.get("effect"),
            drop_serial=True,
        ),
        "options": sorted(
            semantic_option_key(observation, index)
            for index in range(len(options))
        ),
    }
    logs = _normalise(
        observation.get("logs") or [],
        drop_serial=True,
    )
    canonical_key = _hash({
        "current": _normalise(
            observation.get("current") or {},
            drop_serial=True,
        ),
        "select": canonical_select,
        "recentLogs": logs[-24:],
    })
    return exact_key, canonical_key
