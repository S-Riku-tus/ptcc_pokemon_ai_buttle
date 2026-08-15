"""Public-information matchup observer for Grimmsnarl ML v28.

The router identifies the one route that switches rankers (damage-immune
walls) and also gives Teal Mask Ogerpon its own telemetry label.  Ogerpon still
uses the v25 race policy: the measured 93.9% next-turn lethal exposure makes it
a structural race, not a safe target for another unvalidated hard shell.

v18 also resets the sticky route when the public turn counter moves backwards.
The Kaggle process may be reused across episodes; without this reset, one
mirror seen by v16/v17 could leave the Froslass escalation suspended in every
later game handled by that process, even against a non-mirror deck.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

PENDING = "pending"
DEFAULT = "v8_default"
MIRROR = "v8_mirror"
ALAKAZAM = "v8_alakazam_guarded"
WALL = "v8_wall_guarded"
OGERPON = "v28_ogerpon_race"

ALAKAZAM_IDS = frozenset({741, 742, 743})
GRIMMSNARL_IDS = frozenset({646, 647, 648})
WALL_IDS = frozenset({117, 330, 344, 345})
TEAL_OGERPON_IDS = frozenset({96})
NEUTRALIZATION_ZONE_ID = 1247


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [card for card in (player.get(area) or []) if isinstance(card, dict)]


def _public_ids(player: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for area in ("active", "bench", "discard"):
        for card in _cards(player, area):
            result.add(int(card.get("id", -1)))
            for previous in card.get("preEvolution") or []:
                if isinstance(previous, dict):
                    result.add(int(previous.get("id", -1)))
    result.discard(-1)
    return result


def classify(observation: dict[str, Any]) -> str:
    if not isinstance(observation, dict):
        return PENDING
    current = observation.get("current") or {}
    players = current.get("players") or []
    your = int(current.get("yourIndex", 0))
    other = 1 - your
    if len(players) <= other or not isinstance(players[other], dict):
        return PENDING
    opponent = players[other]
    ids = _public_ids(opponent)
    stadium_ids = {
        int(card.get("id", -1))
        for card in (current.get("stadium") or [])
        if isinstance(card, dict)
    }
    if ids & WALL_IDS or NEUTRALIZATION_ZONE_ID in stadium_ids:
        return WALL
    if ids & TEAL_OGERPON_IDS:
        return OGERPON
    if ids & ALAKAZAM_IDS:
        return ALAKAZAM
    if ids & GRIMMSNARL_IDS:
        return MIRROR
    return DEFAULT if (_cards(opponent, "active") or _cards(opponent, "bench")) else PENDING


class PolicyRouter:
    """Observe the most specific public matchup; never select a ranker."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.route = PENDING
        self._last_turn = -1
        self.new_games = 0
        self.decisions: Counter[str] = Counter()
        self.transitions: list[dict[str, Any]] = []

    def choose(self, observation: dict[str, Any]) -> str:
        detected = classify(observation)
        current = observation.get("current") or {}
        turn = int(current.get("turn", -1))
        if turn < self._last_turn:
            old_route = self.route
            self.route = PENDING
            self.new_games += 1
            self.transitions.append({
                "from": old_route,
                "to": PENDING,
                "turn": turn,
                "reason": "new_game",
            })
        self._last_turn = turn
        previous = self.route
        if detected not in (PENDING, DEFAULT):
            # Wall evidence has safety relevance and may appear after a generic
            # or other named lead. Other named archetypes stay stable.
            if previous in (PENDING, DEFAULT) or detected == WALL:
                self.route = detected
        elif previous == PENDING and detected == DEFAULT:
            self.route = DEFAULT
        if self.route != previous:
            self.transitions.append(
                {"from": previous, "to": self.route, "turn": turn}
            )
        effective = DEFAULT if self.route == PENDING else self.route
        self.decisions[effective] += 1
        return effective

    def snapshot(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "decisions": dict(self.decisions),
            "transitions": list(self.transitions),
            "policy_switches": 0,
            "new_game_detected": self.new_games,
        }
