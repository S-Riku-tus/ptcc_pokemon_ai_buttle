"""Public-information matchup observer for Grimmsnarl ML v14.

Unlike v13, this module does not switch rankers.  It records which matchup has
become publicly identifiable so deployment logs remain attributable without
allowing a weak specialist to overwrite the champion.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

PENDING = "pending"
DEFAULT = "v8_default"
MIRROR = "v8_mirror"
ALAKAZAM = "v8_alakazam_guarded"
WALL = "v8_wall_guarded"

ALAKAZAM_IDS = frozenset({741, 742, 743})
GRIMMSNARL_IDS = frozenset({646, 647, 648})
WALL_IDS = frozenset({117, 330, 344, 345})
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
        self.decisions: Counter[str] = Counter()
        self.transitions: list[dict[str, Any]] = []

    def choose(self, observation: dict[str, Any]) -> str:
        detected = classify(observation)
        current = observation.get("current") or {}
        turn = int(current.get("turn", -1))
        previous = self.route
        if detected not in (PENDING, DEFAULT):
            # Wall evidence has safety relevance and may appear after a generic
            # or Alakazam-looking lead. Other named archetypes stay stable.
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
        }
