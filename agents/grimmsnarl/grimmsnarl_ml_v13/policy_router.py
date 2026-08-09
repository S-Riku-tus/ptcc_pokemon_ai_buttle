"""Sticky matchup router for Grimmsnarl ML v13.

The router only reads public zones.  It deliberately never inspects either
deck or the opponent's hand: those appear in visualizer payloads but are not
legal policy inputs.  Once a public card identifies the opponent, the route is
fixed for the rest of the game so decisions from different experts cannot be
stitched together inside one trajectory.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

PENDING = "pending"
DEFAULT = "v8_default"
MIRROR = "v8_mirror"
ALAKAZAM = "v9_alakazam"
WALL = "wall_state_machine"

ALAKAZAM_IDS = frozenset({741, 742, 743})
GRIMMSNARL_IDS = frozenset({646, 647, 648})
WALL_IDS = frozenset({117, 330, 344, 345})
NEUTRALIZATION_ZONE_ID = 1247


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [card for card in (player.get(area) or []) if isinstance(card, dict)]


def _public_ids(player: dict[str, Any]) -> set[int]:
    """Card ids legally revealed by the opponent so far."""
    result: set[int] = set()
    for area in ("active", "bench", "discard"):
        for card in _cards(player, area):
            result.add(int(card.get("id", -1)))
            for previous in card.get("preEvolution") or []:
                if isinstance(previous, dict):
                    result.add(int(previous.get("id", -1)))
    result.discard(-1)
    return result


def _visible_opponent(observation: dict[str, Any]) -> tuple[set[int], bool]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    your = int(current.get("yourIndex", 0))
    other = 1 - your
    if len(players) <= other or not isinstance(players[other], dict):
        return set(), False
    opponent = players[other]
    ids = _public_ids(opponent)
    has_board = bool(_cards(opponent, "active") or _cards(opponent, "bench"))
    return ids, has_board


def classify(observation: dict[str, Any]) -> str:
    """Classify from public information, without making the choice sticky."""
    if not isinstance(observation, dict):
        return PENDING
    ids, has_board = _visible_opponent(observation)
    current = observation.get("current") or {}
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
    return DEFAULT if has_board else PENDING


class PolicyRouter:
    """Choose one expert and hold it for the rest of the game."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.route = PENDING
        self.lock_turn: int | None = None
        self.decisions: Counter[str] = Counter()
        self.lock_counts: Counter[str] = Counter()
        self.transitions: list[dict[str, Any]] = []

    def choose(self, observation: dict[str, Any]) -> str:
        detected = classify(observation)
        current = observation.get("current") or {}
        turn = int(current.get("turn", -1))
        if self.route == PENDING:
            # Keep using v8 provisionally during setup/turn 1.  This gives a
            # signature basic (Abra, Impidimp or Dwebble) time to become public
            # before the normal sticky lock is made.
            should_lock = detected not in (PENDING, DEFAULT) or turn >= 2
            if detected != PENDING and should_lock:
                self.route = detected
                self.lock_turn = turn
                self.lock_counts[detected] += 1
                self.transitions.append({
                    "from": PENDING, "to": detected, "turn": turn,
                })
        elif self.route == DEFAULT and detected == WALL:
            # A generic-looking lead can conceal Dwebble/Crustle until after
            # the default route locks.  Allow exactly one safety promotion,
            # never a switch back: the wall state machine is meaningful only
            # once a public card proves the immunity plan exists.
            self.route = WALL
            self.lock_turn = turn
            self.lock_counts[WALL] += 1
            self.transitions.append({
                "from": DEFAULT, "to": WALL, "turn": turn,
            })
        effective = DEFAULT if self.route == PENDING else self.route
        self.decisions[effective] += 1
        return effective

    def snapshot(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "lock_turn": self.lock_turn,
            "decisions": dict(self.decisions),
            "locks": dict(self.lock_counts),
            "transitions": list(self.transitions),
        }
