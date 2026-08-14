"""Mirror-only veto for Froslass evolution over the frozen v22 policy.

The 194-game v22 pool leaves one decision-level signal: in 56 mirrors, games
without a Froslass evolution went 23-9 while games with one went 7-17.  The
effect survives opponent-rating and turn-order controls and is absent outside
the mirror.  Freezing Shroud's board ledger does not explain it, so this layer
does not inspect ``shroud_net`` or invent a favourable-board exception.

The intervention is deliberately the measured one and no wider.  Once the
opponent has publicly revealed the Grimmsnarl line, a v22 choice to evolve a
Froslass is replaced by v22's highest-scored non-Froslass option.  Snorunt
setup, searches, attacks, targets and every non-mirror decision stay v22.
"""

from __future__ import annotations

from typing import Any

import ml_features as mf

GRIMMSNARL_LINE_IDS = frozenset({
    mf.IMPIDIMP_ID,
    mf.MORGREM_ID,
    mf.GRIMMSNARL_EX_ID,
})


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [
        card for card in (player.get(area) or [])
        if isinstance(card, dict)
    ]


def _public_opponent_ids(observation: dict[str, Any]) -> set[int]:
    """IDs visible in play/discard, including revealed pre-evolutions."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    your = int(current.get("yourIndex", 0))
    other = 1 - your
    if len(players) <= other or not isinstance(players[other], dict):
        return set()

    result: set[int] = set()
    opponent = players[other]
    for area in ("active", "bench", "discard"):
        for card in _cards(opponent, area):
            result.add(int(card.get("id", -1)))
            for previous in card.get("preEvolution") or []:
                if isinstance(previous, dict):
                    result.add(int(previous.get("id", -1)))
    result.discard(-1)
    return result


def _best_by_score(
    candidates: list[int], scores: dict[int, float] | None,
) -> int:
    """Preserve v22's ordering inside the set allowed by the veto."""
    if not scores:
        return candidates[0]
    scored = [slot for slot in candidates if slot in scores]
    if not scored:
        return candidates[0]
    return max(scored, key=lambda slot: scores[slot])


class MirrorFroslassGuard:
    """Suppress only a selected Froslass evolution in a visible mirror."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mirror = False
        self._last_turn: int | None = None
        self.stats: dict[str, int] = {
            "observations": 0,
            "mirror_transitions": 0,
            "mirror_decisions": 0,
            "froslass_offered": 0,
            "froslass_selected": 0,
            "overrides": 0,
            "no_alternative": 0,
            "new_game_detected": 0,
            "errors": 0,
        }

    @property
    def is_mirror(self) -> bool:
        return self._mirror

    def observe(self, observation: dict[str, Any]) -> bool:
        """Update the sticky matchup flag from public information only."""
        try:
            self.stats["observations"] += 1
            current = observation.get("current") or {}
            turn = int(current.get("turn", -1))
            if self._last_turn is not None and turn < self._last_turn:
                self._mirror = False
                self.stats["new_game_detected"] += 1
            self._last_turn = turn
            if (
                not self._mirror
                and _public_opponent_ids(observation) & GRIMMSNARL_LINE_IDS
            ):
                self._mirror = True
                self.stats["mirror_transitions"] += 1
            return self._mirror
        except Exception:  # noqa: BLE001 - a guard must never crash a game
            self.stats["errors"] += 1
            return self._mirror

    @staticmethod
    def _is_froslass_evolve(
        current: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
    ) -> bool:
        if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
            return False
        if mf.action_type(current, option, select) != "evolve":
            return False
        card = mf.candidate_card(current, option, select) or {}
        return int(card.get("id", -1)) == mf.FROSLASS_ID

    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None = None,
    ) -> int:
        """Return v22's index unless it is the mirror Froslass evolution."""
        try:
            if not self._mirror:
                return index
            self.stats["mirror_decisions"] += 1
            current = observation.get("current") or {}
            options = list(select.get("option") or [])
            if not 0 <= index < len(options):
                return index

            froslass = [
                slot for slot, option in enumerate(options)
                if self._is_froslass_evolve(current, select, option)
            ]
            if not froslass:
                return index
            self.stats["froslass_offered"] += 1
            if index not in froslass:
                return index
            self.stats["froslass_selected"] += 1

            alternatives = [
                slot for slot in range(len(options)) if slot not in froslass
            ]
            if not alternatives:
                self.stats["no_alternative"] += 1
                return index
            self.stats["overrides"] += 1
            return _best_by_score(alternatives, scores)
        except Exception:  # noqa: BLE001 - preserve the legal base choice
            self.stats["errors"] += 1
            return index

    def snapshot(self) -> dict[str, int | bool]:
        return {**self.stats, "is_mirror": self._mirror}
