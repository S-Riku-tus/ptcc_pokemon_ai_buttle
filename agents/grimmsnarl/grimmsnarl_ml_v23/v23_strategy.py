"""Small goal layer for the two failure modes measured in the v22 ladder run.

The ranker still decides ordinary play.  This module intervenes only when a
legal option advances one of two explicit state goals:

* finish the first Grimmsnarl route during the opening; or
* against Alakazam, establish a second route before the first attacker falls.

It consumes the ranker's already-computed feature rows.  No second feature
extractor and no second model score are run, and the ranker's score remains the
tie-break inside the set of options that satisfy the goal.
"""

from __future__ import annotations

from typing import Any

MAIN_CONTEXT = 0
DECK_SEARCH_CONTEXT = 7

IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL_EX = 648
ALAKAZAM_LINE = frozenset({741, 742, 743})


def own_turn(current: dict[str, Any]) -> int:
    """One-based turn number for this seat; setup is turn zero."""
    turn = int(current.get("turn", 0) or 0)
    your = int(current.get("yourIndex", 0) or 0)
    first_value = current.get("firstPlayer", -1)
    first = int(-1 if first_value is None else first_value)
    return max(0, (turn + int(your == first)) // 2)


def _cards(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        card
        for area in ("active", "bench")
        for card in (player.get(area) or [])
        if isinstance(card, dict)
    ]


def _opponent_has_alakazam(current: dict[str, Any]) -> bool:
    players = current.get("players") or []
    your = int(current.get("yourIndex", 0) or 0)
    if len(players) != 2 or your not in (0, 1):
        return False
    return any(int(card.get("id", -1)) in ALAKAZAM_LINE for card in _cards(players[1 - your]))


def _best(candidates: list[int], scores: dict[int, float]) -> int | None:
    if not candidates:
        return None
    return max(candidates, key=lambda slot: (scores.get(slot, float("-inf")), -slot))


class StrategyPlanner:
    """Lexicographic route planner with a frozen, auditable footprint."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stats = {
            "decisions_seen": 0,
            "opening_route_considered": 0,
            "opening_route_overrides": 0,
            "opening_complete_overrides": 0,
            "opening_bridge_overrides": 0,
            "opening_first_body_overrides": 0,
            "alakazam_continuity_considered": 0,
            "alakazam_continuity_overrides": 0,
            "alakazam_search_overrides": 0,
            "alakazam_evolve_overrides": 0,
            "alakazam_bench_overrides": 0,
        }

    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float],
        features: list[dict[str, Any]] | None,
    ) -> int:
        self.stats["decisions_seen"] += 1
        if not features or not (0 <= index < len(features)):
            return index
        representatives = sorted(
            slot for slot in scores if 0 <= slot < len(features)
        )
        if not representatives:
            return index

        current = observation.get("current") or {}
        turn = own_turn(current)
        context = int(select.get("context", -1))

        if 1 <= turn <= 3:
            self.stats["opening_route_considered"] += 1
            moved, reason = self._opening_route(
                context, index, representatives, scores, features
            )
            if moved != index:
                self.stats["opening_route_overrides"] += 1
                self.stats[f"opening_{reason}_overrides"] += 1
                return moved

        if turn <= 5 and _opponent_has_alakazam(current):
            self.stats["alakazam_continuity_considered"] += 1
            moved, reason = self._alakazam_continuity(
                context, index, representatives, scores, features
            )
            if moved != index:
                self.stats["alakazam_continuity_overrides"] += 1
                self.stats[f"alakazam_{reason}_overrides"] += 1
                return moved
        return index

    @staticmethod
    def _opening_route(
        context: int,
        index: int,
        representatives: list[int],
        scores: dict[int, float],
        rows: list[dict[str, Any]],
    ) -> tuple[int, str]:
        state = rows[index]
        if int(state.get("attacker_body_count", 0)) > 0:
            return index, "complete"

        if context == MAIN_CONTEXT:
            finish = [
                slot
                for slot in representatives
                if int(rows[slot].get("candidate_is_grimmsnarl", 0))
                and (
                    int(rows[slot].get("evolve_into_attacker", 0))
                    or int(rows[slot].get("candy_into_attacker", 0))
                    or int(rows[slot].get("triggers_punk_up", 0))
                )
            ]
            moved = _best(finish, scores)
            return (moved, "complete") if moved is not None else (index, "complete")

        if context != DECK_SEARCH_CONTEXT:
            return index, "complete"

        # This is the exact missing-Rare-Candy / missing-Grimmsnarl shape from
        # the two slow Alakazam losses: one search result completes the route
        # already represented by the visible hand and board.
        finish = [
            slot
            for slot in representatives
            if int(rows[slot].get("ctx_completes_candy_route", 0))
            or (
                int(rows[slot].get("candidate_is_grimmsnarl", 0))
                and int(rows[slot].get("morgrem_route_available", 0))
            )
        ]
        moved = _best(finish, scores)
        if moved is not None:
            return moved, "complete"

        # With an Impidimp already in play, another Impidimp does not shorten
        # the first-attack ETA.  Morgrem does.  This occurred in episode
        # 92623138, where both learned pilot conditions chose the extra Basic.
        field_impidimp = int(state.get(f"field_{IMPIDIMP}", 0))
        field_morgrem = int(state.get(f"field_{MORGREM}", 0))
        if field_impidimp > 0 and field_morgrem == 0:
            bridges = [
                slot
                for slot in representatives
                if int(rows[slot].get("candidate_is_morgrem", 0))
            ]
            moved = _best(bridges, scores)
            if moved is not None:
                return moved, "bridge"

        if int(state.get("marnie_body_count", 0)) == 0:
            basics = [
                slot
                for slot in representatives
                if int(rows[slot].get("candidate_is_impidimp", 0))
            ]
            moved = _best(basics, scores)
            if moved is not None:
                return moved, "first_body"
        return index, "complete"

    @staticmethod
    def _alakazam_continuity(
        context: int,
        index: int,
        representatives: list[int],
        scores: dict[int, float],
        rows: list[dict[str, Any]],
    ) -> tuple[int, str]:
        state = rows[index]
        if (
            int(state.get("attacker_body_count", 0)) < 1
            or int(state.get("backup_attacker_ready", 0)) > 0
        ):
            return index, "search"

        if context == DECK_SEARCH_CONTEXT:
            finish = [
                slot
                for slot in representatives
                if int(rows[slot].get("ctx_completes_candy_route", 0))
            ]
            moved = _best(finish, scores)
            if moved is not None:
                return moved, "search"

            # A visible Impidimp is the durable state; fetch its bridge before
            # optional Froslass pieces.  This directly addresses episode
            # 92641973, where the only ready attacker fell and four turns had
            # no attack afterwards.
            if int(state.get(f"field_{IMPIDIMP}", 0)) > 0:
                bridges = [
                    slot
                    for slot in representatives
                    if int(rows[slot].get("candidate_is_morgrem", 0))
                ]
                moved = _best(bridges, scores)
                if moved is not None:
                    return moved, "search"

            if int(state.get("marnie_body_count", 0)) < 2:
                basics = [
                    slot
                    for slot in representatives
                    if int(rows[slot].get("candidate_is_impidimp", 0))
                ]
                moved = _best(basics, scores)
                if moved is not None:
                    return moved, "search"
            return index, "search"

        if context != MAIN_CONTEXT:
            return index, "search"

        evolutions = [
            slot
            for slot in representatives
            if int(rows[slot].get("candidate_is_grimmsnarl", 0))
            and (
                int(rows[slot].get("evolve_into_attacker", 0))
                or int(rows[slot].get("candy_into_attacker", 0))
                or int(rows[slot].get("triggers_punk_up", 0))
            )
        ]
        moved = _best(evolutions, scores)
        if moved is not None:
            return moved, "evolve"

        if int(state.get("marnie_body_count", 0)) < 2:
            basics = [
                slot
                for slot in representatives
                if int(rows[slot].get("candidate_is_impidimp", 0))
                and int(rows[slot].get("is_bench", 0))
            ]
            moved = _best(basics, scores)
            if moved is not None:
                return moved, "bench"

        return index, "search"

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)
