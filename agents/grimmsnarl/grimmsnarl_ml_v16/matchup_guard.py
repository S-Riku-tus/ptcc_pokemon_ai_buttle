"""Narrow matchup safety gates layered over the frozen v8 policy."""

from __future__ import annotations

from typing import Any

import ml_features as mf


class WallSafetyGuard:
    """Use fallback for one development action, never as the wall policy.

    A zero-damage Shadow Bullet is kept when it takes a Bench prize or when the
    only fallback answer is another closing action.  That last case is
    intentional: archived top pilots commonly take the free swing after all
    useful development is exhausted.  The guard fires only when v8 would close
    the turn and fallback can still develop the board or play a useful card.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stats: dict[str, int] = {
            "considered": 0,
            "shadow_bullet_selected": 0,
            "active_walled": 0,
            "bench_prize_kept": 0,
            "invalid_fallback_kept": 0,
            "closing_fallback_kept": 0,
            "development_overrides": 0,
            "errors": 0,
        }

    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        rule_choice: Any,
    ) -> int:
        try:
            self.stats["considered"] += 1
            if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
                return index
            options = list(select.get("option") or [])
            if not 0 <= index < len(options):
                return index

            current = observation.get("current") or {}
            chosen = options[index]
            if (
                mf.action_type(current, chosen, select) != "attack"
                or mf._int(chosen.get("attackId")) != mf.SHADOW_BULLET_ID
            ):
                return index
            self.stats["shadow_bullet_selected"] += 1

            players = current.get("players") or []
            your = int(current.get("yourIndex", 0))
            if len(players) < 2 or not 0 <= your < len(players):
                return index
            opponent = players[1 - your]
            active = (mf._cards(opponent, "active") or [{}])[0]
            if int(active.get("id", -1)) < 0:
                return index
            stadium_id = mf._stadium_id(current)
            if mf.shadow_damage_to(active, stadium_id) > 0.0:
                return index
            self.stats["active_walled"] += 1

            # Do not throw away a prize already secured by Bench-30.
            if mf.turn_routes(current, opponent)["no_boss_prizes"] > 0:
                self.stats["bench_prize_kept"] += 1
                return index

            if not (
                isinstance(rule_choice, list)
                and len(rule_choice) == 1
                and isinstance(rule_choice[0], int)
                and 0 <= rule_choice[0] < len(options)
            ):
                self.stats["invalid_fallback_kept"] += 1
                return index
            alternative = int(rule_choice[0])
            alternative_action = mf.action_type(
                current, options[alternative], select
            )
            if alternative_action in ("attack", "end"):
                self.stats["closing_fallback_kept"] += 1
                return index

            self.stats["development_overrides"] += 1
            return alternative
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1
            return index

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)
