"""Post-Shadow mirror invariant: never pass an immediate larger prize KO.

The v18 audit separates target availability from target preference over 152
exact-60 mirrors: 35 deployed v15 games and 117 games from three public pilots
rated 1114--1151.  Broadly preferring a damaged Grimmsnarl ex is wrong.  For
example, putting 30 on a 40-HP Munkidori can let Freezing Shroud take a Prize
at the next checkup, while putting it on a Grimmsnarl merely stores damage.

One state is outcome-complete rather than heuristic: if the damage being
placed *now* knocks out an opponent's Benched Pokemon, take the legal target
worth the most Prize cards.  The deployed games took all 25/25 immediate
Shadow Bullet Bench KOs.  The high-rated pilots took 103/103, including 5/5
Adrena-Brain KOs of a two-Prize Benched Grimmsnarl while a one-Prize target was
also offered.  Replaying those five boards through v17 gave the same answer
5/5.  This guard therefore seals an already saturated invariant and does not
invent a wider target preference from outcome-correlated boards.

It is mirror-only and activates only after this seat's first Shadow Bullet.
It sees public board state and legal options only.  ``adjust`` never raises.
"""

from __future__ import annotations

from typing import Any

import ml_features as mf

OPTION_ATTACK = 13
SHADOW_BULLET_ID = mf.SHADOW_BULLET_ID
TARGET_CONTEXTS = frozenset({mf.CTX_DAMAGE_COUNTER, mf.CTX_DAMAGE})


class MirrorPrizeGuard:
    """Return the base index unless an immediate larger Prize KO is proven."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mirror = False
        self._saw_shadow = False
        self._last_turn = -1
        self.stats: dict[str, int] = {
            "considered": 0,
            "post_shadow_target_prompts": 0,
            "immediate_ko_prompts": 0,
            "already_max_prizes": 0,
            "overrides": 0,
            "shadow_bench_overrides": 0,
            "adrena_overrides": 0,
            "new_game_detected": 0,
            "errors": 0,
        }

    def set_mirror(self, value: bool) -> None:
        self._mirror = bool(value)

    def note(self, observation: dict[str, Any]) -> None:
        """Detect a reused process starting a new episode."""
        try:
            turn = int((observation.get("current") or {}).get("turn", -1))
            if turn < self._last_turn:
                self._saw_shadow = False
                self.stats["new_game_detected"] += 1
            self._last_turn = turn
        except Exception:  # noqa: BLE001 - a guard must never crash a game
            self.stats["errors"] += 1

    @staticmethod
    def _effect_id(select: dict[str, Any]) -> int:
        return mf._first_nested_id(select.get("effect"))

    @staticmethod
    def _is_shadow(
        current: dict[str, Any], select: dict[str, Any], option: dict[str, Any]
    ) -> bool:
        return (
            int(select.get("context", -1)) == mf.MAIN_CONTEXT
            and mf.action_type(current, option, select) == "attack"
            and mf._int(option.get("attackId")) == SHADOW_BULLET_ID
        )

    def record(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        chosen: int,
    ) -> None:
        """Advance the first-Shadow gate with an action actually selected."""
        try:
            options = list(select.get("option") or [])
            current = observation.get("current") or {}
            if 0 <= chosen < len(options) and self._is_shadow(
                current, select, options[chosen]
            ):
                self._saw_shadow = True
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1

    @staticmethod
    def _damage(select: dict[str, Any], context: int) -> int:
        if context == mf.CTX_DAMAGE:
            return int(mf.SHADOW_BULLET_BENCH_DAMAGE)
        pending = int(select.get("remainDamageCounter") or 0)
        return 10 * pending if pending > 0 else 0

    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None = None,
    ) -> int:
        """Apply the invariant, then record MAIN actions. Never raises."""
        try:
            self.stats["considered"] += 1
            current = observation.get("current") or {}
            options = list(select.get("option") or [])
            if not 0 <= index < len(options):
                return index

            context = int(select.get("context", -1))
            if context == mf.MAIN_CONTEXT:
                self.record(observation, select, index)
                return index
            if (
                not self._mirror
                or not self._saw_shadow
                or context not in TARGET_CONTEXTS
                or int(select.get("minCount") or 0) > 1
                or int(select.get("maxCount") or 0) != 1
                or len(options) < 2
            ):
                return index
            if context == mf.CTX_DAMAGE_COUNTER:
                if self._effect_id(select) != mf.MUNKIDORI_ID:
                    return index
            elif self._effect_id(select) != mf.GRIMMSNARL_EX_ID:
                return index

            self.stats["post_shadow_target_prompts"] += 1
            damage = self._damage(select, context)
            if damage <= 0:
                return index
            players = current.get("players") or [{}, {}]
            your = int(current.get("yourIndex", 0))
            opponent = players[1 - your] if len(players) > 1 else {}
            stadium_id = mf._stadium_id(current)
            shield_ids = {
                int(card.get("id", -1)) for card in mf._in_play(opponent)
            }
            candidates: list[tuple[int, int]] = []
            for slot, option in enumerate(options):
                card, owner_is_self, area = mf.resolve_option(
                    current, select, option
                )
                card = card or {}
                hp = int(card.get("hp", 0) or 0)
                if (
                    owner_is_self
                    or area != mf.AREA_BENCH
                    or not 0 < hp <= damage
                ):
                    continue
                if (
                    context == mf.CTX_DAMAGE
                    and not mf.bench_snipe_lands(
                        card, stadium_id, shield_ids
                    )
                ):
                    continue
                if (
                    context == mf.CTX_DAMAGE_COUNTER
                    and stadium_id == mf.BATTLE_CAGE_ID
                ):
                    continue
                candidates.append(
                    (slot, mf.prize_value(int(card.get("id", -1))))
                )
            if not candidates:
                return index
            self.stats["immediate_ko_prompts"] += 1
            best_prizes = max(prizes for _, prizes in candidates)
            best = [slot for slot, prizes in candidates if prizes == best_prizes]
            if index in best:
                self.stats["already_max_prizes"] += 1
                return index
            # Preserve the ranker's ordering when multiple equally valuable
            # immediate KOs exist.  Missing scores fall back to option order.
            moved = max(best, key=lambda slot: (scores or {}).get(slot, -1e300))
            self.stats["overrides"] += 1
            key = (
                "adrena_overrides"
                if context == mf.CTX_DAMAGE_COUNTER
                else "shadow_bench_overrides"
            )
            self.stats[key] += 1
            return moved
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1
            return index

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)
