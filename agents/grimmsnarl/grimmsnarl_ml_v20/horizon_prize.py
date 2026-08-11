"""Conservative two-turn Prize tie-breaker for the mirror endgame.

The ranker remains the policy.  This layer may only reorder options whose raw
ranker score is within a small band of its best option, and only after the
first Shadow Bullet in an exact-60 mirror once either player has three or
fewer Prize cards.  It combines the damage already being placed with two
Freezing Shroud checkups, one next-turn Shadow Bullet and at most one future
Adrena-Brain.  Unknown draws and the opponent's hidden choices are never
invented.

The purpose is deliberately narrower than a search policy: distinguish two
targets the fitted model regards as near-ties when one has a strictly higher
observable two-turn Prize ceiling.  Immediate KOs remain protected by
``mirror_prize.py`` and all safety routes run after this layer.
"""

from __future__ import annotations

import os
from typing import Any

import ml_features as mf

OPTION_ATTACK = 13
TARGET_CONTEXTS = frozenset({mf.CTX_DAMAGE_COUNTER, mf.CTX_DAMAGE})
BOSS_CONTEXTS = frozenset({mf.CTX_SWITCH, mf.CTX_TO_ACTIVE})
DEFAULT_SCORE_TOLERANCE = 0.08


def _score_tolerance() -> float:
    try:
        return max(
            0.0,
            float(os.environ.get(
                "GRIMMSNARL_HORIZON_SCORE_TOLERANCE",
                DEFAULT_SCORE_TOLERANCE,
            )),
        )
    except (TypeError, ValueError):
        return DEFAULT_SCORE_TOLERANCE


class HorizonPrizePlanner:
    """Break a ranker near-tie only for a strictly larger two-turn ceiling."""

    def __init__(self) -> None:
        self.score_tolerance = _score_tolerance()
        self.reset()

    def reset(self) -> None:
        self._mirror = False
        self._saw_shadow = False
        self._last_turn = -1
        self.stats: dict[str, int | float] = {
            "considered": 0,
            "eligible_endgame_prompts": 0,
            "score_band_prompts": 0,
            "horizon_differentiated": 0,
            "already_horizon_best": 0,
            "overrides": 0,
            "damage_target_overrides": 0,
            "boss_target_overrides": 0,
            "rejected_score_gap": 0,
            "new_game_detected": 0,
            "errors": 0,
            "score_tolerance_milli": int(round(
                self.score_tolerance * 1000
            )),
        }

    def set_mirror(self, value: bool) -> None:
        self._mirror = bool(value)

    def note(self, observation: dict[str, Any]) -> None:
        try:
            turn = int((observation.get("current") or {}).get("turn", -1))
            if turn < self._last_turn:
                self._saw_shadow = False
                self.stats["new_game_detected"] += 1
            self._last_turn = turn
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1

    @staticmethod
    def _effect_id(select: dict[str, Any]) -> int:
        return mf._first_nested_id(select.get("effect"))

    @staticmethod
    def _is_shadow(
        current: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
    ) -> bool:
        return (
            int(select.get("context", -1)) == mf.MAIN_CONTEXT
            and mf.action_type(current, option, select) == "attack"
            and mf._int(option.get("attackId")) == mf.SHADOW_BULLET_ID
        )

    def record(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        chosen: int,
    ) -> None:
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
    def _sides(
        current: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], int] | None:
        players = current.get("players") or []
        your = int(current.get("yourIndex", 0))
        if len(players) < 2 or not 0 <= your < len(players):
            return None
        return players[your], players[1 - your], your

    @staticmethod
    def _remaining_prizes(player: dict[str, Any]) -> int:
        prize = player.get("prize")
        if isinstance(prize, list):
            return len(prize)
        return int(player.get("prizeCount", 6) or 0)

    @staticmethod
    def _ready_shadow(me: dict[str, Any]) -> bool:
        return any(
            int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf._dark_energy_count(card) >= mf.SHADOW_BULLET_COST
            for card in mf._in_play(me)
        )

    @staticmethod
    def _future_adrena_damage(me: dict[str, Any]) -> int:
        ready = sum(
            int(
                int(card.get("id", -1)) == mf.MUNKIDORI_ID
                and mf._dark_energy_count(card) > 0
            )
            for card in mf._in_play(me)
        )
        counter_supply = sum(
            max(
                0,
                int(card.get("maxHp", 0) or 0)
                - int(card.get("hp", 0) or 0),
            )
            for card in mf._in_play(me)
        )
        # A single future move is enough to distinguish the common 40/60/90
        # HP breakpoints.  Assuming every Munkidori fires would turn a ceiling
        # into a hidden-opportunity model and overvalue fragile lines.
        return 30 if ready > 0 and counter_supply >= 30 else 0

    @staticmethod
    def _shroud_damage(
        card: dict[str, Any],
        opponent: dict[str, Any],
        stadium_id: int,
        froslass_count: int,
    ) -> int:
        if froslass_count <= 0:
            return 0
        serial = mf._int(card.get("serial"), -1)
        vulnerable = any(
            mf._int(target.get("serial"), -2) == serial
            for target in mf.shroud_side(
                opponent, stadium_id, is_own_side=False
            )
        )
        # Two checkups occur before our next turn's attack.
        return 20 * froslass_count if vulnerable else 0

    def _damage_value(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
        me: dict[str, Any],
        opponent: dict[str, Any],
        your: int,
    ) -> tuple[int, int, int, int] | None:
        context = int(select.get("context", -1))
        card, owner_is_self, area = mf.resolve_option(current, select, option)
        if not card or owner_is_self or area != mf.AREA_BENCH:
            return None
        stadium_id = mf._stadium_id(current)
        shield_ids = {
            int(body.get("id", -1)) for body in mf._in_play(opponent)
        }
        if context == mf.CTX_DAMAGE:
            if self._effect_id(select) != mf.GRIMMSNARL_EX_ID:
                return None
            immediate = (
                mf.SHADOW_BULLET_BENCH_DAMAGE
                if mf.bench_snipe_lands(card, stadium_id, shield_ids)
                else 0
            )
        elif context == mf.CTX_DAMAGE_COUNTER:
            if self._effect_id(select) != mf.MUNKIDORI_ID:
                return None
            blocked = stadium_id == mf.BATTLE_CAGE_ID
            immediate = 0 if blocked else min(
                30, 10 * int(select.get("remainDamageCounter") or 0)
            )
        else:
            return None
        hp = int(card.get("hp", 0) or 0)
        if hp <= 0 or immediate <= 0:
            return None
        froslass_count = sum(
            int(int(body.get("id", -1)) == mf.FROSLASS_ID)
            for body in mf._in_play(me)
        )
        automatic = self._shroud_damage(
            card, opponent, stadium_id, froslass_count
        )
        next_shadow = (
            mf.SHADOW_BULLET_BENCH_DAMAGE
            if self._ready_shadow(me)
            and mf.bench_snipe_lands(card, stadium_id, shield_ids)
            else 0
        )
        next_adrena = self._future_adrena_damage(me)
        # The in-flight Adrena-Brain is already represented by ``immediate``;
        # do not claim the same Munkidori twice in the two-turn ceiling.
        if context == mf.CTX_DAMAGE_COUNTER:
            next_adrena = 0
        projected = immediate + automatic + next_shadow + next_adrena
        prizes = mf.prize_value(int(card.get("id", -1)))
        ko_prizes = prizes if hp <= projected else 0
        if hp <= immediate:
            eta = 0
        elif hp <= immediate + automatic:
            eta = 1
        elif hp <= projected:
            eta = 2
        else:
            eta = 9
        remaining = max(0, hp - projected)
        return ko_prizes, -eta, -remaining, prizes

    def _boss_value(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
        me: dict[str, Any],
        opponent: dict[str, Any],
        your: int,
    ) -> tuple[int, int, int, int] | None:
        card, owner_is_self, area = mf.resolve_option(current, select, option)
        if not card or owner_is_self or area != mf.AREA_BENCH:
            return None
        # A generic promotion prompt shares these context ids.  Boss targets
        # are the all-opponent form and require a ready current attacker.
        if not self._ready_shadow(me):
            return None
        routes = mf.turn_routes(current, opponent)
        entry = next(
            (
                route for route in routes["per_target"]
                if route["index"] == mf._int(option.get("index"))
                and route["card_id"] == int(card.get("id", -1))
            ),
            None,
        )
        if entry is None:
            return None
        current_prizes = int(entry["total"])
        hp = int(card.get("hp", 0) or 0)
        first = int(entry["shadow_damage"])
        froslass_count = sum(
            int(int(body.get("id", -1)) == mf.FROSLASS_ID)
            for body in mf._in_play(me)
        )
        shroud = self._shroud_damage(
            card, opponent, mf._stadium_id(current), froslass_count
        )
        future = first + shroud + self._future_adrena_damage(me)
        target_prizes = mf.prize_value(int(card.get("id", -1)))
        delayed = int(first < hp <= first + future) * target_prizes
        two_turn = current_prizes + delayed
        remaining = max(0, hp - first - future)
        return two_turn, current_prizes, -remaining, target_prizes

    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None = None,
    ) -> int:
        """Return the base option or a near-tied larger-ceiling option."""
        try:
            self.stats["considered"] += 1
            options = list(select.get("option") or [])
            if not 0 <= index < len(options):
                return index
            context = int(select.get("context", -1))
            current = observation.get("current") or {}
            if context == mf.MAIN_CONTEXT:
                self.record(observation, select, index)
                return index
            if (
                not self._mirror
                or not self._saw_shadow
                or context not in TARGET_CONTEXTS | BOSS_CONTEXTS
                or int(select.get("minCount") or 0) > 1
                or int(select.get("maxCount") or 0) != 1
                or len(options) < 2
                or not scores
            ):
                return index
            sides = self._sides(current)
            if sides is None:
                return index
            me, opponent, your = sides
            if min(
                self._remaining_prizes(me),
                self._remaining_prizes(opponent),
            ) > 3:
                return index
            self.stats["eligible_endgame_prompts"] += 1

            finite = {
                slot: float(score) for slot, score in scores.items()
                if 0 <= slot < len(options)
            }
            if index not in finite or not finite:
                return index
            best_score = max(finite.values())
            if best_score - finite[index] > self.score_tolerance + 1e-12:
                self.stats["rejected_score_gap"] += 1
                return index
            band = [
                slot for slot, score in finite.items()
                if best_score - score <= self.score_tolerance + 1e-12
            ]
            if len(band) < 2:
                self.stats["rejected_score_gap"] += 1
                return index
            self.stats["score_band_prompts"] += 1

            value_fn = (
                self._damage_value
                if context in TARGET_CONTEXTS else self._boss_value
            )
            values = {
                slot: value_fn(
                    current, select, options[slot], me, opponent, your
                )
                for slot in band
            }
            values = {
                slot: value for slot, value in values.items()
                if value is not None
            }
            if index not in values or len(values) < 2:
                return index
            best_value = max(values.values())
            best = [slot for slot, value in values.items() if value == best_value]
            if values[index] == best_value:
                self.stats["already_horizon_best"] += 1
                return index
            if best_value[0] <= values[index][0]:
                # ETA/remaining HP only breaks a tie *after* both routes have
                # the same non-zero Prize ceiling.  It never turns speculative
                # damage into a reason to abandon the ranker's target.
                return index
            self.stats["horizon_differentiated"] += 1
            moved = max(best, key=lambda slot: finite[slot])
            self.stats["overrides"] += 1
            key = (
                "damage_target_overrides"
                if context in TARGET_CONTEXTS else "boss_target_overrides"
            )
            self.stats[key] += 1
            return moved
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1
            return index

    def snapshot(self) -> dict[str, int | float]:
        return dict(self.stats)
