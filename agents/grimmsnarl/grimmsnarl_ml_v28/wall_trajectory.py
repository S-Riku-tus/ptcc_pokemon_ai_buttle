"""Build and preserve the non-ex attacker route before a wall becomes Active.

The older WallBreakGuard is reactive: it is excellent once the opposing
Active provably blanks Shadow Bullet.  v25's logs show the loss often occurs
earlier, however.  Energy is split across two Morgrem, the last Morgrem is
evolved away, or the fifth Bench slot is occupied before a breaker is ready.

This guard becomes sticky only after a wall card is public.  It then makes
four narrowly mechanical interventions: concentrate spare Punk Up Energy on
one Morgrem, redirect a non-critical manual attachment that completes it,
preserve the last breaker when a ready Grimmsnarl ex already exists, and keep
the last Bench slot open when a third Munkidori would remove every remaining
route.  It does not choose attacks and never runs in a non-wall matchup.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import ml_features as mf


OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_END = 14


class WallTrajectoryGuard:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.wall_known = False
        self._last_turn = -1
        self._committed_serial: int | None = None
        self.stats: Counter[str] = Counter()
        self.records: list[dict[str, Any]] = []

    def note(self, observation: dict[str, Any], *, wall_known: bool) -> None:
        current = observation.get("current") or {}
        turn = int(current.get("turn", -1))
        if turn < self._last_turn:
            self.wall_known = False
            self._committed_serial = None
            self.stats["new_games"] += 1
        self._last_turn = turn
        self.wall_known = self.wall_known or bool(wall_known)
        if not self.wall_known:
            return
        me, _opponent = self._sides(current)
        if me is None:
            return
        morgrems = [
            body for body in mf._in_play(me)
            if int(body.get("id", -1)) == mf.MORGREM_ID
        ]
        serials = {self._serial(body) for body in morgrems}
        if self._committed_serial not in serials:
            target = max(
                morgrems,
                key=lambda body: (
                    mf._dark_energy_count(body),
                    float(body.get("hp", 0) or 0),
                ),
                default=None,
            )
            self._committed_serial = self._serial(target)

    @staticmethod
    def _serial(card: dict[str, Any] | None) -> int | None:
        if not isinstance(card, dict):
            return None
        value = card.get("serial")
        return int(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _sides(current: dict[str, Any]):
        players = current.get("players") or []
        your = int(current.get("yourIndex", 0) or 0)
        if len(players) < 2 or your not in (0, 1):
            return None, None
        return players[your], players[1 - your]

    def _record(self, observation: dict[str, Any], kind: str,
                before: int, after: int) -> int:
        self.stats[kind] += 1
        self.stats["overrides"] += int(before != after)
        if before != after and len(self.records) < 24:
            current = observation.get("current") or {}
            self.records.append({
                "turn": int(current.get("turn", -1)),
                "kind": kind,
                "from_index": before,
                "to_index": after,
                "committed_serial": self._committed_serial,
            })
        return after

    def _option_for_serial(self, current: dict[str, Any],
                           select: dict[str, Any], serial: int | None) -> int | None:
        if serial is None:
            return None
        for index, option in enumerate(select.get("option") or []):
            card, is_self, _area = mf.resolve_option(current, select, option)
            if is_self and self._serial(card) == serial:
                return index
        return None

    @staticmethod
    def _end_option(options: list[dict[str, Any]]) -> int | None:
        for index, option in enumerate(options):
            if int(option.get("type", -1)) == OPTION_END:
                return index
        return None

    def _safe_alternative(self, current: dict[str, Any],
                          select: dict[str, Any], unsafe: int,
                          rule_choice: Any) -> int | None:
        options = list(select.get("option") or [])
        if isinstance(rule_choice, list) and len(rule_choice) == 1:
            candidate = rule_choice[0]
            if isinstance(candidate, int) and 0 <= candidate < len(options):
                if candidate != unsafe:
                    return candidate
        return self._end_option(options)

    def _punk_target(self, observation: dict[str, Any],
                     select: dict[str, Any], index: int,
                     me: dict[str, Any]) -> int:
        effect = select.get("effect")
        if not isinstance(effect, dict) or int(effect.get("id", -1)) != mf.GRIMMSNARL_EX_ID:
            return index
        trigger_serial = self._serial(effect)
        trigger = next((
            body for body in mf._in_play(me)
            if self._serial(body) == trigger_serial
        ), None)
        # Punk Up's first invariant is still to make its own Grimmsnarl ready.
        if trigger is None or mf._dark_energy_count(trigger) < 2:
            self.stats["punk_trigger_not_ready"] += 1
            return index
        morgrems = [
            body for body in mf._in_play(me)
            if int(body.get("id", -1)) == mf.MORGREM_ID
            and mf._dark_energy_count(body) < 2
        ]
        if not morgrems:
            return index
        target = next((
            body for body in morgrems
            if self._serial(body) == self._committed_serial
        ), None)
        if target is None:
            target = max(morgrems, key=mf._dark_energy_count)
            self._committed_serial = self._serial(target)
        wanted = self._option_for_serial(
            observation.get("current") or {}, select, self._committed_serial
        )
        if wanted is None or wanted == index:
            self.stats["punk_concentrated_kept"] += int(wanted == index)
            return index
        return self._record(observation, "punk_concentrated", index, wanted)

    def _redirect_attachment(self, observation: dict[str, Any],
                             select: dict[str, Any], index: int,
                             me: dict[str, Any]) -> int:
        options = list(select.get("option") or [])
        chosen = options[index]
        if int(chosen.get("type", -1)) != OPTION_ATTACH:
            return index
        energy = mf.candidate_card(
            observation.get("current") or {}, chosen, select
        )
        if int((energy or {}).get("id", -1)) != mf.DARK_ENERGY_ID:
            return index
        current = observation.get("current") or {}
        chosen_target = mf.candidate_target(current, chosen)
        # Do not steal the attachment which makes the primary attacker live.
        if (
            int((chosen_target or {}).get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf._dark_energy_count(chosen_target or {}) < 2
        ):
            self.stats["primary_attachment_preserved"] += 1
            return index
        # Adrena-Brain is the other half of the wall route.  Its first Energy
        # is productive every turn, so only a redundant second Energy may be
        # redirected from Munkidori to complete Morgrem.
        if (
            int((chosen_target or {}).get("id", -1)) == mf.MUNKIDORI_ID
            and mf._dark_energy_count(chosen_target or {}) < 1
        ):
            self.stats["munkidori_activation_preserved"] += 1
            return index
        targets = [
            body for body in mf._in_play(me)
            if int(body.get("id", -1)) == mf.MORGREM_ID
            and mf._dark_energy_count(body) == 1
        ]
        target = next((
            body for body in targets
            if self._serial(body) == self._committed_serial
        ), None)
        if target is None and targets:
            target = targets[0]
            self._committed_serial = self._serial(target)
        if target is None or self._serial(chosen_target) == self._serial(target):
            return index
        for wanted, option in enumerate(options):
            if int(option.get("type", -1)) != OPTION_ATTACH:
                continue
            option_energy = mf.candidate_card(current, option, select)
            option_target = mf.candidate_target(current, option)
            if (
                int((option_energy or {}).get("id", -1)) == mf.DARK_ENERGY_ID
                and self._serial(option_target) == self._serial(target)
            ):
                return self._record(
                    observation, "manual_attachment_completed", index, wanted
                )
        return index

    def _preserve_breaker(self, observation: dict[str, Any],
                          select: dict[str, Any], index: int,
                          rule_choice: Any, me: dict[str, Any]) -> int:
        current = observation.get("current") or {}
        options = list(select.get("option") or [])
        chosen = options[index]
        if int(chosen.get("type", -1)) != OPTION_EVOLVE:
            return index
        evolution = mf.candidate_card(current, chosen, select)
        target = mf.candidate_target(current, chosen)
        if (
            int((evolution or {}).get("id", -1)) != mf.GRIMMSNARL_EX_ID
            or int((target or {}).get("id", -1)) != mf.MORGREM_ID
        ):
            return index
        breakers = [
            body for body in mf._in_play(me)
            if int(body.get("id", -1)) in (mf.IMPIDIMP_ID, mf.MORGREM_ID)
        ]
        ready_grimmsnarl = any(
            int(body.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf._dark_energy_count(body) >= 2
            for body in mf._in_play(me)
        )
        if len(breakers) != 1 or not ready_grimmsnarl:
            return index
        alternative = self._safe_alternative(
            current, select, index, rule_choice
        )
        if alternative is None:
            return index
        return self._record(
            observation, "last_breaker_preserved", index, alternative
        )

    def _reserve_bench(self, observation: dict[str, Any],
                       select: dict[str, Any], index: int,
                       me: dict[str, Any]) -> int:
        current = observation.get("current") or {}
        options = list(select.get("option") or [])
        chosen = options[index]
        if int(chosen.get("type", -1)) != OPTION_PLAY:
            return index
        card = mf.candidate_card(current, chosen, select)
        if int((card or {}).get("id", -1)) != mf.MUNKIDORI_ID:
            return index
        bench = mf._cards(me, "bench")
        counts = Counter(int(body.get("id", -1)) for body in mf._in_play(me))
        ready_breaker = any(
            int(body.get("id", -1)) == mf.MORGREM_ID
            and mf._dark_energy_count(body) >= 2
            for body in mf._in_play(me)
        )
        if (
            len(bench) < 4
            or counts[mf.MUNKIDORI_ID] < 2
            or counts[mf.SNORUNT_ID] + counts[mf.FROSLASS_ID] > 0
            or ready_breaker
        ):
            return index
        # Use the last slot for the missing route if it is offered now.
        wanted_id = (
            mf.SNORUNT_ID
            if counts[mf.SNORUNT_ID] + counts[mf.FROSLASS_ID] == 0
            else mf.IMPIDIMP_ID
        )
        for wanted, option in enumerate(options):
            option_card = mf.candidate_card(current, option, select)
            if (
                int(option.get("type", -1)) == OPTION_PLAY
                and int((option_card or {}).get("id", -1)) == wanted_id
            ):
                return self._record(
                    observation, "last_bench_route_reserved", index, wanted
                )
        end = self._end_option(options)
        if end is not None:
            return self._record(
                observation, "last_bench_slot_kept_open", index, end
            )
        return index

    def adjust(self, observation: dict[str, Any], select: dict[str, Any],
               index: int, rule_choice: Any = None) -> int:
        try:
            if not self.wall_known:
                self.stats["skip_non_wall"] += 1
                return index
            options = list(select.get("option") or [])
            if not 0 <= index < len(options) or int(select.get("maxCount") or 0) != 1:
                return index
            current = observation.get("current") or {}
            me, _opponent = self._sides(current)
            if me is None:
                return index
            self.stats["considered"] += 1
            context = int(select.get("context", -1))
            if context == mf.CTX_ATTACH_FROM:
                return self._punk_target(observation, select, index, me)
            if context != mf.MAIN_CONTEXT:
                return index
            index = self._redirect_attachment(observation, select, index, me)
            index = self._preserve_breaker(
                observation, select, index, rule_choice, me
            )
            return self._reserve_bench(observation, select, index, me)
        except Exception:  # every unexpected shape is a safe no-op
            self.stats["errors"] += 1
            return index

    def snapshot(self) -> dict[str, Any]:
        return {
            **dict(self.stats),
            "wall_known": self.wall_known,
            "committed_serial": self._committed_serial,
            "records": list(self.records),
        }
