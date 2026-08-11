"""Attack-access invariants over the frozen v8 policy.

What this fixes
---------------

v14 is v8 plus two narrow residuals, and its 42 rated ladder games say the
board-building half of the policy is not the problem: 2.21 Grimmsnarl ex
evolutions a game against the rank-3 pilot's 1.95, Boss's Orders converting to
an attack on 79% of plays against their 82%, Unfair Stamp on 90%.  What it does
not do is *start attacking*.  First Shadow Bullet lands on turn 3.60 against
their 2.96, and the tail is entirely in the states where the Active is not the
attacker: 4.11 with an opening Snorunt against 3.13, 3.52 with an opening
Impidimp against 2.87.  Episode 91548124 is the extreme case - two Grimmsnarl
ex finished on our second turn, the opening Snorunt still Active, ten Basic
Darkness spread over Grimmsnarl, Impidimp and Munkidori, none on the Snorunt,
and the first Shadow Bullet on turn 15.

The cause is a distinction the policy never draws.  ``ready_grimms()`` asks
whether a Grimmsnarl ex *can pay Shadow Bullet*; nothing asks whether that body
can be **in the Active spot this turn**.  Retreat is the only switch this deck
runs, and every non-Grimmsnarl body it plays retreats for exactly one Energy
(Snorunt, Munkidori, Impidimp, Morgrem and Froslass all cost 1; Grimmsnarl ex
costs 2), so the whole route is

    attach one Darkness to the Active -> retreat -> promote -> Shadow Bullet

and the resource it needs is the once-a-turn manual attachment.  That is the
one resource every other useful play also wants, which is why the ranker loses
it: on the boards above it attached to Munkidori instead, and after that no
amount of later scoring can re-open the route.

Why a hard invariant rather than a feature or a preference
----------------------------------------------------------

v8's rule policy already scores exactly these two actions above everything else
it can do:

* ``score_attach`` returns 990,000 for a manual attachment onto a
  non-Grimmsnarl Active while a ready Grimmsnarl is on the Bench; and
* ``score_retreat`` returns 995,000 whenever a ready Grimmsnarl is on the Bench
  and the Active cannot Shadow Bullet.

Those are the two highest non-lethal scores in the whole rule set.  But MAIN is
decided by the ranker, so in v14 the rule policy's answer is advisory and these
two rules are effectively unreachable.  This module therefore adds no new
strategy: it makes the ranker respect v8's own two top-priority rules, and it
fires on a **strict subset** of the boards where those rules apply, because it
additionally requires

1. the route to be completable this turn (debt of at most one Energy, the
   manual attachment unused, and the Active neither asleep, paralysed nor
   already retreated);
2. the Shadow Bullet at the end of it to be worth something - real damage to
   their Active, or a Bench-30 that takes a prize now.  A damage-immune wall
   with no Bench prize is left to v14's ``WallSafetyGuard`` exactly as before;
   and
3. the ranker to not already be doing something that reaches an attack this
   turn (attacking, retreating, attaching to the Active, evolving the Active
   into Grimmsnarl ex, or playing the Rare Candy that does).

Route stickiness without a stored plan
--------------------------------------

v12 failed by re-searching every micro-decision, so the intended shape here is
a route that survives its own first step.  It gets that without storing a plan:
the route is recomputed from the board on every decision, and paying the escape
debt is precisely what turns the next recomputation from "attach" into
"retreat".  A plan that cannot go stale cannot be followed off a cliff, and the
only cross-decision state kept is a one-shot flag that makes the promotion
select after our own forced retreat choose the ready attacker.

Three teeth, and why the set is closed
--------------------------------------

Within one turn MAIN repeats until an attack or an END, so no non-terminal
action can *delay* an attack - it can only spend a card.  A turn can therefore
fail to attack in exactly two ways, and both are covered:

* **ACCESS** - the attacker is unreachable, so the route above is forced.
* **CONVERT** - the turn is ended with a worthwhile Shadow Bullet unspent.
* **BRIDGE** - the turn is ended while v8's own scoring says some attack is
  worth making.  The rank-3 pilot uses Filch 47 times over 136 mirror games,
  concentrated on their first three turns; v14's mirror games contain
  essentially no non-Shadow attack at all.  Nothing in this deck's playable
  attacks has a drawback (Filch draws a card, Corkscrew Punch is plain damage),
  and both END and ATTACK end the turn, so taking v8's attack instead of its
  END cannot cost anything but the swing itself.

Deliberately not done here
--------------------------

* No general pre-Shadow setup gate.  The measured pre-first-Shadow counts
  (Poké Pad 1.73 against 1.30, Poffin 1.40 against 1.00) are confounded by
  having more turns before the first Shadow, and because setup does not end the
  turn it cannot be what delays the attack.  Suppressing it inside the trapped
  state falls out of the route forcing above, and that is the only place the
  mechanism is visible.
* No change to ``punk_search_budget``.  Punk Up pulls from the deck and only
  reaches "your Marnie's Pokémon", so it can never pay a Snorunt's or a
  Munkidori's escape; reserving against it would trade a measured fit to the
  elite band (their own count reproduced exactly on 51.9% of activations) for a
  second-order effect on draw probability.
* No new preference over Boss's Orders, Unfair Stamp, Petrel, Froslass or the
  number of Grimmsnarl lines.  Every one of those was re-measured on v14's own
  games and is at or above the rank-3 pilot's conversion rate.
"""

from __future__ import annotations

from typing import Any

import ml_features as mf

try:  # only used for the card table and the Dodge/Hide tracker
    import fallback_policy as _fallback
except Exception:  # noqa: BLE001 - the guard must load without the rule policy
    _fallback = None

MAIN_CONTEXT = mf.MAIN_CONTEXT
PROMOTION_CONTEXTS = (mf.CTX_SWITCH, mf.CTX_TO_ACTIVE)

GRIMMSNARL_EX_ID = mf.GRIMMSNARL_EX_ID
DARK_ENERGY_ID = mf.DARK_ENERGY_ID
IMPIDIMP_ID = mf.IMPIDIMP_ID
RARE_CANDY_ID = mf.RARE_CANDY_ID
SHADOW_BULLET_ID = mf.SHADOW_BULLET_ID
SHADOW_BULLET_COST = mf.SHADOW_BULLET_COST
HANDHELD_FAN_ID = 1161
HANDHELD_FAN_REDUCTION = 2

# Retreat cost of every body this deck can put in the Active spot, from the
# card database. Every one of them except Grimmsnarl ex costs a single Energy,
# which is what makes "one manual attachment" a complete escape route rather
# than a special case. Asserted against the database in the v15 tests.
RETREAT_COST = {
    IMPIDIMP_ID: 1,
    mf.MORGREM_ID: 1,
    GRIMMSNARL_EX_ID: 2,
    mf.SNORUNT_ID: 1,
    mf.FROSLASS_ID: 1,
    mf.MUNKIDORI_ID: 1,
}

OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_CARD = 3
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14


class AttackAccessGuard:
    """Never let a turn pass with a reachable Shadow Bullet unplayed."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._turn = -2
        self._pending_promote = False
        self._trapped_seen = False
        self.stats: dict[str, int] = {
            "considered": 0,
            "main_decisions": 0,
            # the state the analysis asked to be measured
            "trapped_turns": 0,
            "trapped_turns_worth": 0,
            "no_route_available": 0,
            "route_compatible_kept": 0,
            # the route itself
            "enable_attach_forced": 0,
            "escape_attach_forced": 0,
            "retreat_forced": 0,
            "attack_overridden_for_route": 0,
            "promote_forced": 0,
            "promote_already_right": 0,
            # conversion
            "end_replaced_by_shadow": 0,
            "end_replaced_by_bridge": 0,
            # played-action telemetry (this layer returns the played index)
            "played_shadow_bullets": 0,
            "played_other_attacks": 0,
            "played_ends": 0,
            "ends_with_ready_attacker": 0,
            "new_game_detected": 0,
            "errors": 0,
        }

    # ----- turn bookkeeping -------------------------------------------------
    def note(self, observation: dict[str, Any]) -> None:
        """Drop per-turn state whenever the turn counter moves.

        Nothing calls ``reset`` between episodes on Kaggle, so a turn number
        that goes backwards is a new game in a reused process.
        """
        try:
            turn = int((observation.get("current") or {}).get("turn", -1))
            if turn == self._turn:
                return
            if turn < self._turn:
                self.stats["new_game_detected"] += 1
            self._turn = turn
            self._pending_promote = False
            self._trapped_seen = False
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1

    # ----- entry point ------------------------------------------------------
    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        rule_choice: Any = None,
    ) -> int:
        """The index to play: the caller's, or the route's replacement."""
        try:
            self.stats["considered"] += 1
            options = list(select.get("option") or [])
            if not 0 <= index < len(options):
                return index
            if int(select.get("maxCount") or 0) != 1:
                return index  # multi-pick selects belong to the rule policy
            context = int(select.get("context", -1))
            current = observation.get("current") or {}
            me, opponent = self._sides(current)
            if me is None:
                return index
            if context in PROMOTION_CONTEXTS:
                return self._promote(current, select, options, index)
            if context != MAIN_CONTEXT:
                return index
            return self._main(
                current, select, options, index, me, opponent, rule_choice
            )
        except Exception:  # noqa: BLE001 - any surprise means v14's answer
            self.stats["errors"] += 1
            return index

    # ----- MAIN -------------------------------------------------------------
    def _main(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        index: int,
        me: dict[str, Any],
        opponent: dict[str, Any],
        rule_choice: Any,
    ) -> int:
        self.stats["main_decisions"] += 1
        active = (mf._cards(me, "active") or [None])[0]
        if not isinstance(active, dict) or int(active.get("id", -1)) < 0:
            return index
        chosen = options[index]
        chosen_type = mf._int(chosen.get("type"))

        ready_bench = self._ready_bench(me)
        active_ready = self._is_ready_attacker(active)
        worth = self._shadow_worth(current, opponent)

        # The policy's own retreat is a route too: keep the promotion honest.
        if chosen_type == OPTION_RETREAT and ready_bench:
            self._pending_promote = True

        if ready_bench and not active_ready and not self._trapped_seen:
            self._trapped_seen = True
            self.stats["trapped_turns"] += 1
            self.stats["trapped_turns_worth"] += int(worth)

        # ---- ACCESS ---------------------------------------------------------
        # Only a board that owns an attacker can be denied access to one:
        # either a finished Grimmsnarl ex on the Bench, or one in front that is
        # short of Shadow Bullet. A board with no Grimmsnarl ex at all is v14's
        # to build, not this guard's, and counting it as a closed route would
        # bury the measurement this layer exists to produce.
        has_body = (
            bool(ready_bench)
            or int(active.get("id", -1)) == GRIMMSNARL_EX_ID
        )
        if worth and not active_ready and has_body:
            forced = self._access(
                current, select, options, index, me, active, ready_bench
            )
            if forced is not None:
                return self._played(forced, options)

        # ---- CONVERT --------------------------------------------------------
        if chosen_type == OPTION_END and active_ready:
            self.stats["ends_with_ready_attacker"] += 1
            if worth:
                slot = self._shadow_option(options)
                if slot is not None:
                    self.stats["end_replaced_by_shadow"] += 1
                    return self._played(slot, options)

        # ---- BRIDGE ---------------------------------------------------------
        if chosen_type == OPTION_END:
            slot = self._rule_attack(options, rule_choice)
            if slot is not None:
                self.stats["end_replaced_by_bridge"] += 1
                return self._played(slot, options)

        return self._played(index, options)

    def _access(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        index: int,
        me: dict[str, Any],
        active: dict[str, Any],
        ready_bench: list[dict[str, Any]],
    ) -> int | None:
        """The next step of an ETA-0 attack route, or None."""
        step, kind = self._access_step(
            current, select, options, me, active, ready_bench
        )
        if step is None:
            self.stats["no_route_available"] += 1
            return None
        if step == index:
            self.stats["route_compatible_kept"] += 1
            return None
        chosen = options[index]
        if self._reaches_attack(current, select, chosen, me, active):
            self.stats["route_compatible_kept"] += 1
            return None
        if mf._int(chosen.get("type")) == OPTION_ATTACK:
            # A non-Shadow attack while a full Shadow route is open: Shadow
            # Bullet's 180 + 30 dominates Corkscrew Punch's 60 and Filch's
            # draw, so the route wins. Counted, because it is the only tooth
            # that replaces one closing action with another.
            self.stats["attack_overridden_for_route"] += 1
        self.stats[kind] += 1
        if kind == "retreat_forced":
            self._pending_promote = True
        return step

    def _access_step(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        me: dict[str, Any],
        active: dict[str, Any],
        ready_bench: list[dict[str, Any]],
    ) -> tuple[int | None, str]:
        active_id = int(active.get("id", -1))
        energy = mf._energy_count(active)
        attach_used = bool(current.get("energyAttached"))

        # The Active is the attacker and is one Energy short: fuel it. This is
        # strictly better than retreating to a Bench body, and it is the only
        # route when the Active is Grimmsnarl ex (retreat cost 2).
        if active_id == GRIMMSNARL_EX_ID:
            if SHADOW_BULLET_COST - energy == 1 and not attach_used:
                slot = self._dark_attach_option(current, select, options)
                if slot is not None:
                    return slot, "enable_attach_forced"
            return None, ""

        if not ready_bench:
            return None, ""

        debt = max(0, self._retreat_cost(active) - energy)
        if debt == 0:
            slot = self._retreat_option(options)
            if slot is not None:
                return slot, "retreat_forced"
            return None, ""
        blocked = self._retreat_blocked(current, me)
        if debt == 1 and not attach_used and not blocked:
            slot = self._dark_attach_option(current, select, options)
            if slot is not None:
                return slot, "escape_attach_forced"
        return None, ""

    def _reaches_attack(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
        me: dict[str, Any],
        active: dict[str, Any],
    ) -> bool:
        """The chosen action already reaches an attack this turn."""
        option_type = mf._int(option.get("type"))
        if option_type == OPTION_RETREAT:
            return True
        if option_type == OPTION_ATTACK:
            return mf._int(option.get("attackId")) == SHADOW_BULLET_ID
        if option_type == OPTION_ATTACH:
            return (
                mf._int(option.get("inPlayArea")) == mf.AREA_ACTIVE
                and self._is_dark(mf.candidate_card(current, option, select))
            )
        if option_type == OPTION_EVOLVE:
            card = mf.candidate_card(current, option, select)
            return (
                mf._int(option.get("inPlayArea")) == mf.AREA_ACTIVE
                and int((card or {}).get("id", -1)) == GRIMMSNARL_EX_ID
            )
        if option_type == OPTION_PLAY:
            card = mf.candidate_card(current, option, select)
            if int((card or {}).get("id", -1)) != RARE_CANDY_ID:
                return False
            # Rare Candy on the Active Impidimp lands a Grimmsnarl ex there and
            # Punk Up fuels it, so the Active becomes the attacker without a
            # retreat at all.
            return int(active.get("id", -1)) == IMPIDIMP_ID and any(
                int(card.get("id", -1)) == GRIMMSNARL_EX_ID
                for card in mf._cards(me, "hand")
            )
        return False

    # ----- promotion after a retreat ---------------------------------------
    def _promote(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        index: int,
    ) -> int:
        """Put the ready attacker in front, but only on our own route."""
        if not self._pending_promote:
            return index
        self._pending_promote = False
        best: int | None = None
        best_key: tuple[float, int] | None = None
        for slot, option in enumerate(options):
            if mf._int(option.get("type")) != OPTION_CARD:
                continue
            card, is_self, _area = mf.resolve_option(current, select, option)
            if card is None or not is_self:
                continue
            if not self._is_ready_attacker(card):
                continue
            key = (float(card.get("hp", 0) or 0), mf._energy_count(card))
            if best_key is None or key > best_key:
                best, best_key = slot, key
        if best is None:
            return index
        if best == index:
            self.stats["promote_already_right"] += 1
            return index
        self.stats["promote_forced"] += 1
        return best

    # ----- board reading ----------------------------------------------------
    @staticmethod
    def _sides(
        current: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        players = current.get("players") or []
        your = int(current.get("yourIndex", 0))
        if len(players) < 2 or not 0 <= your < len(players):
            return None, None
        return players[your], players[1 - your]

    @staticmethod
    def _is_ready_attacker(card: dict[str, Any] | None) -> bool:
        if not isinstance(card, dict):
            return False
        return (
            int(card.get("id", -1)) == GRIMMSNARL_EX_ID
            and mf._energy_count(card) >= SHADOW_BULLET_COST
        )

    def _ready_bench(self, me: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            card for card in mf._cards(me, "bench")
            if self._is_ready_attacker(card)
        ]

    @staticmethod
    def _is_dark(card: dict[str, Any] | None) -> bool:
        return int((card or {}).get("id", -1)) == DARK_ENERGY_ID

    @staticmethod
    def _retreat_cost(card: dict[str, Any]) -> int:
        card_id = int(card.get("id", -1))
        cost = RETREAT_COST.get(card_id)
        if cost is None:
            table = (
                getattr(_fallback, "card_table", None) if _fallback else None
            )
            data = table.get(card_id) if isinstance(table, dict) else None
            cost = int(getattr(data, "retreatCost", 1) or 0) if data else 1
        for tool in (card.get("tools") or []):
            if int((tool or {}).get("id", -1)) == HANDHELD_FAN_ID:
                cost = max(0, cost - HANDHELD_FAN_REDUCTION)
        return cost

    @staticmethod
    def _retreat_blocked(current: dict[str, Any], me: dict[str, Any]) -> bool:
        """Retreat is impossible this turn regardless of Energy."""
        return bool(
            current.get("retreated")
            or me.get("asleep")
            or me.get("paralyzed")
        )

    def _shadow_worth(
        self, current: dict[str, Any], opponent: dict[str, Any] | None
    ) -> bool:
        """Shadow Bullet does something: real damage, or a Bench prize now.

        This is the same gate v14's wall guard uses, so a damage-immune Active
        with nothing to snipe keeps v14's behaviour exactly: the route is not
        forced and a fresh two-prize ex is not walked into a wall.
        """
        if opponent is None:
            return False
        stadium_id = mf._stadium_id(current)
        active = (mf._cards(opponent, "active") or [{}])[0]
        if int(active.get("id", -1)) < 0:
            return False
        if (
            mf.shadow_damage_to(active, stadium_id) > 0.0
            and not self._dodged(current, active)
        ):
            return True
        shield_ids = {
            int(card.get("id", -1)) for card in mf._in_play(opponent)
        }
        return any(
            mf.snipe_prizes(card, stadium_id, shield_ids) > 0
            for card in mf._cards(opponent, "bench")
        )

    @staticmethod
    def _dodged(current: dict[str, Any], card: dict[str, Any]) -> bool:
        """That body used a Dodge/Hide attack and takes no damage from us."""
        try:
            table = (
                getattr(_fallback, "TEMP_IMMUNITY", None)
                if _fallback else None
            )
            serial = card.get("serial")
            if not isinstance(table, dict) or serial is None:
                return False
            expiry = table.get(int(serial))
            if expiry is None:
                return False
            return int(current.get("turn", 0)) <= int(expiry)
        except Exception:  # noqa: BLE001
            return False

    # ----- option lookups ---------------------------------------------------
    def _dark_attach_option(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
    ) -> int | None:
        for slot, option in enumerate(options):
            if mf._int(option.get("type")) != OPTION_ATTACH:
                continue
            if mf._int(option.get("inPlayArea")) != mf.AREA_ACTIVE:
                continue
            if self._is_dark(mf.candidate_card(current, option, select)):
                return slot
        return None

    @staticmethod
    def _retreat_option(options: list[dict[str, Any]]) -> int | None:
        for slot, option in enumerate(options):
            if mf._int(option.get("type")) == OPTION_RETREAT:
                return slot
        return None

    @staticmethod
    def _shadow_option(options: list[dict[str, Any]]) -> int | None:
        for slot, option in enumerate(options):
            if (
                mf._int(option.get("type")) == OPTION_ATTACK
                and mf._int(option.get("attackId")) == SHADOW_BULLET_ID
            ):
                return slot
        return None

    @staticmethod
    def _rule_attack(
        options: list[dict[str, Any]], rule_choice: Any
    ) -> int | None:
        """v8's own answer, if v8 would attack here."""
        if not (
            isinstance(rule_choice, list)
            and len(rule_choice) == 1
            and isinstance(rule_choice[0], int)
        ):
            return None
        slot = int(rule_choice[0])
        if not 0 <= slot < len(options):
            return None
        if mf._int(options[slot].get("type")) != OPTION_ATTACK:
            return None
        return slot

    # ----- telemetry --------------------------------------------------------
    def _played(self, index: int, options: list[dict[str, Any]]) -> int:
        option = options[index]
        option_type = mf._int(option.get("type"))
        if option_type == OPTION_ATTACK:
            if mf._int(option.get("attackId")) == SHADOW_BULLET_ID:
                self.stats["played_shadow_bullets"] += 1
            else:
                self.stats["played_other_attacks"] += 1
        elif option_type == OPTION_END:
            self.stats["played_ends"] += 1
        return index

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)
