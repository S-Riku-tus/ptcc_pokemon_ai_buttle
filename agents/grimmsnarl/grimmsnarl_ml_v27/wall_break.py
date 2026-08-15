"""The attacker that can actually hit a damage-immune Active.

What this fixes
---------------

v15 closed the attack-access gap and the 110 rated games say so: first Shadow
Bullet on own turn 2.84 against v14's 3.60, 82.7% of games with one by turn 3,
and the turn of the first Grimmsnarl ex *equals* the turn of the first Shadow
Bullet in the losses (3.048 and 3.048).  Nothing is waiting to attack any more.

What is left is a swing that buys nothing.  Across the two runs there are **84
Shadow Bullets that were provably worth zero when they were thrown**: the
opposing Active prevents all damage from us, and the Bench-30 takes no prize
either.  They are not spread thinly - they are the wall matchup, which is 7-8
with 8.25 Shadow Bullets a game in the losses, 86.4% of them taking no prize,
and a mean deck of 7.6 cards left.  Episode 91663479 is the whole failure in
one game: Cornerstone Mask Ogerpon ex Active, **an empty Bench for all 24
turns**, 21 Shadow Bullets, 0 prizes, deck 0, lost by deck-out while holding
two Boss's Orders that had nothing to gust.

The route out of that state exists and this deck has always had it.  Three
different Abilities wall us and none of them walls everything:

* Crustle and Sylveon prevent damage from *Pokemon ex*;
* Cornerstone Mask Ogerpon ex prevents damage from *Pokemon with an Ability*;
* Neutralization Zone prevents damage to a non-Rule-Box body from *ex and V*.

Marnie's Morgrem is neither a Pokemon ex nor a Pokemon with an Ability, so
Corkscrew Punch's 60 lands through all three, and Marnie's Impidimp's 10 does
too.  Against a 210 HP Cornerstone Mask Ogerpon ex that is four swings, and
episode 91663479 had twenty-five turns to make them.

Why only Morgrem and Impidimp
-----------------------------

Because they are the only two.  Froslass's Frost Smash costs {W}{C} and
Munkidori's Mind Bend costs {P}{C}; the deck's only Energy is Basic Darkness,
so neither can ever attack at all, and Snorunt's Chilly costs {W}.  A wall
breaker in this list is Morgrem for 60 or Impidimp for 10, and nothing else.
``tests/test_v16_wall_break.py`` asserts that against the card database.

Two teeth
---------

* **BREAK** - while a Shadow Bullet is provably worth zero, advance a route to
  the breaker instead of throwing it: attack with it if it is Active and paid
  for, fuel it if it is one Darkness short, retreat to it if it is Benched and
  ready.  The route is recomputed from the board every decision exactly as
  ``attack_access`` does, so it has no plan to go stale.
* **PRESERVE** - do not spend the last breaker on a Grimmsnarl ex evolution
  while that wall is up.  This happened 8 times over the two runs, in 7 of the
  15 wall games including all three worst, and every one of them already had a
  fuelled Grimmsnarl ex in play, so keeping the Morgrem cost nothing.  In
  91663479 it is why 17 of the 21 dead swings had no breaker left on the board.

v17 closes the conservative gap left in that second tooth.  v16 only refused
2 of the 8 evolutions because it required the breaker to be an immediately
viable route step.  In the other 6, the last body was usually an Impidimp: 10
damage was too slow for BREAK, but preserving it was still the only way to
evolve into the 60-damage Morgrem on a later turn.  Every one of the 8 stored
decisions offered a dead Shadow Bullet and END, and a fuelled Grimmsnarl ex was
already in play.  v17 therefore closes the turn with that free swing instead
of consuming the last future breaker.  BREAK remains just as conservative;
only PRESERVE becomes complete.

The same audit closes the Punk Up target that v16 left explicitly unmeasured.
Among 83 stored wall-game allocation decisions, only 3 occur under a dead wall
after the triggering Grimmsnarl can already attack and offer an underfuelled,
route-viable Morgrem.  v16 feeds it on 2; on the third it splits the last two
Energy across Morgrem and Impidimp, leaving neither Morgrem-ready.  v17 keeps
the second Energy on that same Morgrem.  It does not change how many Energy
Punk Up searches, and stands down until the triggering Grimmsnarl is ready.

Deliberately not done here
--------------------------

* **No forced Boss's Orders.** Gusting is the other way to leave a wall, and it
  is already played: over 446 Shadow Bullets there is exactly *one* where a
  playable Boss in hand would have added a prize.  The guard therefore stands
  down whenever a playable Boss would take a prize this turn and leaves that
  decision where it already works.
* **No ban on the zero-damage Shadow Bullet.** Banning it outright is what took
  a previous wall specialist to 1-10: both END and ATTACK close the turn, so a
  worthless swing costs nothing by itself.  This guard only fires when it has
  somewhere better to send the turn.
* **No stall circuit breaker.** A repeated worthless swing does not cause the
  deck-out - the draw at the start of each turn does - so refusing to attack
  does not save a single card.  What loses those games is having no route, and
  the route is what this adds.
"""

from __future__ import annotations

import math
from typing import Any

import ml_features as mf

MAIN_CONTEXT = mf.MAIN_CONTEXT
PROMOTION_CONTEXTS = (mf.CTX_SWITCH, mf.CTX_TO_ACTIVE)
PUNK_TARGET_CONTEXT = mf.CTX_ATTACH_FROM

GRIMMSNARL_EX_ID = mf.GRIMMSNARL_EX_ID
MORGREM_ID = mf.MORGREM_ID
IMPIDIMP_ID = mf.IMPIDIMP_ID
DARK_ENERGY_ID = mf.DARK_ENERGY_ID
BOSS_ID = mf.BOSS_ID
SHADOW_BULLET_ID = mf.SHADOW_BULLET_ID

OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_CARD = 3
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14

# The attacks this deck can actually pay for, as (attack id, Darkness needed,
# printed damage). Basic Darkness is the only Energy in the 60, so an attack
# whose cost names any other type can never be paid: Froslass's Frost Smash is
# {W}{C}, Munkidori's Mind Bend is {P}{C} and Snorunt's Chilly is {W}. All
# three are absent on purpose, and the test asserts that against the database.
DARK_ATTACKS: dict[int, tuple[int, int, float]] = {
    GRIMMSNARL_EX_ID: (SHADOW_BULLET_ID, 2, 180.0),
    MORGREM_ID: (936, 2, 60.0),
    IMPIDIMP_ID: (935, 1, 10.0),
}
# The bodies that break a wall: everything above except the one that is walled.
BREAKER_IDS = frozenset({MORGREM_ID, IMPIDIMP_ID})

# ``mf.EX_DAMAGE_BLOCKER_IDS`` merges two different Abilities. Splitting them
# is the whole point of this module, because our Morgrem is neither a Pokemon
# ex nor a Pokemon with an Ability and is therefore stopped by neither.
EX_BLOCKER_IDS = frozenset({345, 330})       # Crustle, Sylveon
ABILITY_BLOCKER_IDS = frozenset({117})       # Cornerstone Mask Ogerpon ex
OUR_EX_IDS = frozenset({GRIMMSNARL_EX_ID})
OUR_ABILITY_IDS = mf.ABILITY_HOLDER_IDS      # Froslass, Munkidori, Grimmsnarl

# Retreat costs, as in ``attack_access``: every body but Grimmsnarl ex is 1.
RETREAT_COST = {
    IMPIDIMP_ID: 1, MORGREM_ID: 1, GRIMMSNARL_EX_ID: 2,
    mf.SNORUNT_ID: 1, mf.FROSLASS_ID: 1, mf.MUNKIDORI_ID: 1,
}
HANDHELD_FAN_ID = 1161
HANDHELD_FAN_REDUCTION = 2

# A route is only worth starting if it finishes. Both bounds are the same
# question - will this knock the wall out before the game ends - asked of the
# turn count and of the deck.
MAX_ROUTE_TURNS = 8
# Promoting a body the wall can knock out hands over a prize, so that is only
# taken once the alternative is established to be nothing: the second
# consecutive own turn on which a Shadow Bullet is provably worth zero.
SACRIFICE_AFTER_DEAD_TURNS = 1


class WallBreakGuard:
    """Send the turn to a body the wall cannot ignore."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._turn = -2
        self._turn_dead = False
        self._dead_turns = 0
        self._promote_serial: int | None = None
        self.stats: dict[str, int] = {
            "considered": 0,
            "main_decisions": 0,
            # the state this module exists to measure
            "dead_swing_turns": 0,
            "dead_swing_decisions": 0,
            "boss_prize_deferred": 0,
            "no_breaker_in_play": 0,
            "breaker_too_slow": 0,
            "breaker_would_be_sacrificed": 0,
            # BREAK
            "attack_forced": 0,
            "fuel_attach_forced": 0,
            "escape_attach_forced": 0,
            "retreat_forced": 0,
            "attachment_redirected": 0,
            "dead_shadow_replaced": 0,
            "end_replaced": 0,
            "route_compatible_kept": 0,
            "promote_forced": 0,
            "promote_already_right": 0,
            # PRESERVE
            "last_breaker_evolve_refused": 0,
            "last_breaker_evolve_kept": 0,
            "last_breaker_only_body": 0,
            "last_breaker_fastest_route": 0,
            "last_breaker_preserve_shadow": 0,
            "last_breaker_preserve_end": 0,
            "punk_breaker_considered": 0,
            "punk_breaker_forced": 0,
            "punk_breaker_kept": 0,
            "new_game_detected": 0,
            "errors": 0,
        }

    # ----- turn bookkeeping -------------------------------------------------
    def note(self, observation: dict[str, Any]) -> None:
        """Roll the per-turn state, and the consecutive dead-turn counter."""
        try:
            turn = int((observation.get("current") or {}).get("turn", -1))
            if turn == self._turn:
                return
            if turn < self._turn:
                self.stats["new_game_detected"] += 1
                self._dead_turns = 0
            elif self._turn_dead:
                self._dead_turns += 1
            else:
                self._dead_turns = 0
            self._turn = turn
            self._turn_dead = False
            self._promote_serial = None
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
            if me is None or opponent is None:
                return index
            if context in PROMOTION_CONTEXTS:
                return self._promote(current, select, options, index)
            if context == PUNK_TARGET_CONTEXT:
                return self._punk_allocate(
                    current, select, options, index, me, opponent
                )
            if context != MAIN_CONTEXT:
                return index
            return self._main(
                current, select, options, index, me, opponent, rule_choice
            )
        except Exception:  # noqa: BLE001 - any surprise means v15's answer
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
        their_active = (mf._cards(opponent, "active") or [{}])[0]
        if not self._dead_swing(current, me, opponent, their_active):
            return index
        self.stats["dead_swing_decisions"] += 1
        if not self._turn_dead:
            self._turn_dead = True
            self.stats["dead_swing_turns"] += 1

        stadium_id = mf._stadium_id(current)
        target = self._breaker(me, opponent, their_active, stadium_id)
        step, kind = (
            self._route_step(current, select, options, me, target)
            if target is not None else (None, "")
        )

        chosen = options[index]
        chosen_type = mf._int(chosen.get("type"))

        # ---- PRESERVE -------------------------------------------------------
        if self._spends_last_breaker(current, select, chosen, me):
            replacement, replacement_kind = step, kind
            if replacement is None:
                # No route step to spend the turn on, so fall back to whatever
                # v8 would do instead. That is not a route step, so it must not
                # be counted as one or set the promotion.
                replacement, replacement_kind = self._rule_alternative(
                    options, rule_choice, index
                ), ""
                if (
                    replacement is not None
                    and self._spends_last_breaker(
                        current, select, options[replacement], me
                    )
                ):
                    replacement = None
            if replacement is None:
                # v16 stopped here when the current breaker itself failed the
                # eight-turn route test.  That is correct for BREAK but not
                # for PRESERVE: an Impidimp which is too slow today is still
                # the only body that can become a Morgrem tomorrow.  A dead
                # Shadow and END both close the turn just as the evolution
                # would, while retaining that future route.
                replacement, replacement_kind = self._preserve_closer(
                    options, index
                )
            if replacement is not None and replacement != index:
                self.stats["last_breaker_evolve_refused"] += 1
                return self._commit(
                    replacement, replacement_kind, options, me, target
                )
            self.stats["last_breaker_evolve_kept"] += 1
            return index

        if step is None:
            return index
        if step == index:
            self.stats["route_compatible_kept"] += 1
            return index

        # ---- BREAK ----------------------------------------------------------
        # Only a closing action can lose the route: MAIN repeats until an
        # attack or an END, so any other play can still be followed by the
        # route's next step in the same turn. The one exception is the manual
        # attachment, which is the resource the route needs and which a
        # non-closing action can spend.
        if chosen_type == OPTION_ATTACK:
            if self._real_damage(chosen, me, their_active, stadium_id):
                self.stats["route_compatible_kept"] += 1
                return index  # already a swing the wall cannot ignore
            self.stats["dead_shadow_replaced"] += 1
        elif chosen_type == OPTION_END:
            self.stats["end_replaced"] += 1
        elif chosen_type == OPTION_ATTACH and kind in (
            "fuel_attach_forced", "escape_attach_forced"
        ):
            self.stats["attachment_redirected"] += 1
        else:
            return index

        return self._commit(step, kind, options, me, target)

    # ----- Punk Up can pay the wall route without a manual attachment ------
    def _punk_allocate(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        index: int,
        me: dict[str, Any],
        opponent: dict[str, Any],
    ) -> int:
        effect = select.get("effect")
        if not isinstance(effect, dict):
            return index
        if mf._int(effect.get("id")) != GRIMMSNARL_EX_ID:
            return index

        # The planner's first invariant is that the Grimmsnarl which triggered
        # Punk Up ends ready to attack.  Never steal either of its first two.
        trigger_serial = effect.get("serial")
        trigger = next(
            (
                body for body in mf._in_play(me)
                if body.get("serial") == trigger_serial
            ),
            None,
        )
        if (
            trigger is None
            or mf._dark_energy_count(trigger) < DARK_ATTACKS[
                GRIMMSNARL_EX_ID
            ][1]
        ):
            return index

        their_active = (mf._cards(opponent, "active") or [{}])[0]
        if not self._dead_swing(current, me, opponent, their_active):
            return index
        target = self._breaker(
            me, opponent, their_active, mf._stadium_id(current)
        )
        if (
            target is None
            or int(target.get("id", -1)) != MORGREM_ID
            or mf._dark_energy_count(target) >= DARK_ATTACKS[MORGREM_ID][1]
        ):
            return index

        target_serial = target.get("serial")
        wanted: int | None = None
        for slot, option in enumerate(options):
            body, is_self, _area = mf.resolve_option(
                current, select, option
            )
            if (
                body is not None
                and is_self
                and body.get("serial") == target_serial
            ):
                wanted = slot
                break
        if wanted is None:
            return index
        self.stats["punk_breaker_considered"] += 1
        if wanted == index:
            self.stats["punk_breaker_kept"] += 1
            return index
        self.stats["punk_breaker_forced"] += 1
        return wanted

    def _commit(
        self,
        step: int,
        kind: str,
        options: list[dict[str, Any]],
        me: dict[str, Any],
        target: dict[str, Any] | None,
    ) -> int:
        if kind:
            self.stats[kind] += 1
        if kind == "retreat_forced" and target is not None:
            serial = target.get("serial")
            self._promote_serial = (
                int(serial) if serial is not None else None
            )
        return step

    # ----- is a Shadow Bullet provably worth nothing right now? -------------
    def _dead_swing(
        self,
        current: dict[str, Any],
        me: dict[str, Any],
        opponent: dict[str, Any],
        their_active: dict[str, Any],
    ) -> bool:
        if int(their_active.get("id", -1)) < 0:
            return False
        stadium_id = mf._stadium_id(current)
        if mf.shadow_damage_to(their_active, stadium_id) > 0.0:
            return False
        routes = mf.turn_routes(current, opponent)
        if routes["no_boss_prizes"] > 0:
            return False  # the Bench-30 is taking a prize; the swing is real
        if routes["best_boss_prizes"] > 0 and self._boss_playable(current, me):
            # Gusting is the other way out of a wall and the ranker already
            # finds it: one missed Boss prize in 446 swings. Stand down.
            self.stats["boss_prize_deferred"] += 1
            return False
        return True

    @staticmethod
    def _boss_playable(current: dict[str, Any], me: dict[str, Any]) -> bool:
        if current.get("supporterPlayed"):
            return False
        return any(
            int(card.get("id", -1)) == BOSS_ID
            for card in mf._cards(me, "hand")
        )

    # ----- damage this deck can actually deal -------------------------------
    @staticmethod
    def _blocked(attacker_id: int, defender_id: int, stadium_id: int) -> bool:
        if defender_id in EX_BLOCKER_IDS and attacker_id in OUR_EX_IDS:
            return True
        if (
            defender_id in ABILITY_BLOCKER_IDS
            and attacker_id in OUR_ABILITY_IDS
        ):
            return True
        if (
            stadium_id == mf.NEUTRALIZATION_ZONE_ID
            and attacker_id in OUR_EX_IDS
            and not mf._is_rule_box(defender_id)
        ):
            return True
        return False

    @classmethod
    def _damage(
        cls,
        attacker: dict[str, Any],
        defender: dict[str, Any],
        stadium_id: int,
    ) -> tuple[float, int, int]:
        """(damage, Darkness needed, attack id) for one attacker."""
        attacker_id = int(attacker.get("id", -1))
        spec = DARK_ATTACKS.get(attacker_id)
        if spec is None:
            return 0.0, 0, -1
        attack_id, need, damage = spec
        defender_id = int(defender.get("id", -1))
        if cls._blocked(attacker_id, defender_id, stadium_id):
            return 0.0, need, attack_id
        attacker_type = mf.POKEMON_TYPE_IDS.get(attacker_id, -1)
        if mf.POKEMON_WEAKNESS_IDS.get(defender_id) == attacker_type:
            damage *= 2.0
        elif mf.POKEMON_RESISTANCE_IDS.get(defender_id) == attacker_type:
            damage = max(0.0, damage - 30.0)
        return damage, need, attack_id

    def _breaker(
        self,
        me: dict[str, Any],
        opponent: dict[str, Any],
        their_active: dict[str, Any],
        stadium_id: int,
    ) -> dict[str, Any] | None:
        """The body that knocks this wall out soonest, or None."""
        hp = float(their_active.get("hp", 0) or 0)
        deck_turns = int(me.get("deckCount", 0) or 0)
        threats = mf._in_play(opponent)
        best: dict[str, Any] | None = None
        best_key: tuple | None = None
        too_slow = sacrificial = False
        for area in ("active", "bench"):
            for slot, body in enumerate(mf._cards(me, area)):
                if int(body.get("id", -1)) not in BREAKER_IDS:
                    continue
                damage, need, _attack = self._damage(
                    body, their_active, stadium_id
                )
                if damage <= 0.0 or hp <= 0.0:
                    continue
                fuel = max(0, need - mf._dark_energy_count(body))
                eta = fuel + int(math.ceil(hp / damage))
                if eta > MAX_ROUTE_TURNS or eta >= max(1, deck_turns):
                    too_slow = True
                    continue
                if (
                    mf.incoming_damage(threats, body)
                    >= float(body.get("hp", 0) or 0)
                    and self._dead_turns < SACRIFICE_AFTER_DEAD_TURNS
                ):
                    # It dies to their next swing and the wall has not yet
                    # proved permanent: do not hand over a prize on spec.
                    sacrificial = True
                    continue
                key = (eta, -damage, -float(body.get("hp", 0) or 0))
                if best_key is None or key < best_key:
                    best, best_key = dict(body), key
                    best["_area"] = area
                    best["_slot"] = slot
        if best is None:
            if too_slow:
                self.stats["breaker_too_slow"] += 1
            elif sacrificial:
                self.stats["breaker_would_be_sacrificed"] += 1
            else:
                self.stats["no_breaker_in_play"] += 1
        return best

    # ----- the next step of the route ---------------------------------------
    def _route_step(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        me: dict[str, Any],
        target: dict[str, Any] | None,
    ) -> tuple[int | None, str]:
        if target is None:
            return None, ""
        attack_id = DARK_ATTACKS[int(target["id"])][0]
        need = DARK_ATTACKS[int(target["id"])][1]
        have = mf._dark_energy_count(target)
        attach_used = bool(current.get("energyAttached"))

        if target["_area"] == "active":
            if have >= need:
                slot = self._attack_option(options, attack_id)
                if slot is None:
                    return None, ""
                return slot, "attack_forced"
            if not attach_used:
                slot = self._dark_attach_option(
                    current, select, options, mf.AREA_ACTIVE
                )
                if slot is not None:
                    return slot, "fuel_attach_forced"
            return None, ""

        # Benched: fuel it first, then retreat into it.
        if have < need:
            if not attach_used:
                slot = self._dark_attach_option(
                    current, select, options,
                    mf.AREA_BENCH, int(target["_slot"]),
                )
                if slot is not None:
                    return slot, "fuel_attach_forced"
            return None, ""
        if self._retreat_blocked(current, me):
            return None, ""
        active = (mf._cards(me, "active") or [{}])[0]
        debt = max(
            0, self._retreat_cost(active) - mf._energy_count(active)
        )
        if debt == 0:
            slot = self._retreat_option(options)
            if slot is None:
                return None, ""
            return slot, "retreat_forced"
        if debt == 1 and not attach_used:
            slot = self._dark_attach_option(
                current, select, options, mf.AREA_ACTIVE
            )
            if slot is not None:
                return slot, "escape_attach_forced"
        return None, ""

    # ----- PRESERVE ---------------------------------------------------------
    def _route_eta(
        self,
        body: dict[str, Any],
        their_active: dict[str, Any],
        stadium_id: int,
        deck_left: int,
    ) -> float:
        """Own turns for this breaker to knock the wall out, or ``inf``.

        The same arithmetic ``_breaker`` uses to rank a route, minus the
        sacrifice test: PRESERVE only declines an evolution, so keeping a body
        that might trade badly still costs nothing. One turn is charged per
        missing Darkness because a single manual attachment is available per
        turn and Punk Up cannot fire without consuming a breaker.
        """
        damage, need, _attack = self._damage(body, their_active, stadium_id)
        hp = float(their_active.get("hp", 0) or 0)
        if damage <= 0.0 or hp <= 0.0:
            return math.inf
        remaining = max(hp - float(body.get("damage") or 0.0), 1.0)
        fuel = max(0, need - mf._dark_energy_count(body))
        eta = fuel + int(math.ceil(remaining / damage))
        if eta > MAX_ROUTE_TURNS or eta >= max(1, deck_left):
            return math.inf
        return float(eta)

    def _spends_last_breaker(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
        me: dict[str, Any],
    ) -> bool:
        """This evolution consumes our best route to the wall.

        v16-v20 tested ``len(breakers) != 1``, a raw count of Morgrem and
        Impidimp bodies. A second Morgrem with no Darkness therefore disarmed
        the guard even though it could not reach the wall for two more turns,
        which is exactly episode 92168220 turn 13: Crustle Active, a fuelled
        Morgrem evolved away, the survivor unable to attack, nine dead Shadow
        Bullets and a deck-out for zero prizes. Over the 529 stored ladder
        games the count test refuses 17 evolutions and this one refuses 31,
        the extra 14 spread over 13 episodes of which 10 are losses.

        The count test is kept as a disjunct rather than replaced, so nothing
        v17 preserved can be lost: an Impidimp too slow to be a route today is
        still the only body that can become a Morgrem tomorrow.

        Gated on already owning a fuelled Grimmsnarl ex, so declining never
        costs the attacker itself - only the Punk Up that comes with the
        evolution, on a board where the extra Energy has nothing to attack.
        """
        if mf._int(option.get("type")) != OPTION_EVOLVE:
            return False
        card = mf.candidate_card(current, option, select)
        if int((card or {}).get("id", -1)) != GRIMMSNARL_EX_ID:
            return False
        in_play = mf._in_play(me)
        breakers = [
            body for body in in_play
            if int(body.get("id", -1)) in BREAKER_IDS
        ]
        if not breakers:
            return False
        ready = any(
            int(body.get("id", -1)) == GRIMMSNARL_EX_ID
            and mf._dark_energy_count(body) >= 2
            for body in in_play
        )
        if not ready:
            return False
        # The evolution has to be the one landing on a breaker.
        area = mf._int(option.get("inPlayArea"))
        slot = mf._int(option.get("inPlayIndex"))
        pool = mf._cards(
            me, "active" if area == mf.AREA_ACTIVE else "bench"
        )
        if not 0 <= slot < len(pool):
            return False
        target = pool[slot]
        serial = target.get("serial")
        if not any(serial == body.get("serial") for body in breakers):
            return False

        if len(breakers) == 1:
            self.stats["last_breaker_only_body"] += 1
            return True

        players = current.get("players") or []
        your = int(current.get("yourIndex", 0))
        opponent = players[1 - your] if len(players) > 1 else {}
        their_active = (mf._cards(opponent, "active") or [{}])[0]
        stadium_id = mf._stadium_id(current)
        deck_left = int(me.get("deckCount", 0) or 0)
        own = self._route_eta(target, their_active, stadium_id, deck_left)
        if own == math.inf:
            return False
        others = [
            self._route_eta(body, their_active, stadium_id, deck_left)
            for body in breakers
            if body.get("serial") != serial
        ]
        if own < min(others, default=math.inf):
            self.stats["last_breaker_fastest_route"] += 1
            return True
        return False

    @staticmethod
    def _rule_alternative(
        options: list[dict[str, Any]], rule_choice: Any, index: int
    ) -> int | None:
        """v8's own answer, when it is a different action."""
        if not (
            isinstance(rule_choice, list)
            and len(rule_choice) == 1
            and isinstance(rule_choice[0], int)
        ):
            return None
        slot = int(rule_choice[0])
        if not 0 <= slot < len(options) or slot == index:
            return None
        return slot

    @staticmethod
    def _preserve_closer(
        options: list[dict[str, Any]], index: int
    ) -> tuple[int | None, str]:
        """A cost-free way to end the turn without consuming the breaker.

        This is reached only after ``_dead_swing`` proved Shadow deals zero to
        the Active and takes no Bench prize, and ``_spends_last_breaker``
        proved another fuelled Grimmsnarl ex already exists.  Prefer the free
        Bench-30 over END because it can still help a later route.
        """
        for slot, option in enumerate(options):
            if slot == index or mf._int(option.get("type")) != OPTION_ATTACK:
                continue
            if mf._int(option.get("attackId")) == SHADOW_BULLET_ID:
                return slot, "last_breaker_preserve_shadow"
        for slot, option in enumerate(options):
            if slot != index and mf._int(option.get("type")) == OPTION_END:
                return slot, "last_breaker_preserve_end"
        return None, ""

    # ----- promotion after our own retreat ----------------------------------
    def _promote(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        index: int,
    ) -> int:
        if self._promote_serial is None:
            return index
        wanted = self._promote_serial
        self._promote_serial = None
        for slot, option in enumerate(options):
            if mf._int(option.get("type")) != OPTION_CARD:
                continue
            card, is_self, _area = mf.resolve_option(current, select, option)
            if card is None or not is_self:
                continue
            if card.get("serial") != wanted:
                continue
            if slot == index:
                self.stats["promote_already_right"] += 1
                return index
            self.stats["promote_forced"] += 1
            return slot
        return index

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
    def _retreat_cost(card: dict[str, Any]) -> int:
        cost = RETREAT_COST.get(int(card.get("id", -1)), 1)
        for tool in (card.get("tools") or []):
            if int((tool or {}).get("id", -1)) == HANDHELD_FAN_ID:
                cost = max(0, cost - HANDHELD_FAN_REDUCTION)
        return cost

    @staticmethod
    def _retreat_blocked(current: dict[str, Any], me: dict[str, Any]) -> bool:
        return bool(
            current.get("retreated")
            or me.get("asleep")
            or me.get("paralyzed")
        )

    def _real_damage(
        self,
        option: dict[str, Any],
        me: dict[str, Any],
        their_active: dict[str, Any],
        stadium_id: int,
    ) -> bool:
        """The chosen attack already puts damage on their Active."""
        active = (mf._cards(me, "active") or [{}])[0]
        damage, _need, attack_id = self._damage(
            active, their_active, stadium_id
        )
        return damage > 0.0 and mf._int(option.get("attackId")) == attack_id

    # ----- option lookups ---------------------------------------------------
    @staticmethod
    def _attack_option(
        options: list[dict[str, Any]], attack_id: int
    ) -> int | None:
        for slot, option in enumerate(options):
            if (
                mf._int(option.get("type")) == OPTION_ATTACK
                and mf._int(option.get("attackId")) == attack_id
            ):
                return slot
        return None

    @staticmethod
    def _retreat_option(options: list[dict[str, Any]]) -> int | None:
        for slot, option in enumerate(options):
            if mf._int(option.get("type")) == OPTION_RETREAT:
                return slot
        return None

    @staticmethod
    def _dark_attach_option(
        current: dict[str, Any],
        select: dict[str, Any],
        options: list[dict[str, Any]],
        area: int,
        in_play_index: int | None = None,
    ) -> int | None:
        """A manual Darkness attachment onto one specific body.

        ``inPlayIndex`` is only checked when a slot is named, because the
        Active spot is a single slot and some option shapes leave the index
        off it entirely.
        """
        for slot, option in enumerate(options):
            if mf._int(option.get("type")) != OPTION_ATTACH:
                continue
            if mf._int(option.get("inPlayArea")) != area:
                continue
            if (
                in_play_index is not None
                and mf._int(option.get("inPlayIndex")) != in_play_index
            ):
                continue
            card = mf.candidate_card(current, option, select)
            if int((card or {}).get("id", -1)) == DARK_ENERGY_ID:
                return slot
        return None

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)
