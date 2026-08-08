"""One-ply route planner over the imitation ranker's answer.

Standard library only; it ships inside the Kaggle runtime beside the ranker.

Why a shell at all, when the Alakazam line measured one eating 5.16 points of
what its agent actually played: because two of the v2 ladder defects are not
preference errors the ranker could learn out of a better feature, they are
*arithmetic* the ranker is being asked to infer from separate columns.

* Shadow Bullet is 180 to the Active and 30 to one Bench in the same swing, so
  Boss's Orders, the 180 and the 30 are one plan. v2 gusted a body the Bench-30
  killed for free 4 times out of the 7 it could (the 1220-rated pilot: 8.6%),
  and in episode 89678716 that cost it the game.
* Adrena-Brain moves up to 3 counters off one of our bodies. Whether a heal
  changes anything depends on the opponent's next hit and on how many Munkidori
  are still live this turn - a 140 HP Grimmsnarl facing 180 needs two
  moves, and one is worth nothing. That is a threshold over a turn, not
  over a single candidate.

So the rules here are deliberately not preferences. Each fires only when
the observation *proves* the ranker's pick is dominated on prizes taken
this turn or on whether a body of ours survives the next attack, each
keeps the ranker's ordering as the tie-break, and every firing is counted
so the deployed override rate is a measured number, not an assumption.
"""

from __future__ import annotations

from typing import Any

import ml_features as mf

AREA_BENCH = 5
BOSS_TARGET_CONTEXTS = (mf.CTX_SWITCH, mf.CTX_TO_ACTIVE)
# Shadow Bullet costs {D}{D}; past four a body is holding energy nothing uses.
PUNK_ATTACK_ENERGY = 2
PUNK_MAX_STACK = 4


class Planner:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._turn: int | None = None
        self._munkidori_uses = 0
        self.stats: dict[str, int] = {
            "considered": 0,
            "boss_route_considered": 0,
            "boss_route_overrides": 0,
            "heal_considered": 0,
            "heal_overrides": 0,
            "froslass_considered": 0,
            "froslass_overrides": 0,
            "wall_unlock_considered": 0,
            "wall_unlock_overrides": 0,
            "punk_alloc_considered": 0,
            "punk_alloc_trigger_overrides": 0,
            "punk_alloc_stack_overrides": 0,
            "errors": 0,
        }

    # ----- turn bookkeeping -------------------------------------------------
    def note(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        chosen: int | list[int],
    ) -> None:
        """Record the action actually taken, for the per-turn heal budget.

        Adrena-Brain is once per turn *per Munkidori*, so how many heals are
        still available depends on how many have already fired this turn. The
        ranker's own intra-turn history cannot answer that: it counts
        offers and passes, not activations.
        """
        try:
            if isinstance(chosen, list):
                if len(chosen) != 1:
                    # Multi/zero-pick selects cannot activate a single
                    # Munkidori source, but replay harnesses still route them
                    # through this bookkeeping hook.
                    return
                chosen = chosen[0]
            current = observation.get("current") or {}
            turn = int(current.get("turn", -1))
            if turn != self._turn:
                self._turn = turn
                self._munkidori_uses = 0
            if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
                return
            options = list(select.get("option") or [])
            if not 0 <= chosen < len(options):
                return
            option = options[chosen]
            if mf.action_type(current, option, select) != "ability":
                return
            card = mf.candidate_card(current, option, select) or {}
            if int(card.get("id", -1)) == mf.MUNKIDORI_ID:
                self._munkidori_uses += 1
        except Exception:
            self.stats["errors"] += 1

    def heals_available(self, current: dict[str, Any]) -> int:
        """Adrena-Brain moves still landable this turn, counting this one.

        The MAIN activation is committed before the source select arrives, so
        the in-flight move is already inside ``_munkidori_uses``.
        """
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", 0))
        me = players[your] if your < len(players) else {}
        ready = sum(
            int(
                int(card.get("id", -1)) == mf.MUNKIDORI_ID
                and mf._dark_energy_count(card) > 0
            )
            for card in mf._in_play(me)
        )
        return max(1, ready - max(0, self._munkidori_uses - 1))

    # ----- the rules --------------------------------------------------------
    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None = None,
    ) -> int:
        """The ranker's index, or a dominating one. Never raises."""
        try:
            self.stats["considered"] += 1
            context = int(select.get("context", -1))
            if context in BOSS_TARGET_CONTEXTS:
                return self._boss_route(observation, select, index, scores)
            if context == mf.CTX_REMOVE_DAMAGE_COUNTER:
                return self._heal_source(observation, select, index, scores)
            if context == mf.CTX_ATTACH_FROM:
                return self._punk_allocation(
                    observation, select, index, scores
                )
            if context == mf.MAIN_CONTEXT:
                moved = self._wall_unlock(observation, select, index, scores)
                if moved != index:
                    return moved
                return self._froslass_guard(observation, select, index, scores)
            return index
        except Exception:
            self.stats["errors"] += 1
            return index

    def _boss_route(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None,
    ) -> int:
        """Gusting a body the free Bench-30 already kills, for fewer prizes.

        Fires only on that exact shape: the ranker's target dies to the
        Bench-30 on its own, and some other target takes strictly more
        prizes this turn with the same one swing. Any other disagreement
        about which body to gust is left to the ranker - denying an
        attacker or breaking a wall are reasons a prize count cannot see.
        """
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", 0))
        if len(players) < 2:
            return index
        me, opponent = players[your], players[1 - your]
        options = list(select.get("option") or [])
        if not 0 <= index < len(options):
            return index
        # Every option has to be one of their benched bodies, or this is not a
        # gust select at all (TO_ACTIVE also promotes our own body after a KO).
        for option in options:
            owner = option.get("playerIndex")
            if not isinstance(owner, int) or owner == your:
                return index
            if mf._int(option.get("area")) != AREA_BENCH:
                return index

        active = (mf._cards(me, "active") or [{}])[0]
        if int(active.get("id", -1)) != mf.GRIMMSNARL_EX_ID:
            return index
        if mf._dark_energy_count(active) < mf.SHADOW_BULLET_COST:
            return index

        routes = mf.turn_routes(current, opponent)
        by_index = {entry["index"]: entry for entry in routes["per_target"]}
        chosen = by_index.get(mf._int(options[index].get("index")))
        if chosen is None or not chosen["dies_to_snipe_alone"]:
            return index
        self.stats["boss_route_considered"] += 1

        best = chosen["total"]
        candidates: list[int] = []
        for slot, option in enumerate(options):
            entry = by_index.get(mf._int(option.get("index")))
            if entry is None:
                continue
            if entry["total"] > best:
                best = entry["total"]
                candidates = [slot]
            elif entry["total"] == best and slot != index:
                candidates.append(slot)
        if best <= chosen["total"] or not candidates:
            return index
        self.stats["boss_route_overrides"] += 1
        return _best_by_score(candidates, scores)

    def _heal_source(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None,
    ) -> int:
        """Adrena-Brain's source, when another choice saves a body and
        this one does not.

        The override needs all three of:

        * some offered body of ours is knocked out by the opponent's best hit
          next turn but survives if the heals still available this turn land on
          it - that is the threshold v2 could not see, and it is why 42.5% of
          the time it healed something else;
        * it is worth at least as many prizes as the body the ranker picked;
        * it carries at least as many movable counters, so the override never
          deals less damage than the ranker's answer would have.
        """
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", 0))
        if len(players) < 2:
            return index
        opponent = players[1 - your]
        options = list(select.get("option") or [])
        if not 0 <= index < len(options):
            return index
        threats = mf._in_play(opponent)
        budget = self.heals_available(current)

        resolved: dict[int, dict[str, Any]] = {}
        for slot, option in enumerate(options):
            card, owner_is_self, area = mf.resolve_option(
                current, select, option
            )
            if not card or not owner_is_self:
                continue
            counters = mf.movable_counters(card)
            if counters <= 0:
                continue
            hp = float(card.get("hp", 0))
            # Only the Active is in front of the next attack. A benched body is
            # reachable by spread and snipes this does not model, so calling it
            # "savable" would let the planner move counters for a threat that
            # was never coming.
            threat = (
                mf.incoming_damage(threats, card)
                if area == mf.AREA_ACTIVE else 0.0
            )
            needed = mf.heals_needed(hp, threat)
            resolved[slot] = {
                "counters": counters,
                "needed": needed,
                "savable": 0 < needed <= budget,
                "prizes": mf.prize_value(int(card.get("id", -1))),
            }
        chosen = resolved.get(index)
        if chosen is None:
            return index
        self.stats["heal_considered"] += 1
        if chosen["savable"]:
            return index  # the ranker is already healing a body this saves

        candidates = [
            slot for slot, item in resolved.items()
            if item["savable"]
            and item["prizes"] >= chosen["prizes"]
            and item["counters"] >= chosen["counters"]
        ]
        if not candidates:
            return index
        # Among the bodies worth saving, the most prizes on the line
        # first, then the most counters moved, then the ranker's order.
        top = max(
            (resolved[slot]["prizes"], resolved[slot]["counters"])
            for slot in candidates
        )
        candidates = [
            slot for slot in candidates
            if (resolved[slot]["prizes"], resolved[slot]["counters"]) == top
        ]
        self.stats["heal_overrides"] += 1
        return _best_by_score(candidates, scores)

    def _wall_unlock(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None,
    ) -> int:
        """Swinging into a damage-immune Active while Boss would expose a
        body we can actually damage.

        This is the one shape in the Mega Kangaskhan ex / Crustle matchup that
        is provable rather than preferred. Both closing moves - the swing and
        the pass - leave the board unchanged: the swing deals 0 to a
        damage-immune Active and takes no prize off the Bench. Playing Boss
        instead keeps the turn alive and puts a body Shadow Bullet damages into
        the Active spot, so the same swing is worth at least 180. The supporter
        slot is not a cost here: the turn was about to end with it unused.

        The narrowness matters. It does *not* veto the swing - the top-50
        pilots take a worthless swing on 88.6% of turns where it is the last
        thing available (the elite pair 91.1%), because by then it is free. It
        only refuses to *close* the turn while a gust that converts the wall is
        still in hand: 216 teacher turns have this shape and they play Boss on
        63.0% of them, the pinned pilot on 83.3%, and v3 on 1 of 5.
        """
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", 0))
        if len(players) < 2:
            return index
        me, opponent = players[your], players[1 - your]
        options = list(select.get("option") or [])
        if not 0 <= index < len(options):
            return index
        # Both of these close the turn. Measuring v3 on its own 65 games found
        # the attack alone is the wrong trigger: on all 5 turns that had this
        # shape it never picked the swing *at* a decision where the gust was
        # also offered, so a rule keyed on the attack fires never - which is
        # how v3's Boss rule ended up at 0 firings too.
        action = mf.action_type(current, options[index], select)
        if action == "attack":
            if mf._int(options[index].get("attackId")) != mf.SHADOW_BULLET_ID:
                return index
        elif action != "end":
            return index

        stadium_id = mf._stadium_id(current)
        opp_active = (mf._cards(opponent, "active") or [{}])[0]
        if int(opp_active.get("id", -1)) < 0:
            return index
        if mf.shadow_damage_to(opp_active, stadium_id) > 0.0:
            return index  # the swing damages the Active; nothing to prove

        # The swing must take no prize at all, or "more damage" is a trade the
        # planner has no standing to make: a Bench-30 kill is a prize in hand.
        routes = mf.turn_routes(current, opponent)
        if routes["no_boss_prizes"] > 0:
            return index

        bench = mf._cards(opponent, "bench")
        if not any(mf.shadow_damage_to(c, stadium_id) > 0.0 for c in bench):
            return index

        gusts = [
            slot for slot, option in enumerate(options)
            if mf.action_type(current, option, select) == "boss"
        ]
        if not gusts:
            return index
        self.stats["wall_unlock_considered"] += 1

        # Retreating for an attacker costs energy this deck cannot spare, so
        # the gust only converts if the attacker is Active and fuelled -
        # which the legal Shadow Bullet we are overriding proves.
        active = (mf._cards(me, "active") or [{}])[0]
        if int(active.get("id", -1)) != mf.GRIMMSNARL_EX_ID:
            return index
        if mf._dark_energy_count(active) < mf.SHADOW_BULLET_COST:
            return index

        self.stats["wall_unlock_overrides"] += 1
        return _best_by_score(gusts, scores)

    def _punk_allocation(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None,
    ) -> int:
        """Where one Punk Up energy lands, once the search stops taking five.

        v5 trims the Punk Up deck search from "every card the select allows" to
        what the board wants, and the promise that budget is computed against
        is that the body which just evolved ends the activation able to attack.
        The allocation itself is the ranker's - and it is good at it: over 330
        stored attaches it never once fed a body already at two while a
        hungrier one was offered, better than any teacher band. These guards
        exist so a tighter budget cannot be spent wrong, not because the ranker
        spends it wrong today.

        * the triggering Grimmsnarl ex is still short of Shadow Bullet and
          is offered. The teachers put the energy on it on 96.1% (>= 1100),
          98.8% and 99.5% of such offers, and v4 on 115 of 115: this makes the
          invariant the budget assumes hold even when the budget is exactly the
          deficit.
        * nothing goes onto a body already holding four Darkness while a body
          holding fewer is offered. 0.16% of elite attaches, 0.91% of ours.
        """
        current = observation.get("current") or {}
        effect = select.get("effect")
        if not isinstance(effect, dict):
            return index
        if mf._int(effect.get("id")) != mf.GRIMMSNARL_EX_ID:
            return index
        options = list(select.get("option") or [])
        if not 0 <= index < len(options):
            return index

        resolved: dict[int, dict[str, Any]] = {}
        for slot, option in enumerate(options):
            card, owner_is_self, _ = mf.resolve_option(current, select, option)
            if not card or not owner_is_self:
                return index  # not the Punk Up target select we know
            resolved[slot] = {
                "serial": mf._int(card.get("serial"), -2),
                "energy": mf._dark_energy_count(card),
            }
        if len(resolved) < 2:
            return index
        self.stats["punk_alloc_considered"] += 1

        trigger_serial = mf._int(effect.get("serial"), -3)
        chosen = resolved[index]
        trigger = [
            slot for slot, item in resolved.items()
            if item["serial"] == trigger_serial
            and item["energy"] < PUNK_ATTACK_ENERGY
        ]
        if trigger and index not in trigger:
            self.stats["punk_alloc_trigger_overrides"] += 1
            return _best_by_score(trigger, scores)

        if chosen["energy"] >= PUNK_MAX_STACK:
            lighter = [
                slot for slot, item in resolved.items()
                if item["energy"] < chosen["energy"]
            ]
            if lighter:
                self.stats["punk_alloc_stack_overrides"] += 1
                return _best_by_score(lighter, scores)
        return index

    def _froslass_guard(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        scores: dict[int, float] | None,
    ) -> int:
        """Refuse only the Froslass evolve that hands over a prize at once.

        v2 evolves into Froslass in 81.8% of the boards where Freezing
        Shroud is net negative for us against 21.4% for the 1220-rated
        pilot, but "net negative" is a judgement, and vetoing a judgement
        is how a shell starts costing more than it saves. The dominated
        case is narrower: the next checkup knocks out a body of ours and
        none of theirs, so the evolve gives up a prize for nothing.
        Everything else stays with the ranker.
        """
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", 0))
        if len(players) < 2:
            return index
        me, opponent = players[your], players[1 - your]
        options = list(select.get("option") or [])
        if not 0 <= index < len(options):
            return index
        option = options[index]
        if mf.action_type(current, option, select) != "evolve":
            return index
        card = mf.candidate_card(current, option, select) or {}
        if int(card.get("id", -1)) != mf.FROSLASS_ID:
            return index
        self.stats["froslass_considered"] += 1

        ours = mf.shroud_targets(mf._in_play(me))
        theirs = mf.shroud_targets(mf._in_play(opponent))
        our_deaths = sum(int(0 < float(c.get("hp", 0)) <= 10.0) for c in ours)
        their_deaths = sum(
            int(0 < float(c.get("hp", 0)) <= 10.0) for c in theirs
        )
        if not our_deaths or their_deaths or len(theirs) > len(ours):
            return index
        alternatives = [slot for slot in range(len(options)) if slot != index]
        if not alternatives:
            return index
        self.stats["froslass_overrides"] += 1
        return _best_by_score(alternatives, scores)

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)


def _best_by_score(
    candidates: list[int],
    scores: dict[int, float] | None,
) -> int:
    """Keep the ranker's preference inside the set the planner allows."""
    if not scores:
        return candidates[0]
    scored = [slot for slot in candidates if slot in scores]
    if not scored:
        return candidates[0]
    return max(scored, key=lambda slot: scores[slot])
