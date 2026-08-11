"""A residual over v8, on one decision class and nothing else.

v8 is the policy. This layer returns v8's own index on every decision in the
game except one narrow shape, and even there only when a panel of pilots that
do not share v8's pin agree on the same replacement.

Why this shape and not another
------------------------------

Fifty rated ladder games of v8 (submission 55317804) were re-walked decision by
decision with five independent advisors scoring the identical boards. The
result that decided this version is a negative one: a k-of-n consensus among
those advisors disagrees with v8 on 3.4% of decisions and that rate is *flat*
across the outcome - 3.41% in games v8 lost against 3.67% in games it won,
Fisher p = 0.68 - so a general consensus override is a coin flip with respect
to winning, which is the thing the Alakazam line already paid 5.16 points of
agreement to learn.

One class survives that filter, and it is the class v6 measured, pre-registered
as "the v7 class", and then held back so that one ladder run measured one
change. It is the Petrel search:

* **the dead Unfair Stamp.** Unfair Stamp is playable only if the opponent took
  a Prize card during their last turn. Taking it out of a Petrel search on a
  turn when they did not is a card that does nothing this turn. Over the 3,642
  same-deck games in the archive, the rate at which a pilot takes a dead Stamp
  runs *against* pilot rating at Spearman rho = -0.607, p = 0.0036 over 21
  pilots, and the live case does not (rho = -0.607, p = 0.148 on n = 7): the
  refusal is specific to the dead board, which is what makes it a policy rather
  than a dislike of the card. v8's own pin is 0.708, third-highest of 21, and
  v8 on its own ladder run is **0.743** against a field 0.570 and 0.34-0.52 for
  the five highest-rated pilots.

On v8's own 35 dead-Stamp offers, every advisor - the four re-pinned pilots and
the separately-trained current-top-four model - takes the dead Stamp *less*
often than v8 does, 0.514 to 0.657 against 0.743. Unanimous direction, measured
on v8's boards rather than on the advisors' own.

What this deliberately does not do
----------------------------------

The same pass found a second significant gradient on the same select - taking
Boss's Orders out of a Petrel search runs *with* rating, rho = +0.581,
p = 0.0058, and v8 takes it on 0 of 64 offers against 0.12-0.16 for the top
pilots. It is not acted on here, because every advisor is also near zero on
v8's boards (0.016-0.109), so there is no consensus to gate on and the only way
to move it would be a preference this layer has no evidence for. It is written
down in ``experiments/grimmsnarl_ml_v10_safe_residual/RESULTS.md`` as the next
candidate, not implemented as a rule.

The safety argument
-------------------

The gate is a *context* gate, not a scoring heuristic, and that is what makes
the rest of the agent provably untouched. Context 7 is "put a card in hand". It
cannot be an energy attachment, an evolve, an attack, a knockout, a prize pick
or a gust target, so the four invariants this version is held to - never lose
a legal attachment, never lose the attachment that turns an attack on, never
reduce a Grimmsnarl ex or Froslass evolve, never overwrite a certain knockout
or the maximum prizes - are satisfied by construction, not by a threshold.
Replaying all 51 stored games confirms it: every one of those rates is
bit-identical to v8's.

Every refusal to fire is counted, so the deployed override rate stays a
measured number.
"""

from __future__ import annotations

from typing import Any

import ml_features as mf

CTX_TO_HAND = mf.CTX_TO_HAND
STAMP_ID = mf.UNFAIR_STAMP_ID
PETREL_ID = mf.PETREL_ID

# ``teacher_team_id`` reaches the trees as a dense 0..20 code, allocated by the
# corpus builder in ascending team-id order over the 21 same-deck pilots. These
# are asserted against the corpus in ``tests/test_v10_residual.py``; if the
# corpus is ever rebuilt with a different team set the codes move and that test
# is what fails instead of the panel silently scoring four other pilots.
PANEL = (
    {"team": 16371703, "code": 0},   # 1220.2, dead Stamp 0.512 over 203 offers
    {"team": 16422241, "code": 9},   # 1172.6, 0.521 over 71
    {"team": 16452116, "code": 12},  # 1101.8, 0.471 over 193
    {"team": 16561259, "code": 20},  # 1126.3, 0.340 over 100
)
# Strict supermajority. The firing count is flat at 3 and 4 of 4 on the stored
# run, so this is not a threshold fitted to the number it produces; 4 of 4 was
# rejected as a rule rather than as a result, because one advisor failing to
# score should not be able to silence the panel.
PANEL_NEED = 3
# Prizes each side starts with, used when a Petrel search happens before any
# earlier turn of ours was observed.
STARTING_PRIZES = 6


class Residual:
    """v8's index, or the panel's, on the one class this version owns."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._opponent_prize_by_turn: dict[int, int] = {}
        self._last_turn = -1
        self.stats: dict[str, int] = {
            "considered": 0,
            "petrel_searches": 0,
            "stamp_offered": 0,
            "stamp_already_in_hand": 0,
            "stamp_live_kept": 0,
            "dead_stamp_chosen": 0,
            "panel_scored": 0,
            "panel_short": 0,
            "panel_agreed_with_v8": 0,
            "panel_alternative_was_stamp": 0,
            "overrides": 0,
            "errors": 0,
            "new_game_detected": 0,
        }

    # ----- turn bookkeeping -------------------------------------------------
    def note(self, observation: dict[str, Any]) -> None:
        """Record the opponent's prize count the first time we see a turn.

        Whether an Unfair Stamp is playable is a fact about *their* last turn,
        and nothing in the observation states it. The first prize count we see
        on each turn is the same quantity the offline measurement used, so the
        deployed gate and the measured gap are the same definition.
        """
        try:
            current = observation.get("current") or {}
            players = current.get("players") or []
            your = int(current.get("yourIndex", 0))
            if len(players) < 2 or your >= len(players):
                return
            turn = int(current.get("turn", -1))
            # Turns never go backwards inside a game, so one that does means a
            # new game in a reused process. Nothing calls ``reset`` between
            # episodes on Kaggle, and a stale entry for turn 1 would make the
            # first Petrel search of the next game read the *previous* game's
            # prize count - the one board where getting live/dead wrong costs
            # a playable Unfair Stamp.
            if turn < self._last_turn:
                self.stats["new_game_detected"] += 1
                self._opponent_prize_by_turn = {}
            self._last_turn = turn
            opponent = players[1 - your]
            self._opponent_prize_by_turn.setdefault(
                turn, len(opponent.get("prize") or [])
            )
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1

    def stamp_is_live(self, current: dict[str, Any]) -> bool:
        """Did the opponent take a prize during their last turn?"""
        players = current.get("players") or []
        your = int(current.get("yourIndex", 0))
        opponent = players[1 - your]
        now = len(opponent.get("prize") or [])
        earlier = [
            turn for turn in self._opponent_prize_by_turn
            if turn < int(current.get("turn", -1))
        ]
        prior = (
            self._opponent_prize_by_turn[max(earlier)]
            if earlier else STARTING_PRIZES
        )
        return now < prior

    # ----- the class --------------------------------------------------------
    def adjust(
        self,
        observation: dict[str, Any],
        select: dict[str, Any],
        index: int,
        ranker: Any,
        scores: dict[int, float] | None = None,
    ) -> int:
        """v8's index, or the panel's replacement for a dead Unfair Stamp."""
        try:
            self.stats["considered"] += 1
            if int(select.get("context", -1)) != CTX_TO_HAND:
                return index
            if _nested_id(select.get("effect")) != PETREL_ID:
                return index
            self.stats["petrel_searches"] += 1

            current = observation.get("current") or {}
            players = current.get("players") or []
            your = int(current.get("yourIndex", 0))
            if len(players) < 2 or your >= len(players):
                return index
            options = list(select.get("option") or [])
            if not 0 <= index < len(options):
                return index
            resolved = [
                int((mf.resolve_option(current, select, option)[0] or {})
                    .get("id", -1))
                for option in options
            ]
            if STAMP_ID not in resolved:
                return index
            self.stats["stamp_offered"] += 1

            me = players[your]
            if any(
                int(card.get("id", -1)) == STAMP_ID
                for card in (me.get("hand") or [])
                if isinstance(card, dict)
            ):
                # A second copy cannot be played anyway, and the measured gap
                # excluded this board, so the gate must too.
                self.stats["stamp_already_in_hand"] += 1
                return index
            if resolved[index] != STAMP_ID:
                return index
            if self.stamp_is_live(current):
                # The live case has no rating gradient behind it (p = 0.148)
                # and the field takes it *more* than the dead one, 0.767
                # against 0.570. Refusing it would be a change with no
                # evidence pointing either way.
                self.stats["stamp_live_kept"] += 1
                return index
            self.stats["dead_stamp_chosen"] += 1

            replacement = self._panel(observation, ranker, index, resolved)
            if replacement is None:
                return index
            self.stats["overrides"] += 1
            return replacement
        except Exception:  # noqa: BLE001
            # An advisor failure, a missing feature or a board shape this was
            # never measured on all land here, and all of them mean v8.
            self.stats["errors"] += 1
            return index

    def _panel(
        self,
        observation: dict[str, Any],
        ranker: Any,
        index: int,
        resolved: list[int],
    ) -> int | None:
        """The slot at least ``PANEL_NEED`` pilots agree on, if not v8's."""
        features, representatives = ranker._rows(observation)
        # Idempotent for the decision the ranker just scored: it resets only
        # when the turn changes, and the turn has not.
        ranker._turn_state(observation, features)
        if len(representatives) < 2:
            return None
        votes: dict[int, int] = {}
        scored = 0
        for member in PANEL:
            try:
                best, _ = ranker._score(
                    features, representatives, member["code"]
                )
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
                continue
            scored += 1
            votes[best] = votes.get(best, 0) + 1
        self.stats["panel_scored"] += scored
        if scored < PANEL_NEED:
            self.stats["panel_short"] += 1
            return None
        slot, count = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))
        if count < PANEL_NEED:
            self.stats["panel_short"] += 1
            return None
        if slot == index:
            self.stats["panel_agreed_with_v8"] += 1
            return None
        if not 0 <= slot < len(resolved) or resolved[slot] == STAMP_ID:
            # An interchangeable second copy of the same card is not a
            # different decision, so it is not a reason to override.
            self.stats["panel_alternative_was_stamp"] += 1
            return None
        return slot

    def snapshot(self) -> dict[str, int]:
        return dict(self.stats)


def _nested_id(value: Any) -> int:
    """The first ``id`` anywhere inside a select's ``effect``."""
    if isinstance(value, dict):
        if "id" in value:
            try:
                return int(value["id"])
            except (TypeError, ValueError):
                return -1
        for item in value.values():
            found = _nested_id(item)
            if found >= 0:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _nested_id(item)
            if found >= 0:
                return found
    return -1
