"""Small deterministic goal selector for the Alakazam agent.

This module deliberately contains no learned parameters.  The runtime policy
enumerates only routes that are legal and deterministic in the current public
state, then this module compares their outcomes lexicographically.  Keeping the
comparison separate from card-level scores prevents an unrelated setup bonus
from changing the intended target halfway through a turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable


# A route may contain several actions that all used to receive the same large
# score.  Choosing one canonical order makes the route stable and keeps Boss
# until every required setup/draw action has finished.
ACTION_ORDER = (
    "evolve_alakazam",
    "evolve_kadabra",
    "rare_candy",
    "dudun",
    "enriching",
    "dawn",
    "hilda",
    "hammer",
    "boss",
)

# Outside a proven same-turn KO, build the recyclable single-prize engine before
# exposing the two-prize alternate attacker.  This is the small deterministic
# replacement for the old model's most useful development choice.
DEVELOPMENT_ORDER = (
    "evolve_alakazam",
    "evolve_kadabra",
    "evolve_dudunsparce",
    "deploy_fezandipiti",
)


@dataclass(frozen=True)
class GoalCandidate:
    """A concrete same-turn KO route and the public outcome it guarantees."""

    target_key: Hashable
    payload: Any
    winning: bool
    prizes: int
    priority: int
    action_count: int
    deck_cost: int
    needs_boss: bool
    is_active: bool
    damage: int
    target_hp: int

    def rank_key(self) -> tuple[int, ...]:
        """Outcome-first ordering, independent of arbitrary score magnitudes."""
        overkill = max(0, self.damage - self.target_hp)
        return (
            int(self.winning),
            self.prizes,
            self.priority,
            -self.action_count,
            -self.deck_cost,
            int(self.is_active),
            -overkill,
        )


def choose_goal(
    candidates: Iterable[GoalCandidate],
    *,
    locked_target_key: Hashable | None = None,
    role_upgrade_margin: int = 2500,
) -> GoalCandidate | None:
    """Choose the best proven route while avoiding needless target oscillation.

    A target selected earlier in the turn remains the goal when it is still
    reachable.  Replanning is allowed for an immediate win, a larger prize KO,
    or a very large same-prize strategic upgrade.  The lock is therefore a
    consistency aid rather than a rule that can hide a newly found win.
    """
    choices = list(candidates)
    if not choices:
        return None
    best = max(choices, key=lambda candidate: candidate.rank_key())
    if locked_target_key is None:
        return best

    locked = next(
        (candidate for candidate in choices
         if candidate.target_key == locked_target_key),
        None,
    )
    if locked is None:
        return best
    if best.target_key == locked.target_key:
        return locked
    if best.winning and not locked.winning:
        return best
    if best.prizes > locked.prizes:
        return best
    if (
            best.prizes == locked.prizes
            and best.priority >= locked.priority + role_upgrade_margin):
        return best
    return locked


def choose_next_action(actions: Iterable[str]) -> str | None:
    """Return one stable next step; Boss is intentionally last."""
    remaining = set(actions)
    for action in ACTION_ORDER:
        if action in remaining:
            return action
    return min(remaining) if remaining else None


def choose_development_action(actions: Iterable[str]) -> str | None:
    """Choose a stable non-KO development action."""
    remaining = set(actions)
    for action in DEVELOPMENT_ORDER:
        if action in remaining:
            return action
    return min(remaining) if remaining else None


def development_action_score(action: str, competing_score: int = 22000) -> int:
    """Translate a semantic development order into a narrow card score.

    Card scoring remains the compatibility interface of the inherited policy.
    These small separated tiers replace only the old model's constructive
    evolution-versus-Fez ordering; they are not used for KO goals.
    """
    try:
        index = DEVELOPMENT_ORDER.index(action)
    except ValueError:
        return competing_score
    return competing_score + 400 - 100 * index
