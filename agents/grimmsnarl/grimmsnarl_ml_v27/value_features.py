"""Public state-only features for the v7 board-value model.

A full-turn search leaf is observed from the next player's point of view.
Private hand identities and flags which describe only the acting player's
current turn are therefore removed before a state is normalised to the seat
whose eventual win probability is being estimated.  Candidate/action columns
remain absent by construction.
"""

from __future__ import annotations

from typing import Any

import ml_features


_PRIVATE_EXACT = frozenset({
    "energy_attached",
    "retreated",
    "stadium_played",
    "supporter_played",
    "has_impidimp_anywhere",
    "has_grimmsnarl_in_hand",
    "candy_route_available",
    "morgrem_route_available",
    "punk_up_available",
    "snorunt_ready_to_evolve",
    "energy_in_hand",
    "attachment_still_available",
    "attach_enabling_target_count",
    "visible_dark_energy_count",
    "dark_energy_remaining_estimate",
    "dark_hit_probability_draw3",
    "dark_hit_probability_draw6",
})
_PRIVATE_PREFIXES = ("hand_", "boss_", "route_boss_")


def is_public_feature(name: str) -> bool:
    return (
        name not in _PRIVATE_EXACT
        and not name.startswith(_PRIVATE_PREFIXES)
    )


def value_features(
    observation: dict[str, Any],
    perspective: int | None = None,
) -> dict[str, float | int]:
    """Build public state features from ``perspective``'s point of view."""
    current = observation.get("current") or {}
    acting = int(current.get("yourIndex", 0) or 0)
    seat = acting if perspective is None else int(perspective)

    # state_features only reads nested structures.  A shallow copy is enough
    # to change the perspective without mutating the engine observation.
    normalized = dict(current)
    normalized["yourIndex"] = seat
    raw = ml_features.state_features(normalized)
    public = {
        name: value for name, value in raw.items()
        if is_public_feature(name)
    }
    public["perspective_turn"] = int(acting == seat)
    return public


def vector(
    observation: dict[str, Any],
    names: list[str],
    perspective: int | None = None,
) -> list[float]:
    """Return a model-ordered dense row; absent conditional values are -1."""
    row = value_features(observation, perspective)
    return [float(row.get(name, -1)) for name in names]
