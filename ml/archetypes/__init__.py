from __future__ import annotations

from .alakazam import ALAKAZAM_ARCHETYPE, AlakazamArchetype


def get_archetype(name: str):
    normalized = name.lower().strip()
    if normalized in {"alakazam", "alakazam_ml_v2_expanded"}:
        return ALAKAZAM_ARCHETYPE
    raise KeyError(f"Unknown ML archetype: {name}")


__all__ = ["ALAKAZAM_ARCHETYPE", "AlakazamArchetype", "get_archetype"]

