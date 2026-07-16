from __future__ import annotations

from collections import Counter
from typing import Iterable


def counts(card_ids: Iterable[int]) -> Counter[int]:
    return Counter(int(x) for x in card_ids)


def substitution_distance(a: Iterable[int], b: Iterable[int]) -> float:
    """Half L1 count distance: number of card substitutions between equal-size decks."""
    ca, cb = counts(a), counts(b)
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] - cb[k]) for k in keys)


def has_core_cards(card_ids: Iterable[int], core_card_ids: Iterable[int]) -> bool:
    required = {int(x) for x in core_card_ids}
    return required.issubset(set(int(x) for x in card_ids))


def classify_deck(
    card_ids: Iterable[int],
    reference: Iterable[int],
    *,
    core_card_ids: Iterable[int],
    exact_label: str = "reference_exact",
    near_label: str = "reference_near",
    variant_label: str = "archetype_variant",
    other_label: str = "archetype_other",
) -> tuple[str, float]:
    cards = list(card_ids)
    if not has_core_cards(cards, core_card_ids):
        return "outside_archetype", float("inf")
    distance = substitution_distance(cards, reference)
    if distance == 0:
        return exact_label, distance
    if distance <= 4:
        return near_label, distance
    if distance <= 10:
        return variant_label, distance
    return other_label, distance


def _alakazam_core_ids() -> tuple[int, ...]:
    from ml.archetypes import get_archetype

    return tuple(get_archetype("alakazam").core_card_ids)


def has_alakazam_line(card_ids: Iterable[int]) -> bool:
    """Compatibility alias; Alakazam IDs live in the archetype plugin."""
    return has_core_cards(card_ids, _alakazam_core_ids())


def classify(card_ids: Iterable[int], reference: Iterable[int]) -> tuple[str, float]:
    """Compatibility alias used by the current Alakazam pipeline."""
    deck_type, distance = classify_deck(
        card_ids,
        reference,
        core_card_ids=_alakazam_core_ids(),
        exact_label="majkel_exact",
        near_label="majkel_near",
        variant_label="alakazam_variant",
        other_label="alakazam_other",
    )
    if deck_type == "outside_archetype":
        return "non_alakazam", distance
    return deck_type, distance


def major_differences(card_ids: Iterable[int], reference: Iterable[int], limit: int = 12) -> list[dict[str, int]]:
    current, ref = counts(card_ids), counts(reference)
    differences = []
    for card_id in set(current) | set(ref):
        delta = current[card_id] - ref[card_id]
        if delta:
            differences.append({"card_id": int(card_id), "count": int(current[card_id]), "majkel_count": int(ref[card_id]), "delta": int(delta)})
    differences.sort(key=lambda row: (-abs(row["delta"]), row["card_id"]))
    return differences[:limit]
