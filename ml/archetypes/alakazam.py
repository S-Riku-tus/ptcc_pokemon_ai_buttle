from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


DeckAcceptanceRule = Callable[[Iterable[int]], bool]


def _contains_all(card_ids: Iterable[int], required: Iterable[int]) -> bool:
    present = {int(card_id) for card_id in card_ids}
    return all(int(card_id) in present for card_id in required)


@dataclass(frozen=True)
class AlakazamArchetype:
    archetype_name: str = "alakazam"
    core_card_ids: tuple[int, ...] = (741, 742, 743)
    reference_team: str = "Majkel1337"
    reference_deck_selection: str = "latest_or_rank1_modal_exact_team_deck"
    fallback_agent_dir: str = "agents/alakazam/alakazam_ml_v31"
    runtime_agent_dir: str = "agents/alakazam/alakazam_ml_v31"
    important_card_ids: tuple[int, ...] = (
        13, 19, 66, 140, 305, 343, 741, 742, 743, 1079, 1081,
        1086, 1097, 1129, 1152, 1182, 1197, 1225, 1231, 1266,
    )
    hard_fallback_action_types: tuple[str, ...] = ("boss", "retreat", "xerosic", "hammer")
    confidence_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "default_probability": 0.55,
            "default_margin": 0.12,
            "energy_probability": 0.85,
        }
    )

    def deck_acceptance_rule(self, card_ids: Iterable[int]) -> bool:
        return _contains_all(card_ids, self.core_card_ids)

    def deck_distance(self, left: Iterable[int], right: Iterable[int]) -> float:
        from ml.core.deck import substitution_distance

        return substitution_distance(left, right)

    def action_classifier(self, current, option) -> str:
        from ml.core.features import action_type

        return action_type(current, option)

    @property
    def feature_hooks(self) -> tuple[str, ...]:
        return ("ml.core.features.state_features", "ml.core.features.option_features")


ALAKAZAM_ARCHETYPE = AlakazamArchetype()
