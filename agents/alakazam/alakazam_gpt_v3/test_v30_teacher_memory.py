"""Focused regressions for v30 memory and fallback layering."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

from teacher_memory import (
    resolve_semantic_action,
    semantic_action_key,
    teacher_memory_keys,
)
from policy_features import candidate_card, observation_features


HERE = Path(__file__).resolve().parent


def _observation() -> dict:
    card = {
        "id": 741,
        "serial": 10,
        "hp": 50,
        "maxHp": 50,
        "energyCards": [],
        "tools": [],
    }
    return {
        "step": 7,
        "remainingOverageTime": 42.0,
        "logs": [{"type": 6, "playerIndex": 0, "cardId": 741}],
        "current": {
            "yourIndex": 0,
            "turn": 2,
            "players": [
                {
                    "hand": [card, {**card, "serial": 11}],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [],
                },
                {
                    "hand": [],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [],
                },
            ],
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 7, "index": 0},
                {"type": 7, "index": 1},
            ],
        },
    }


def test_v30_exact_key_ignores_clock_and_step_only():
    observation = _observation()
    exact, canonical = teacher_memory_keys(observation)
    observation["step"] = 99
    observation["remainingOverageTime"] = 1.0

    assert teacher_memory_keys(observation) == (exact, canonical)


def test_v30_canonical_key_ignores_serial_identity():
    observation = _observation()
    _, canonical = teacher_memory_keys(observation)
    observation["current"]["players"][0]["hand"][0]["serial"] = 999

    assert teacher_memory_keys(observation)[1] == canonical


def test_v30_memory_resolves_an_equivalent_card_copy():
    observation = _observation()
    semantic = semantic_action_key(observation, [1])

    assert resolve_semantic_action(observation, semantic) == [0]


def test_v30_memory_artifact_is_compact_and_conflict_free():
    payload = json.loads(
        zlib.decompress((HERE / "teacher_memory.bin").read_bytes())
    )

    assert payload["format"] == "v30_teacher_memory_v1"
    assert len(payload["exact"]) == 199_558
    assert len(payload["canonical_repeated"]) == 2


def test_v30_preserves_frozen_v29_model_separately():
    model = json.loads((HERE / "v29_ranker_model.json").read_text("utf-8"))

    assert model["runtime_scope"] == "v29_residual_main_policy"
    assert model["training_decisions"] == 18_336
    assert len(model["feature_names"]) == 422


def test_v30_search_candidate_uses_public_select_deck():
    current = {
        "yourIndex": 0,
        "players": [{"hand": []}, {"hand": []}],
    }
    select = {"deck": [{"id": 743}, {"id": 5}]}

    assert candidate_card(
        current,
        {"type": 3, "area": 1, "index": 1},
        select,
    )["id"] == 5


def test_v30_observation_features_retain_current_turn_sequence():
    observation = _observation()
    observation["logs"] = [
        {"type": 2, "playerIndex": 0},
        {"type": 10, "playerIndex": 0, "cardId": 1182},
        {"type": 4, "playerIndex": 0, "cardId": 5},
    ]
    features = observation_features(observation)

    assert features["turn_self_log_type_10"] == 1
    assert features["turn_self_log_type_4"] == 1
    assert features["recent_log_0_card_id"] == 5
    assert features["recent_log_1_card_id"] == 1182
