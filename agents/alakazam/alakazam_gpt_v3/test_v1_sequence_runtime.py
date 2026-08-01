from __future__ import annotations

import json
from pathlib import Path

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from policy_runtime import HybridRanker


HERE = Path(__file__).resolve().parent


def _main_observation(turn: int = 1) -> dict:
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "players": [
                {
                    "hand": [{"id": 741}],
                    "deck": [],
                    "discard": [],
                    "active": [{"id": 741, "hp": 50, "maxHp": 50}],
                    "bench": [],
                    "prize": [],
                },
                {
                    "hand": [],
                    "deck": [],
                    "discard": [],
                    "active": [{"id": 741, "hp": 50, "maxHp": 50}],
                    "bench": [],
                    "prize": [],
                },
            ],
            "logs": [],
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 7, "index": 0},
                {"type": 14},
            ],
        },
    }


def test_v1_sequence_state_starts_empty_and_uses_missing_sentinel():
    runtime = HybridRanker()
    features = runtime._sequence_features(_main_observation()["current"])
    assert features["seq_prev_1_card_id"] == -1
    assert features["seq_prev_4_action_type"] == -1
    assert features["seq_decision_index"] == 0
    assert features["seq_same_turn_decision_index"] == 0
    assert features["seq_last_attack_turn_gap"] == 99


def test_v1_record_choice_updates_previous_action_and_counts():
    runtime = HybridRanker()
    observation = _main_observation(turn=3)
    runtime.record_choice(observation, [0])
    features = runtime._sequence_features(observation["current"])
    assert features["seq_prev_1_card_id"] == 741
    assert features["seq_count_bench"] == 1
    assert features["seq_decision_index"] == 1
    assert features["seq_same_turn_decision_index"] == 1


def test_v1_new_turn_resets_only_same_turn_index():
    runtime = HybridRanker()
    first = _main_observation(turn=3)
    runtime.record_choice(first, [0])
    later = _main_observation(turn=4)
    features = runtime._sequence_features(later["current"])
    assert features["seq_prev_1_card_id"] == 741
    assert features["seq_decision_index"] == 1
    assert features["seq_same_turn_decision_index"] == 0


def test_v1_compact_model_metadata_matches_runtime_contract():
    model = json.loads((HERE / "ranker_model.json").read_text("utf-8"))
    assert model["runtime_scope"] == "v1_yushin_sequence_state_recency_lambdarank"
    assert model["holdout_test_top1"] == 0.8062839059674503
    assert model["holdout_test_top2"] == 0.9329792043399638
    assert len(model["trees"]) == 312
    assert len(model["feature_names"]) == 674
