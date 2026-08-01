from __future__ import annotations

from array import array
import json
from pathlib import Path

import test_v11_runtime_logic as harness

harness.install_cg_stub()

from policy_runtime import HybridRanker


HERE = Path(__file__).resolve().parent


def _feature(action_type: int, card_id: int, position: int) -> dict:
    spec = json.loads((HERE / "type_runtime_spec.json").read_text("utf-8"))
    feature = {name: 0.0 for name in spec["state_names"]}
    feature.update({name: 0.0 for name in spec["candidate_fields"]})
    feature.update({name: 0.0 for name in spec["rich_top_fields"]})
    feature.update({name: 0.0 for name in spec["rich_agg_fields"]})
    feature.update({
        "action_type_id": action_type,
        "candidate_card_id": card_id,
        "candidate_target_id": -1,
        "candidate_option_position": position,
    })
    return feature


def test_v3_rich_meta_row_matches_training_contract():
    runtime = HybridRanker()
    features = [_feature(2, 741, 0), _feature(4, -1, 1)]
    row = runtime._type_meta_row(features, [0, 1], [1.5, 0.25])
    assert len(row) == 1052
    assert len(runtime.type_model["cols"]) == 356
    assert max(runtime.type_model["cols"]) < len(row)


def test_v3_type_model_has_expected_compact_shape():
    runtime = HybridRanker()
    assert runtime.type_model["format"] == "v3_xgb_rich_type_v1"
    assert len(runtime.type_model["trees"]) == 2497
    assert runtime.type_model["classes"] == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]


def test_v3_split_thresholds_are_float32_quantized_once_at_load():
    runtime = HybridRanker()
    assert runtime.type_model["trees"]
    assert isinstance(runtime.type_model["trees"][0]["v"], array)
    assert runtime.type_model["trees"][0]["v"].typecode == "f"
