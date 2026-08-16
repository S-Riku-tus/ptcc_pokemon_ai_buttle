from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
AGENT = ROOT / "agents" / "dragapult" / "dragapult_ml_v1"
for path in (AGENT, ROOT / "agents" / "_base", ROOT / "vendor", ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from ml.core.replay_io import deck_hash  # noqa: E402


def fresh(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def current(hand=None):
    return {
        "yourIndex": 0,
        "firstPlayer": 0,
        "turn": 3,
        "turnActionCount": 1,
        "energyAttached": False,
        "retreated": False,
        "stadiumPlayed": False,
        "supporterPlayed": False,
        "stadium": [],
        "players": [
            {
                "active": [{"id": 119, "hp": 70, "maxHp": 70, "energies": []}],
                "bench": [],
                "hand": list(hand or []),
                "handCount": len(hand or []),
                "deckCount": 40,
                "discard": [],
                "prize": [None] * 6,
            },
            {
                "active": [{"id": 119, "hp": 70, "maxHp": 70, "energies": []}],
                "bench": [],
                "hand": None,
                "handCount": 5,
                "deckCount": 40,
                "discard": [],
                "prize": [None] * 6,
            },
        ],
    }


def play_observation(*, optional=False):
    hand = [
        {"id": 119, "serial": 1, "playerIndex": 0},
        {"id": 235, "serial": 2, "playerIndex": 0},
    ]
    return {
        "step": 5,
        "remainingOverageTime": 590,
        "logs": [],
        "current": current(hand),
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 0 if optional else 1,
            "maxCount": 1,
            "option": [
                {"type": 7, "index": 0},
                {"type": 7, "index": 1},
            ],
        },
    }


def test_exact_deck_hash_and_counts():
    fallback = fresh("fallback_policy")
    assert len(fallback.MY_DECK) == 60
    assert deck_hash(fallback.MY_DECK) == "202ee2cec6cbe8b4"
    assert Counter(fallback.MY_DECK)[119] == 4
    assert Counter(fallback.MY_DECK)[121] == 3


def test_features_ignore_opponent_private_hand_and_reject_label_columns():
    features = fresh("ml_features")
    base = current([])
    first = features.state_features(base)
    base["players"][1]["hand"] = [
        {"id": 121}, {"id": 121}, {"id": 1080},
    ]
    second = features.state_features(base)
    assert first == second
    features.assert_no_leakage(list(first))
    try:
        features.assert_no_leakage(["turn", "final_reward"])
    except ValueError:
        pass
    else:
        raise AssertionError("final_reward must be rejected")


def test_runtime_falls_back_on_optional_and_unseen_candidate(tmp_path):
    runtime = fresh("ml_runtime")
    model = {
        "feature_names": ["option_type", "candidate_card_id", "teacher_team_id"],
        "trees": [{"v": 0.0}],
        "teacher_team_id": 16380946,
        "teacher_team_code": 0,
        "routed_contexts": [0],
        "runtime_support": {
            "select_context": [0],
            "option_type": [7],
            "candidate_card_id": [119],
            "candidate_attack_id": [-1],
        },
    }
    path = tmp_path / "ranker.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    ranker = runtime.Ranker(str(path))
    assert ranker.choose(play_observation(optional=True)) is None
    assert ranker.snapshot()["optional_fallback"] == 1
    assert ranker.choose(play_observation(optional=False)) is None
    assert ranker.snapshot()["ood_fallback"] == 1


def test_runtime_scores_supported_mandatory_decision(tmp_path):
    runtime = fresh("ml_runtime")
    model = {
        "feature_names": ["candidate_card_id", "teacher_team_id"],
        "trees": [{
            "f": 0, "t": 150.0, "d": "<=", "x": True,
            "l": {"v": 1.0}, "r": {"v": 0.0},
        }],
        "teacher_team_id": 16380946,
        "teacher_team_code": 0,
        "routed_contexts": [0],
        "runtime_support": {
            "select_context": [0],
            "option_type": [7],
            "candidate_card_id": [119, 235],
            "candidate_attack_id": [-1],
        },
    }
    path = tmp_path / "ranker.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    ranker = runtime.Ranker(str(path))
    assert ranker.choose(play_observation()) == 0
    ranker.commit(0)
    assert ranker.snapshot()["ranker_used"] == 1

