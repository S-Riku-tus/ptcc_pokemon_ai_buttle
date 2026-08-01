from __future__ import annotations

import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vendor"))

import main
from scripts.build_submission import build


HERE = Path(__file__).resolve().parent


def test_exact_60_card_teacher_deck():
    deck = [int(value) for value in (HERE / "deck.csv").read_text().splitlines()]
    assert len(deck) == 60
    assert deck == main.MY_DECK
    assert Counter(deck) == Counter({
        11: 4, 13: 1, 14: 3, 66: 4, 174: 1, 305: 4,
        848: 4, 849: 3, 1086: 4, 1121: 4, 1122: 4, 1152: 4,
        1174: 4, 1182: 3, 1197: 1, 1225: 4, 1227: 4, 1229: 4,
    })


def test_deck_request_resets_and_returns_copy():
    first = main.agent({"select": None})
    second = main.agent({"select": None})
    assert first == main.MY_DECK
    assert second == main.MY_DECK
    assert first is not main.MY_DECK


def test_compact_models_are_real_and_match_metadata():
    for filename in ("ranker_model.json", "count_model.json"):
        payload = json.loads((HERE / filename).read_text())
        assert payload["format"] == "lightgbm_tree_v2"
        assert len(payload["trees"]) == payload["tree_count"]
        assert payload["feature_names"]
    deep = json.loads((HERE / "deepset_model.json").read_text())
    assert deep["format"] == "lopunny_deepset_v1"
    assert deep["tensors"]
    assert deep["base_weight"] == 2.0


def test_forced_all_options_is_legal_without_model_walk():
    observation = {
        "select": {
            "minCount": 2,
            "maxCount": 2,
            "option": [{"type": 1}, {"type": 2}],
        }
    }
    assert main.agent(observation) == [0, 1]


def test_zero_selection_is_legal():
    observation = {
        "select": {"minCount": 0, "maxCount": 0, "option": [{"type": 1}]}
    }
    assert main.agent(observation) == []


def test_submission_bundle_contains_complete_runtime(tmp_path):
    fake_cg = tmp_path / "cg"
    fake_cg.mkdir()
    (fake_cg / "api.py").write_text("# fake official cg\n", encoding="utf-8")
    output = tmp_path / "lopunny_submission.tar.gz"

    build(HERE, output, fake_cg)

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert {
        "main.py",
        "deck.csv",
        "cg/api.py",
        "imitation_features.py",
        "tree_runtime.py",
        "fallback_policy.py",
        "policy_base.py",
        "ranker_model.json",
        "count_model.json",
        "deepset_model.json",
    } <= names
    assert not any(
        Path(name).name.startswith(
            ("test_", "ANALYSIS_", "CHANGELOG_", "VALIDATION_REPORT_")
        )
        for name in names
    )
    assert "README.md" not in names
    assert "metadata.json" not in names
