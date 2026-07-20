from collections import Counter
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parent


def test_sources_parse():
    for name in (
        "main.py", "fallback_v3.py", "fallback_v12.py", "policy_base.py",
        "ml_runtime.py", "ml_features.py", "common_runtime.py",
    ):
        ast.parse((ROOT / name).read_text(encoding="utf-8"))


def test_v8_adds_single_shaymin_without_removing_max_rod():
    counts = Counter(map(int, (ROOT / "deck.csv").read_text(encoding="utf-8").split()))
    assert sum(counts.values()) == 60
    assert counts[305] == 3 and counts[66] == 2
    assert counts[5] == 2 and counts[19] == 4
    assert counts[1182] == 3
    assert counts[13] == 0 and counts[343] == 1
    assert counts[142] == 0 and counts[1156] == 0 and counts[858] == 0
    assert counts[1110] == 1


def test_authoritative_runtime_has_absolute_dudunsparce_guard():
    source = (ROOT / "fallback_v3.py").read_text(encoding="utf-8")
    assert "def _board_body_count" in source
    assert "if self._board_body_count() <= 1:" in source
    assert "o.area != AreaType.BENCH and not any" in source


def test_authoritative_runtime_has_value_gated_boss_logic():
    source = (ROOT / "fallback_v3.py").read_text(encoding="utf-8")
    assert "ROCKET_ARTICUNO_ID = 414" in source
    assert "FROSLASS_ID = 104" in source
    assert "def _boss_damage_after_spend" in source
    assert "def _boss_role_bonus" in source
    assert "def _boss_target_score" in source
    assert "Never replace an available Active KO with an equal/worse Bench KO" in source
    assert "def _boss_two_hit_target_score" in source
    assert "return self._boss_two_hit_target_score(target, damage)" in source


def test_shadow_runtime_keeps_boss_rule_only():
    source = (ROOT / "ml_runtime.py").read_text(encoding="utf-8")
    assert '"boss"' in source
    assert "RULE_ONLY_ACTIONS" in source


def test_metadata_is_v8():
    data = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    assert data["name"] == "alakazam_ml_v8"
    assert data["version"] == "0.8.0"
    assert data["base"] == "alakazam741_v3"
    assert "3 Boss's Orders" in data["deck_policy"]
