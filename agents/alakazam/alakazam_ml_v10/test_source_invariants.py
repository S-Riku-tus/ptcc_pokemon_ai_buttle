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


def test_v10_keeps_the_evidence_backed_v9_deck_fixed():
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


def test_v9_has_mist_reservation_and_fez_mode_authority():
    source = (ROOT / "fallback_v3.py").read_text(encoding="utf-8")
    assert "def _mist_probability" in source
    assert "def _should_reserve_last_hammer" in source
    assert "energyIndex" in source
    assert "def _fez_mode" in source
    assert all(mode in source for mode in (
        "DO_NOT_BENCH", "DRAW_ONLY", "PIVOT", "ALTERNATE_ATTACKER"
    ))


def test_shadow_runtime_keeps_boss_rule_only():
    source = (ROOT / "ml_runtime.py").read_text(encoding="utf-8")
    assert '"boss"' in source
    assert "RULE_ONLY_ACTIONS" in source


def test_metadata_is_v10_and_keeps_strategic_ml_shadow_only():
    data = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    assert data["name"] == "alakazam_ml_v10"
    assert data["version"] == "1.0.0"
    assert data["base"] == "alakazam741_v3"
    assert "3 Boss's Orders" in data["deck_policy"]
    assert len(data["v9_rule_changes"]) >= 5


def test_v10_has_general_support_pivot_and_runway_features():
    fallback = (ROOT / "fallback_v3.py").read_text(encoding="utf-8")
    features = (ROOT / "ml_features.py").read_text(encoding="utf-8")
    assert "def _support_pivot_ready" in fallback
    assert "ONE_ENERGY_PIVOT_IDS" in fallback
    assert "support_pivot_ready" in features
    assert "deck_runway_margin" in features
    assert "opp_spread_package_count" in features
