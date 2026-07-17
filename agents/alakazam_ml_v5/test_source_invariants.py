from collections import Counter
from pathlib import Path
import ast, json
ROOT=Path(__file__).resolve().parent

def test_sources_parse():
    for name in ("main.py","fallback_v3.py","fallback_v12.py","policy_base.py","ml_runtime.py","ml_features.py","common_runtime.py"):
        ast.parse((ROOT/name).read_text(encoding="utf-8"))

def test_deck_revision():
    c=Counter(map(int,(ROOT/"deck.csv").read_text(encoding="utf-8").split()))
    assert sum(c.values())==60
    assert c[305]==3 and c[66]==3 and c[858]==0 and c[5]==3

def test_role_cleanup():
    f=(ROOT/"fallback_v12.py").read_text(encoding="utf-8"); m=(ROOT/"ml_runtime.py").read_text(encoding="utf-8")
    assert "PSYDUCK" not in f+m
    assert "ALTERNATE_ATTACKER" not in f
    assert "FezMode" not in f
    assert "ATTACKER_IDS = {C.ALAKAZAM, C.KADABRA}" in f
    assert "def _essential_bench_reserve" in f
    assert "return self._open_bench_slots() > self._essential_bench_reserve()" in f
    assert "def _search_secures_backup" in f
    assert "def _search_deck_cost" in f
    assert "if p.id == C.FEZANDIPITI_EX:" in f
    assert "# Bench draw support only" in f
    assert "if self.hand[C.LUCKY_HELMET] <= 0" in f
    assert "if not self._nighttime_mine_tax_stops_active():" in f

def test_metadata():
    d=json.loads((ROOT/"metadata.json").read_text(encoding="utf-8"))
    assert d["version"]=="0.5.0"
    assert "3 Dunsparce / 3 Dudunsparce" in d["deck_policy"]
    assert d["base"]=="alakazam741_v3"
