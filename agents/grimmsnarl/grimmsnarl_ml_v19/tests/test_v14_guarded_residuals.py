from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from matchup_guard import WallSafetyGuard  # noqa: E402


ATTACK = {"type": 13, "attackId": mf.SHADOW_BULLET_ID}
END = {"type": 14}


def observation(
    *,
    active_id: int = 345,
    bench_hp: int = 100,
    context: int = mf.MAIN_CONTEXT,
) -> dict:
    return {
        "current": {
            "turn": 5,
            "yourIndex": 0,
            "players": [
                {
                    "active": [{"id": mf.GRIMMSNARL_EX_ID, "hp": 340}],
                    "bench": [],
                    "hand": [{"id": 646}],
                    "prize": [{}] * 4,
                },
                {
                    "active": [{"id": active_id, "hp": 100}],
                    "bench": [{"id": 1, "hp": bench_hp}],
                    "discard": [],
                    "prize": [{}] * 4,
                },
            ],
            "stadium": [],
        },
        "select": {
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                ATTACK,
                {"type": 7, "area": mf.AREA_HAND, "index": 0},
                END,
            ],
        },
    }


def test_wall_guard_uses_one_fallback_development_action() -> None:
    guard = WallSafetyGuard()
    obs = observation()
    assert guard.adjust(obs, obs["select"], 0, [1]) == 1
    assert guard.stats["development_overrides"] == 1


def test_wall_guard_keeps_the_free_swing_when_fallback_closes() -> None:
    guard = WallSafetyGuard()
    obs = observation()
    assert guard.adjust(obs, obs["select"], 0, [2]) == 0
    assert guard.stats["closing_fallback_kept"] == 1


def test_wall_guard_keeps_a_bench_prize() -> None:
    guard = WallSafetyGuard()
    obs = observation(bench_hp=20)
    assert guard.adjust(obs, obs["select"], 0, [1]) == 0
    assert guard.stats["bench_prize_kept"] == 1


def test_wall_guard_never_reaches_damageable_active_or_other_context() -> None:
    guard = WallSafetyGuard()
    damageable = observation(active_id=1)
    assert guard.adjust(damageable, damageable["select"], 0, [1]) == 0
    other = observation(context=7)
    assert guard.adjust(other, other["select"], 0, [1]) == 0
    assert guard.stats["development_overrides"] == 0


def test_router_is_public_telemetry_and_detects_late_wall() -> None:
    sys.modules.pop("policy_router", None)
    router = importlib.import_module("policy_router")
    policy = router.PolicyRouter()
    generic = observation(active_id=1)
    generic["current"]["players"][1]["deck"] = [{"id": 741}, {"id": 345}]
    assert router.classify(generic) == router.DEFAULT
    assert policy.choose(generic) == router.DEFAULT

    wall = observation(active_id=344)
    assert policy.choose(wall) == router.WALL
    snapshot = policy.snapshot()
    assert snapshot["route"] == router.WALL
    assert snapshot["policy_switches"] == 0


def test_v13_full_policy_experts_stay_removed() -> None:
    # The v15 artifact assertions live in tests/test_v15_attack_access.py; this
    # one only guards the two v13 specialists against coming back.
    source = (AGENT_DIR / "main.py").read_text(encoding="utf-8")
    assert not (AGENT_DIR / "ranker_model_v9.json").exists()
    assert "wall_state_machine" not in source
    assert "ranker_model_v9.json" not in source
    assert json.loads(
        (AGENT_DIR / "metadata.json").read_text(encoding="utf-8")
    )["router"]["policy_switches"] == 0
