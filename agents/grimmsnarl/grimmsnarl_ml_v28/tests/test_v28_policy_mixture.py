from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
V22 = AGENT.parent / "grimmsnarl_ml_v22"
V25 = AGENT.parent / "grimmsnarl_ml_v25"
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

import main  # noqa: E402
import policy_router  # noqa: E402


class FakeRanker:
    def __init__(self, answer: int) -> None:
        self.answer = answer
        self.teacher_forced = False
        self.last_scores = {0: 0.0, 1: 1.0}
        self.commits: list[int] = []

    @staticmethod
    def is_scorable(_select) -> bool:
        return True

    def choose(self, _observation) -> int:
        return self.answer

    def commit(self, chosen: int) -> None:
        self.commits.append(chosen)

    def observe_external(self, _observation, chosen: int) -> None:
        self.commits.append(chosen)

    def reset(self) -> None:
        self.commits.clear()

    def snapshot(self) -> dict:
        return {"commits": len(self.commits)}


class FakeRouter:
    def __init__(self, route: str) -> None:
        self.route = route

    def choose(self, _observation) -> str:
        return self.route


def observation() -> dict:
    return {
        "current": {"turn": 2, "yourIndex": 0, "players": [{}, {}]},
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}, {"type": 14}],
        },
    }


def install_fakes(monkeypatch, route: str):
    race = FakeRanker(1)
    wall = FakeRanker(0)
    monkeypatch.setattr(main, "_RACE", race)
    monkeypatch.setattr(main, "_WALL", wall)
    monkeypatch.setattr(main, "_ROUTER", FakeRouter(route))
    monkeypatch.setattr(main, "_PLANNER", None)
    monkeypatch.setattr(main, "_PLANNER_DISABLED", True)
    monkeypatch.setattr(main, "_TRAJECTORY", None)
    monkeypatch.setattr(main, "_WALL_BREAK", None)
    monkeypatch.setattr(main, "_DECK_CLOCK", None)
    monkeypatch.setattr(main, "_fallback_agent", lambda _obs: [0])
    monkeypatch.delenv("GRIMMSNARL_V28_POLICY", raising=False)
    return race, wall


def test_default_route_is_v25_and_commits_its_action_to_both_histories(
    monkeypatch,
) -> None:
    race, wall = install_fakes(monkeypatch, "v8_default")
    assert main.agent(observation()) == [1]
    assert race.commits == [1]
    assert wall.commits == [1]
    assert main._LAST_TRACE["policy"] == "v25"


def test_public_wall_route_is_v22_and_commits_its_action_to_both_histories(
    monkeypatch,
) -> None:
    race, wall = install_fakes(monkeypatch, "v8_wall_guarded")
    assert main.agent(observation()) == [0]
    assert race.commits == [0]
    assert wall.commits == [0]
    assert main._LAST_TRACE["policy"] == "v22"


def test_probe_override_can_pin_either_policy(monkeypatch) -> None:
    install_fakes(monkeypatch, "v8_wall_guarded")
    monkeypatch.setenv("GRIMMSNARL_V28_POLICY", "v25")
    assert main.agent(observation()) == [1]

    install_fakes(monkeypatch, "v8_default")
    monkeypatch.setenv("GRIMMSNARL_V28_POLICY", "v22")
    assert main.agent(observation()) == [0]


def test_model_provenance_and_removed_v27_layers() -> None:
    metadata = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    race = (AGENT / "ranker_model.json").read_bytes()
    wall = (AGENT / "ranker_v22_model.json").read_bytes()

    assert race == (V25 / "ranker_model.json").read_bytes()
    assert wall == (V22 / "ranker_model.json").read_bytes()
    assert hashlib.sha256(race).hexdigest() == metadata["race_ranker"]["sha256"]
    assert hashlib.sha256(wall).hexdigest() == metadata["wall_ranker"]["sha256"]
    assert metadata["deck_hash"] == "9714ab5c3996f6cc"
    assert metadata["deck_changed"] is False
    assert not (AGENT / "belief_state.py").exists()
    assert not (AGENT / "h2_search.py").exists()
    assert not (AGENT / "value_model.json").exists()


def test_runtime_keeps_legacy_froslass_escalation_off() -> None:
    runtime = (AGENT / "ml_runtime.py").read_text(encoding="utf-8")
    assert 'ESCALATION_MODE = "off"' in runtime


def test_teal_ogerpon_has_telemetry_without_changing_the_race_policy() -> None:
    obs = {
        "current": {
            "turn": 3,
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 96}], "bench": [], "discard": []},
            ],
            "stadium": [],
        },
        "select": {"option": []},
    }
    assert policy_router.classify(obs) == policy_router.OGERPON
    assert main._policy_name(policy_router.OGERPON) == "v25"
