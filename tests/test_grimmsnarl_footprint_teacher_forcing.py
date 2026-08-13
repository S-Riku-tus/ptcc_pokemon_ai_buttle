from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_grimmsnarl_v21_footprint import answers
from scripts.parallel_policy_footprint import _numeric_leaves


class FakeModule:
    def __init__(self) -> None:
        self._RANKER = SimpleNamespace(teacher_forced=False)
        self.proposal_commits = 0
        self.external_commits = 0

    def diag_reset(self) -> None:
        self._RANKER.teacher_forced = False

    def agent(self, observation):
        if not self._RANKER.teacher_forced:
            self.proposal_commits += 1
        return [0]

    def observe_external(self, observation, chosen) -> None:
        self.external_commits += 1


def test_footprint_advances_history_only_with_stored_action() -> None:
    observation = {
        "current": {"turn": 1, "players": [{}, {}]},
        "select": {
            "context": 0,
            "option": [{"type": 0}, {"type": 1}],
        },
    }
    replay = {
        "steps": [
            [{"observation": observation}],
            [{"action": [1]}],
        ]
    }
    module = FakeModule()

    assert answers(module, replay, 0) == [(0, 1, 0, 0)]
    assert module._RANKER.teacher_forced is True
    assert module.proposal_commits == 0
    assert module.external_commits == 1


def test_numeric_diagnostics_are_flattened_without_status_strings() -> None:
    assert _numeric_leaves({
        "ranker": {"used": 3},
        "strategy": {"overrides": 1},
        "load_error": None,
        "enabled": True,
    }) == {
        "ranker.used": 3,
        "strategy.overrides": 1,
    }
