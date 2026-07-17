from __future__ import annotations

import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import self_play  # noqa: E402


def test_run_matchup_can_reseed_every_game(monkeypatch):
    observed: list[float] = []

    def fake_play_one(seat_specs, seat_agents, seat_decks, max_steps, **kwargs):
        del seat_agents, seat_decks, max_steps, kwargs
        observed.append(random.random())
        return self_play.GameResult(
            matchup="a__vs__b",
            game=0,
            seat0=seat_specs[0],
            seat1=seat_specs[1],
            winner=seat_specs[0],
            result="win",
        )

    monkeypatch.setattr(self_play, "play_one", fake_play_one)
    agent = lambda observation: [5] * 60
    a = self_play.AgentRuntime("a", agent, [5] * 60)
    b = self_play.AgentRuntime("b", agent, [5] * 60)
    self_play.run_matchup(
        a,
        b,
        games=3,
        max_steps=10,
        quiet=True,
        game_seed_base=900,
    )
    assert observed == [random.Random(900 + index).random() for index in range(3)]
