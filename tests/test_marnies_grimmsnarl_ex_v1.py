from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.agent_loader import load_dir_agent
from scripts.validate_agent import validate_agent


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "grimmsnarl" / "marnies_grimmsnarl_ex_v1"


def test_exact_reconstructed_deck() -> None:
    deck = [int(line) for line in (AGENT_DIR / "deck.csv").read_text().splitlines()]
    assert len(deck) == 60
    assert Counter(deck) == Counter(
        {
            7: 10,
            104: 2,
            112: 4,
            646: 4,
            647: 3,
            648: 3,
            860: 2,
            1079: 3,
            1080: 1,
            1086: 4,
            1097: 3,
            1152: 4,
            1161: 2,
            1182: 2,
            1219: 4,
            1227: 4,
            1231: 1,
            1259: 4,
        }
    )


def test_agent_loads_and_returns_its_deck() -> None:
    agent, diag, module = load_dir_agent(AGENT_DIR)
    assert agent({"select": None}) == module.MY_DECK
    assert diag is module.DIAG


def test_static_validation_and_metadata() -> None:
    result = validate_agent(AGENT_DIR)
    metadata = json.loads((AGENT_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert result["deck_size"] == 60
    assert result["warnings"] == []
    assert metadata["archetype"] == "marnies_grimmsnarl_ex"
