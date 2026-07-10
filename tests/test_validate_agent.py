from pathlib import Path

from scripts.validate_agent import read_deck, validate_agent

ROOT = Path(__file__).resolve().parents[1]


def test_current_deck_has_60_cards():
    deck = read_deck(
        ROOT / "agents" / "alakazam741_v2" / "deck.csv"
    )
    assert len(deck) == 60


def test_current_agent_static_validation():
    result = validate_agent(
        ROOT / "agents" / "alakazam741_v2"
    )
    assert result["deck_size"] == 60
    assert result["metadata"]["name"] == "alakazam741_v2"
