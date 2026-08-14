"""The v24 intervention is a mirror-only Froslass-evolution veto."""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from mirror_froslass import MirrorFroslassGuard  # noqa: E402


def body(card_id: int, *, pre: list[dict] | None = None) -> dict:
    return {
        "id": card_id,
        "hp": 100,
        "maxHp": 100,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": list(pre or []),
    }


def player(
    active: list[dict] | None = None,
    bench: list[dict] | None = None,
    hand: list[dict] | None = None,
    discard: list[dict] | None = None,
) -> dict:
    return {
        "active": list(active or []),
        "bench": list(bench or []),
        "hand": list(hand or []),
        "discard": list(discard or []),
        "prize": [None] * 6,
        "deckCount": 30,
        "handCount": len(hand or []),
        "benchMax": 5,
    }


def observation(opponent: dict, *, turn: int = 4) -> tuple[dict, dict]:
    me = player(
        active=[body(mf.MUNKIDORI_ID)],
        bench=[body(mf.SNORUNT_ID)],
        hand=[body(mf.FROSLASS_ID)],
    )
    select = {
        "context": mf.MAIN_CONTEXT,
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {
                "type": 9,
                "index": 0,
                "inPlayArea": mf.AREA_BENCH,
                "inPlayIndex": 0,
            },
            {"type": 14},
            {"type": 7, "index": 0},
        ],
    }
    obs = {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [me, opponent],
        },
        "select": select,
    }
    return obs, select


def test_visible_mirror_vetoes_selected_froslass_evolution() -> None:
    obs, select = observation(player(active=[body(mf.IMPIDIMP_ID)]))
    guard = MirrorFroslassGuard()

    assert guard.observe(obs) is True
    assert guard.adjust(obs, select, 0, {1: 0.4, 2: 0.9}) == 2
    assert guard.stats["froslass_selected"] == 1
    assert guard.stats["overrides"] == 1


def test_veto_does_not_reorder_a_non_froslass_v22_choice() -> None:
    obs, select = observation(player(bench=[body(mf.MORGREM_ID)]))
    guard = MirrorFroslassGuard()
    guard.observe(obs)

    assert guard.adjust(obs, select, 2, {0: 1.0, 1: 0.4, 2: 0.9}) == 2
    assert guard.stats["froslass_offered"] == 1
    assert guard.stats["overrides"] == 0


def test_non_mirror_keeps_the_froslass_evolution() -> None:
    obs, select = observation(player(active=[body(999)]))
    guard = MirrorFroslassGuard()

    assert guard.observe(obs) is False
    assert guard.adjust(obs, select, 0, {0: 1.0, 1: 0.2}) == 0
    assert guard.stats["mirror_decisions"] == 0


def test_detector_uses_discard_and_public_pre_evolutions() -> None:
    guard = MirrorFroslassGuard()
    obs, _ = observation(player(discard=[body(mf.IMPIDIMP_ID)]))
    assert guard.observe(obs) is True

    guard.reset()
    evolved = body(999, pre=[{"id": mf.MORGREM_ID}])
    obs, _ = observation(player(active=[evolved]))
    assert guard.observe(obs) is True


def test_hidden_hand_does_not_identify_the_matchup() -> None:
    guard = MirrorFroslassGuard()
    obs, _ = observation(
        player(active=[body(999)], hand=[body(mf.IMPIDIMP_ID)])
    )
    assert guard.observe(obs) is False

    source = (AGENT_DIR / "mirror_froslass.py").read_text(encoding="utf-8")
    public_reader = source.split("def _public_opponent_ids", 1)[1].split(
        "def _best_by_score", 1
    )[0]
    assert '"hand"' not in public_reader
    assert "deckCount" not in public_reader
    assert '"prize"' not in public_reader


def test_turn_rewind_clears_a_sticky_mirror() -> None:
    guard = MirrorFroslassGuard()
    obs, _ = observation(player(active=[body(mf.IMPIDIMP_ID)]), turn=8)
    assert guard.observe(obs) is True

    next_game, _ = observation(player(active=[body(999)]), turn=1)
    assert guard.observe(next_game) is False
    assert guard.stats["new_game_detected"] == 1


def test_guard_does_not_use_the_rejected_shroud_gate() -> None:
    source = (AGENT_DIR / "mirror_froslass.py").read_text(encoding="utf-8")
    implementation = source.split("class MirrorFroslassGuard", 1)[1]
    assert "shroud_net" not in implementation
