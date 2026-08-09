from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.pop("policy_router", None)
router = importlib.import_module("policy_router")


def observation(
    opponent_ids: list[int] | None = None,
    *,
    turn: int = 2,
    discard_ids: list[int] | None = None,
    deck_ids: list[int] | None = None,
    stadium_id: int | None = None,
) -> dict:
    opponent = {
        "active": [
            {"id": card_id, "preEvolution": []}
            for card_id in (opponent_ids or [])[:1]
        ],
        "bench": [
            {"id": card_id, "preEvolution": []}
            for card_id in (opponent_ids or [])[1:]
        ],
        "discard": [{"id": card_id} for card_id in (discard_ids or [])],
        # Deliberate trap: the router must never use hidden deck contents.
        "deck": [{"id": card_id} for card_id in (deck_ids or [])],
    }
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "players": [{"active": [], "bench": []}, opponent],
            "stadium": [] if stadium_id is None else [{"id": stadium_id}],
        },
        "select": {"option": [{"type": 14}]},
    }


def test_signature_routes() -> None:
    assert router.classify(observation([741])) == router.ALAKAZAM
    assert router.classify(observation([646])) == router.MIRROR
    assert router.classify(observation([344])) == router.WALL
    assert router.classify(observation([117])) == router.WALL
    assert (
        router.classify(observation([756], stadium_id=1247))
        == router.WALL
    )


def test_hidden_deck_is_not_used() -> None:
    obs = observation([756], deck_ids=[741, 742, 743])
    assert router.classify(obs) == router.DEFAULT


def test_default_waits_through_turn_one_then_locks() -> None:
    policy = router.PolicyRouter()
    assert policy.choose(observation([756], turn=1)) == router.DEFAULT
    assert policy.route == router.PENDING
    assert policy.choose(observation([756], turn=2)) == router.DEFAULT
    assert policy.route == router.DEFAULT


def test_specialist_is_sticky_for_the_game() -> None:
    policy = router.PolicyRouter()
    assert policy.choose(observation([741], turn=1)) == router.ALAKAZAM
    # A later wall-looking state cannot stitch a different expert into the
    # already selected Alakazam trajectory.
    assert policy.choose(observation([345], turn=6)) == router.ALAKAZAM
    assert policy.snapshot()["locks"] == {router.ALAKAZAM: 1}


def test_default_can_only_promote_one_way_to_wall() -> None:
    policy = router.PolicyRouter()
    assert policy.choose(observation([756], turn=2)) == router.DEFAULT
    assert policy.choose(observation([344], turn=5)) == router.WALL
    assert policy.choose(observation([741], turn=7)) == router.WALL
    assert policy.snapshot()["transitions"] == [
        {"from": router.PENDING, "to": router.DEFAULT, "turn": 2},
        {"from": router.DEFAULT, "to": router.WALL, "turn": 5},
    ]


def test_wall_can_be_identified_from_public_discard() -> None:
    assert (
        router.classify(observation([756], discard_ids=[344]))
        == router.WALL
    )


def test_v13_artifacts_and_no_broad_search_dependency() -> None:
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert metadata["deck_hash"] == "9714ab5c3996f6cc"
    assert metadata["search"]["broad_whole_turn_search"] is False
    assert (ROOT / "ranker_model.json").exists()
    assert (ROOT / "ranker_model_v9.json").exists()
    assert "from arithmetic_search" not in source
    assert "ranker_model_v9.json" in source
