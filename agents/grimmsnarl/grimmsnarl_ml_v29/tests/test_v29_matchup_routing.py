from __future__ import annotations

import sys
from pathlib import Path


AGENT = Path(__file__).resolve().parents[1]
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

import main  # noqa: E402
import policy_router  # noqa: E402


def observation(*ids: int) -> dict:
    return {
        "current": {
            "turn": 3,
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {
                    "active": ([{"id": ids[0]}] if ids else []),
                    "bench": [{"id": card_id} for card_id in ids[1:]],
                    "discard": [],
                },
            ],
            "stadium": [],
        },
        "select": {"option": []},
    }


def test_lopunny_and_mega_froslass_use_elite_ranker() -> None:
    assert policy_router.classify(observation(848)) == policy_router.LOPUNNY
    assert policy_router.classify(observation(849)) == policy_router.LOPUNNY
    assert policy_router.classify(observation(861)) == policy_router.LOPUNNY
    assert main._policy_name(policy_router.LOPUNNY) == "v22"


def test_hydrapple_is_more_specific_than_teal_ogerpon() -> None:
    assert (
        policy_router.classify(observation(96, 92))
        == policy_router.HYDRAPPLE
    )
    assert main._policy_name(policy_router.HYDRAPPLE) == "v22"


def test_apple_line_without_teal_ogerpon_is_not_hydrapple_route() -> None:
    assert policy_router.classify(observation(92)) == policy_router.DEFAULT
    assert policy_router.classify(observation(93)) == policy_router.DEFAULT


def test_sticky_ogerpon_route_upgrades_after_applin_reveal() -> None:
    router = policy_router.PolicyRouter()
    assert router.choose(observation(96)) == policy_router.OGERPON
    assert router.choose(observation(96, 92)) == policy_router.HYDRAPPLE
    assert router.snapshot()["transitions"][-1]["to"] == policy_router.HYDRAPPLE


def test_pure_teal_ogerpon_stays_on_race_policy() -> None:
    assert policy_router.classify(observation(96)) == policy_router.OGERPON
    assert main._policy_name(policy_router.OGERPON) == "v25"


def test_snorunt_alone_does_not_false_positive_as_lopunny() -> None:
    assert policy_router.classify(observation(860)) == policy_router.DEFAULT
    assert main._policy_name(policy_router.DEFAULT) == "v25"
