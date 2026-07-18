from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_attack_turn_continuity import aggregate, analyze_replay  # noqa: E402


def _seat_state(turn, options, *, status="ACTIVE", action=None):
    return {
        "status": status,
        "action": [] if action is None else action,
        "observation": {
            "current": {
                "turn": turn,
                "yourIndex": 0,
                "players": [
                    {"active": [{"id": 743}], "bench": []},
                    {"active": [{"id": 140}], "bench": []},
                ],
            },
            "select": {"type": 0, "context": 0, "option": options},
        },
    }


def _inactive_state(action):
    return {"status": "INACTIVE", "action": action, "observation": {}}


def test_attack_continuity_separates_opportunities_from_main_turns():
    attack = {"type": 13, "attackId": 1072}
    end = {"type": 14}
    other = {"type": 7}
    replay = {
        "info": {"EpisodeId": 42, "TeamNames": ["teacher", "opponent"]},
        "rewards": [1, -1],
        "steps": [
            [_seat_state(1, [attack, end]), _inactive_state([])],
            [_seat_state(2, [end, other], action=[0]), _inactive_state([])],
            [_seat_state(3, [attack, end], action=[0]), _inactive_state([])],
            [_inactive_state([1]), _inactive_state([])],
        ],
    }

    row = analyze_replay(replay, "teacher")
    assert row is not None
    assert row["main_turns"] == 3
    assert row["attack_turns"] == 1
    assert row["attack_opportunity_turns"] == 2
    assert row["missed_attack_opportunity_turns"] == 1
    assert row["attackable_end_turns"] == 1
    assert row["main_turns_before_first_attack"] == 0

    report = aggregate([row])
    assert report["all_acting_turn_attack_rate"] == 1 / 3
    assert report["main_turn_attack_rate"] == 1 / 3
    assert report["attack_opportunity_conversion_rate"] == 1 / 2
    assert report["post_first_main_turn_attack_rate"] == 1 / 3
