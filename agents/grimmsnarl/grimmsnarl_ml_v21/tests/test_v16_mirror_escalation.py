"""The v6 Froslass escalation stands down on mirror boards, and only there.

v6 handed the ``evolve_froslass`` class to pilot 16371703 because the pinned
teacher takes that evolve on 95.7% of its own turns and that pilot on 80.5%.
Off the mirror the shipped combination lands where it should - 85.6% uptake
over v15's 110 rated games.  On mirror boards it lands nowhere either pilot
stands: replaying all 104 stored mirror decisions that offered the evolve,
v15 takes it 4 times with the escalation on and 33 with it off, while the
mirror opponents, on the identical 60 cards, take it on 12 of 12 offering
turns.  Fisher exact on our 6 of 20 against their 12 of 12 is p = 0.000112.

So the change is a suspension, not a new policy: on a mirror board the class
is not escalated and the pinned teacher answers, which is v5's behaviour and
the field's.  Everything else is v15 to the byte.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import ml_runtime  # noqa: E402
import policy_router  # noqa: E402

FROSLASS_CLASS = {
    "name": "froslass_evolve",
    "context": ml_runtime.MAIN_CONTEXT,
    "column": "evolve_froslass",
    "value": 1,
}


class _Stub(ml_runtime.Ranker):
    """A Ranker with the escalation wired up and no 45 MB model behind it."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips the load
        self.names = ["teacher_team_id", "evolve_froslass"]
        self.teacher_code = 16
        self.teacher_index = 0
        self.escalation_mode = "class"
        self.escalation_in_mirror = ml_runtime.escalation_in_mirror()
        self.escalation_classes = (FROSLASS_CLASS,)
        self.escalation_code = ml_runtime.ESCALATION_TEACHER_CODE
        self.reset()


SELECT = {"context": ml_runtime.MAIN_CONTEXT}
OFFERED = [{"evolve_froslass": 0}, {"evolve_froslass": 1}]
NOT_OFFERED = [{"evolve_froslass": 0}, {"evolve_froslass": 0}]


def test_the_class_escalates_by_default() -> None:
    ranker = _Stub()
    assert ranker._escalated_class(SELECT, OFFERED, [0, 1]) is FROSLASS_CLASS
    assert ranker.stats["escalation_suspended_mirror"] == 0


def test_a_mirror_board_suspends_it() -> None:
    ranker = _Stub()
    ranker.suspend_escalation = True
    assert ranker._escalated_class(SELECT, OFFERED, [0, 1]) is None
    assert ranker.stats["escalation_suspended_mirror"] == 1


def test_suspension_only_counts_where_the_class_was_actually_offered() -> None:
    ranker = _Stub()
    ranker.suspend_escalation = True
    assert ranker._escalated_class(SELECT, NOT_OFFERED, [0, 1]) is None
    assert ranker.stats["escalation_suspended_mirror"] == 0


def test_the_env_override_restores_v15() -> None:
    ranker = _Stub()
    ranker.escalation_in_mirror = True
    ranker.suspend_escalation = True
    assert ranker._escalated_class(SELECT, OFFERED, [0, 1]) is FROSLASS_CLASS
    assert ranker.stats["escalation_suspended_mirror"] == 0


def test_reset_clears_the_flag_so_a_reused_process_starts_neutral() -> None:
    ranker = _Stub()
    ranker.suspend_escalation = True
    ranker.reset()
    assert ranker.suspend_escalation is False


def test_only_the_mirror_route_suspends_anything() -> None:
    """The wall and Alakazam routes keep v15's escalation exactly."""
    import main

    for route, expected in (
        (policy_router.MIRROR, True),
        (policy_router.WALL, False),
        (policy_router.ALAKAZAM, False),
        (policy_router.DEFAULT, False),
        (policy_router.PENDING, False),
    ):
        main._note_matchup(route)
        if main._RANKER is not None:
            assert main._RANKER.suspend_escalation is expected


def test_the_router_reads_only_public_information() -> None:
    """The mirror flag must never come from the opponent's hand or deck."""
    source = (AGENT_DIR / "policy_router.py").read_text(encoding="utf-8")
    assert '"hand"' not in source
    assert "deckCount" not in source
    assert '"prize"' not in source
