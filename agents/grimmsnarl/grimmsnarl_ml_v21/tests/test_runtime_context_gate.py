"""Context routing must not skew the ranker's intra-turn history."""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_features as F  # noqa: E402
from ml_runtime import Ranker  # noqa: E402


def select(context: int) -> dict:
    return {
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": 14}, {"type": 10}],
    }


def test_dropped_context_is_not_routed_but_stays_in_corpus_history() -> None:
    ranker = object.__new__(Ranker)
    ranker.contexts = frozenset({F.MAIN_CONTEXT})
    thin = select(8)
    assert ranker.is_scorable(thin) is False
    assert ranker.is_corpus_scorable(thin) is True


def test_empty_routed_context_list_stays_empty() -> None:
    ranker = object.__new__(Ranker)
    ranker.contexts = frozenset()
    assert ranker.is_scorable(select(F.MAIN_CONTEXT)) is False


def test_teacher_forced_commit_does_not_advance_own_choice() -> None:
    ranker = object.__new__(Ranker)
    ranker.teacher_forced = True
    ranker._pending = [{"candidate_card_id": 1}, {"candidate_card_id": 2}]
    called: list[int] = []
    ranker.note_decision = lambda features, chosen: called.append(chosen)
    ranker.commit(0)
    assert called == []
    assert ranker._pending is None
