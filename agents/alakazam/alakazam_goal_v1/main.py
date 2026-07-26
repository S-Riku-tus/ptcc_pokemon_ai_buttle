"""Deterministic entry point for alakazam_goal_v1."""
from __future__ import annotations

import fallback_policy
from fallback_policy import agent as _goal_agent
from fallback_policy import diag_reset
from fallback_policy import diag_snapshot


_DIAG = fallback_policy._DIAG


def _choose(observation):
    return list(_goal_agent(observation))


# IMPORTANT: Kaggle's Python loader selects the last callable in main.py.
# Keep this function as the final callable definition in this file.
def agent(observation):
    return _choose(observation)
