from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    name: str
    train_decisions: set[str]
    test_decisions: set[str]
    description: str


def _decision_sets(decisions: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series, name: str, description: str) -> Split:
    return Split(name, set(decisions.loc[train_mask, "decision_id"]), set(decisions.loc[test_mask, "decision_id"]), description)


def make_splits(decisions: pd.DataFrame) -> list[Split]:
    splits: list[Split] = []

    # Within-submission chronological holdout. This prevents a recent submission
    # with globally larger episode IDs from becoming the entire time test set.
    test_mask = pd.Series(False, index=decisions.index)
    cutoffs: list[str] = []
    for submission_id, group in decisions.groupby("submission_id"):
        episodes = sorted(group["episode_id"].unique())
        if len(episodes) < 2:
            continue
        start = max(1, int(len(episodes) * 0.8))
        held = set(episodes[start:])
        test_mask.loc[group.index] = group["episode_id"].isin(held)
        cutoffs.append(f"{int(submission_id)}:{len(held)}/{len(episodes)}")
    splits.append(_decision_sets(
        decisions, ~test_mask, test_mask, "time_holdout",
        "last 20% episodes within each submission (" + ", ".join(cutoffs) + ")",
    ))

    teams = sorted(decisions["target_team"].astype(str).unique())
    held_teams = {team for team in teams if int(hashlib.sha1(team.encode()).hexdigest(), 16) % 5 == 0}
    if not held_teams and teams:
        held_teams = {teams[-1]}
    test = decisions["target_team"].astype(str).isin(held_teams)
    splits.append(_decision_sets(decisions, ~test, test, "team_holdout", "held teams: " + ", ".join(sorted(held_teams))))

    # A second Majkel submission exists, enabling a true same-team submission holdout.
    latest_submission = 54662660 if (decisions["submission_id"] == 54662660).any() else decisions["submission_id"].max()
    test = decisions["submission_id"] == latest_submission
    splits.append(_decision_sets(decisions, ~test, test, "submission_holdout", f"submission {latest_submission}"))

    deck_counts = decisions.groupby("deck_hash")["decision_id"].count().sort_values(ascending=False)
    majkel_hashes = decisions.loc[decisions["majkel_distance"].fillna(999) == 0, "deck_hash"].unique()
    candidates = [deck for deck in deck_counts.index if deck not in set(majkel_hashes)]
    held_deck = candidates[0] if candidates else deck_counts.index[-1]
    test = decisions["deck_hash"] == held_deck
    splits.append(_decision_sets(decisions, ~test, test, "deck_holdout", f"deck hash {held_deck}"))
    return [split for split in splits if split.train_decisions and split.test_decisions]
