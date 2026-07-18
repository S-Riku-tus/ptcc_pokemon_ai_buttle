"""Analyze a ranked Alakazam teacher corpus and compare it with a baseline ranker."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from cg.api import all_card_data  # noqa: E402
from ml.core.matrix import load_matrix_store  # noqa: E402
from ml.core.matrix_train import _row_indices, _score, evaluate_arrays  # noqa: E402


CARD = {card.cardId: card for card in all_card_data()}


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    accepted = frame[~frame["fallback"]]
    actions: dict[str, Any] = {}
    for action, group in frame.groupby("selected_action_type", observed=True):
        action_accepted = group[~group["fallback"]]
        actions[str(action)] = {
            "count": int(len(group)),
            "top1": float(group["correct"].mean()),
            "top3": float(group["top3"].mean()),
            "mrr": float(group["reciprocal_rank"].mean()),
            "fallback_rate": float(group["fallback"].mean()),
            "accepted_top1": (
                float(action_accepted["correct"].mean()) if len(action_accepted) else None
            ),
        }
    return {
        "decision_count": int(len(frame)),
        "top1": float(frame["correct"].mean()),
        "top3": float(frame["top3"].mean()),
        "mrr": float(frame["reciprocal_rank"].mean()),
        "fallback_rate": float(frame["fallback"].mean()),
        "accepted_count": int(len(accepted)),
        "accepted_top1": float(accepted["correct"].mean()) if len(accepted) else None,
        "action_type_metrics": actions,
    }


def _metric_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("top1", "top3", "mrr", "fallback_rate", "accepted_top1")
    result = {
        key: float(candidate[key] - baseline[key])
        for key in keys
        if baseline.get(key) is not None and candidate.get(key) is not None
    }
    action_delta: dict[str, Any] = {}
    common = set(baseline["action_type_metrics"]) & set(candidate["action_type_metrics"])
    for action in sorted(common):
        old = baseline["action_type_metrics"][action]
        new = candidate["action_type_metrics"][action]
        action_delta[action] = {
            key: float(new[key] - old[key])
            for key in ("top1", "top3", "mrr", "fallback_rate")
        }
    result["action_type_metrics"] = action_delta
    return result


def _archetype(deck: list[int]) -> str:
    pokemon = Counter(
        card_id for card_id in deck
        if CARD.get(card_id) is not None and CARD[card_id].cardType == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple[Any, ...]:
        card_id, count = item
        card = CARD[card_id]
        return (card.stage2, card.megaEx or card.ex, card.stage1, count, card.hp)

    return str(CARD[max(pokemon.items(), key=key)[0]].name)


def _initial_decks(episode: dict[str, Any]) -> list[list[int]]:
    decks: list[list[int]] = [[], []]
    for step in episode.get("steps", [])[:3]:
        for seat in (0, 1):
            action = step[seat].get("action") if seat < len(step) else None
            if isinstance(action, list) and len(action) == 60:
                decks[seat] = [int(value) for value in action]
    return decks


def _final_current(episode: dict[str, Any]) -> dict[str, Any] | None:
    for step in reversed(episode.get("steps", [])):
        for seat_data in step:
            current = (seat_data.get("observation") or {}).get("current")
            if current:
                return current
    return None


def _end_reason(
    current: dict[str, Any] | None,
    seat: int,
    won: bool,
    statuses: list[Any],
) -> str:
    if current is None:
        return "missing_final_state"
    players = current.get("players") or [{}, {}]
    me = players[seat]
    opponent = players[1 - seat]
    own_status = str(statuses[seat] if seat < len(statuses) else "")
    if any(value in own_status.upper() for value in ("ERROR", "TIMEOUT", "INVALID")):
        return "agent_error"
    if won:
        if len(me.get("prize") or []) == 0:
            return "prizes_taken"
        if int(opponent.get("deckCount") or 0) == 0:
            return "opponent_deckout"
        if not (opponent.get("active") or []) and not (opponent.get("bench") or []):
            return "opponent_boardout"
        return "other_win"
    if int(me.get("deckCount") or 0) == 0:
        return "deckout"
    if len(opponent.get("prize") or []) == 0:
        return "opponent_prizes_taken"
    if not (me.get("active") or []) and not (me.get("bench") or []):
        return "boardout"
    return "other_loss"


def _replay_analysis(manifest: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for zip_path, zip_group in manifest.groupby("zip_path", observed=True):
        with zipfile.ZipFile(str(zip_path)) as archive:
            for replay_path, replay_group in zip_group.groupby("replay_path", observed=True):
                episode = json.loads(archive.read(str(replay_path)))
                decks = _initial_decks(episode)
                current = _final_current(episode)
                statuses = list(episode.get("statuses") or [])
                for row in replay_group.itertuples(index=False):
                    seat = int(row.target_seat)
                    won = bool(row.target_win)
                    me = (current.get("players") or [{}, {}])[seat] if current else {}
                    opponent = (current.get("players") or [{}, {}])[1 - seat] if current else {}
                    records.append({
                        "trajectory_id": str(row.trajectory_id),
                        "submission_id": int(row.submission_id),
                        "rank": int(row.rank),
                        "target_team": str(row.target_team),
                        "episode_id": int(row.episode_id),
                        "target_seat": seat,
                        "won": won,
                        "opponent_archetype": _archetype(decks[1 - seat]),
                        "turn": int(current.get("turn") or 0) if current else None,
                        "end_reason": _end_reason(current, seat, won, statuses),
                        "self_deck_left": int(me.get("deckCount") or 0),
                        "opponent_deck_left": int(opponent.get("deckCount") or 0),
                        "self_prizes_left": len(me.get("prize") or []),
                        "opponent_prizes_left": len(opponent.get("prize") or []),
                    })
    return pd.DataFrame(records)


def _selected_card_counts(
    dataset_path: Path,
    decision_ids: set[str],
) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    card_counts: Counter[int] = Counter()
    usecols = ["decision_id", "label", "candidate_card_id", "action_type"]
    for chunk in pd.read_csv(dataset_path, usecols=usecols, chunksize=250_000):
        selected = chunk[
            chunk["decision_id"].astype(str).isin(decision_ids)
            & chunk["label"].eq(1)
        ]
        action_counts.update(selected["action_type"].astype(str))
        card_counts.update(
            int(value) for value in selected["candidate_card_id"].dropna()
            if int(value) >= 0
        )
    cards = [
        {
            "card_id": card_id,
            "card_name": str(CARD[card_id].name) if card_id in CARD else str(card_id),
            "selected_count": count,
        }
        for card_id, count in card_counts.most_common()
    ]
    return {"action_counts": dict(action_counts), "card_counts": cards}


def _group_records(frame: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    grouped = frame.groupby(keys, dropna=False, observed=True).agg(
        trajectories=("trajectory_id", "size"),
        wins=("won", "sum"),
        win_rate=("won", "mean"),
        mean_turn=("turn", "mean"),
        mean_self_deck_left=("self_deck_left", "mean"),
    ).reset_index()
    grouped["losses"] = grouped["trajectories"] - grouped["wins"]
    return grouped.sort_values(keys).to_dict("records")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    processed = Path(args.processed_dir)
    reports = Path(args.reports_dir)
    manifest = pd.read_csv(processed / "manifest.csv")
    manifest = manifest[
        manifest["usable_manifest"].astype(bool)
        & manifest["rank"].between(args.min_rank, args.max_rank)
    ].copy()
    decisions = pd.read_csv(processed / "decisions.csv")
    selected_decisions = decisions[
        decisions["rank"].between(args.min_rank, args.max_rank)
    ].copy()

    replay_frame = _replay_analysis(manifest)
    action_per_trajectory = (
        selected_decisions.groupby(["trajectory_id", "selected_action_type"], observed=True)
        .size().unstack(fill_value=0)
    )
    trajectory_outcomes = manifest.set_index("trajectory_id")[["target_win"]]
    action_per_trajectory = action_per_trajectory.join(trajectory_outcomes, how="left")
    action_summary: dict[str, Any] = {}
    for outcome_name, outcome_value in (("wins", True), ("losses", False), ("all", None)):
        group = action_per_trajectory
        if outcome_value is not None:
            group = group[group["target_win"].astype(bool).eq(outcome_value)]
        action_summary[outcome_name] = {
            str(column): float(group[column].mean())
            for column in group.columns if column != "target_win"
        }

    candidate_predictions = pd.read_csv(reports / "time_holdout_predictions.csv")
    decision_rank = decisions.set_index("decision_id")["rank"]
    candidate_predictions["teacher_rank"] = candidate_predictions["decision_id"].map(decision_rank)
    candidate_predictions = candidate_predictions[
        candidate_predictions["teacher_rank"].between(args.min_rank, args.max_rank)
    ].copy()
    candidate_metrics = _prediction_metrics(candidate_predictions)

    schema, arrays, indexed = load_matrix_store(processed)
    test_numeric_ids = indexed.loc[
        indexed["decision_id"].astype(str).isin(set(candidate_predictions["decision_id"].astype(str))),
        "decision_numeric_id",
    ].to_numpy(dtype=np.int32)
    rows = _row_indices(
        arrays["decision_index"], np.sort(test_numeric_ids), int(schema["decision_count"])
    )
    baseline_model = joblib.load(args.baseline_model)
    baseline_scores = _score(baseline_model, arrays, rows)
    baseline_schema = json.loads(Path(args.baseline_schema).read_text(encoding="utf-8"))
    baseline_metrics, baseline_predictions = evaluate_arrays(
        arrays,
        indexed,
        rows,
        baseline_scores,
        schema["action_type_map"],
        temperature=float(baseline_schema.get("temperature", 1.0)),
    )

    end_reasons = (
        replay_frame.groupby(["won", "end_reason"], observed=True).size()
        .rename("count").reset_index().to_dict("records")
    )
    matchup = _group_records(replay_frame, ["opponent_archetype"])
    selected_cards = _selected_card_counts(
        processed / "dataset_rows.csv.gz",
        set(selected_decisions["decision_id"].astype(str)),
    )

    submission_rows = manifest.groupby(
        ["rank", "target_team", "submission_id"], observed=True
    ).agg(
        trajectories=("trajectory_id", "size"),
        episodes=("episode_id", "nunique"),
        wins=("target_win", "sum"),
        win_rate=("target_win", "mean"),
        deck_hash=("deck_hash", "first"),
        majkel_distance=("majkel_distance", "first"),
    ).reset_index()
    submission_rows["losses"] = submission_rows["trajectories"] - submission_rows["wins"]

    deck_rows = manifest.groupby("deck_hash", observed=True).agg(
        trajectories=("trajectory_id", "size"),
        teams=("target_team", "nunique"),
        wins=("target_win", "sum"),
        win_rate=("target_win", "mean"),
        majkel_distance=("majkel_distance", "first"),
        major_card_differences_json=("major_card_differences_json", "first"),
        initial_deck_json=("initial_deck_json", "first"),
    ).reset_index()
    deck_rows["losses"] = deck_rows["trajectories"] - deck_rows["wins"]

    result = {
        "rank_range": [args.min_rank, args.max_rank],
        "corpus": {
            "zip_count": int(manifest["zip_name"].nunique()),
            "trajectory_count": int(len(manifest)),
            "episode_count": int(manifest["episode_id"].nunique()),
            "team_count": int(manifest["target_team"].nunique()),
            "submission_count": int(manifest["submission_id"].nunique()),
            "deck_count": int(manifest["deck_hash"].nunique()),
            "wins": int(manifest["target_win"].sum()),
            "losses": int(manifest["target_loss"].sum()),
            "win_rate": float(manifest["target_win"].mean()),
            "decision_count": int(len(selected_decisions)),
        },
        "submissions": submission_rows.sort_values(
            ["rank", "submission_id"]
        ).to_dict("records"),
        "decks": deck_rows.sort_values("majkel_distance").to_dict("records"),
        "end_reasons": end_reasons,
        "matchups": matchup,
        "actions_per_trajectory": action_summary,
        "selected_options": selected_cards,
        "future_holdout_comparison": {
            "definition": "last 20 percent of episodes within each rank-range submission",
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_baseline": _metric_delta(baseline_metrics, candidate_metrics),
        },
        "replay_rows": replay_frame.to_dict("records"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--baseline-schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-rank", type=int, default=21)
    parser.add_argument("--max-rank", type=int, default=40)
    args = parser.parse_args()
    result = analyze(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "corpus": result["corpus"],
        "holdout_delta": result["future_holdout_comparison"]["candidate_minus_baseline"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
