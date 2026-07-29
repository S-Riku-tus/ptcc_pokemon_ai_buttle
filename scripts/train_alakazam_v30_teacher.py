"""Train the v30 coherent-teacher ranker from the expanded Alakazam corpus.

v30 learns MAIN decisions from 2,361 deduplicated trajectories.  It adds
public action-history and selection-context features, uses v29 itself as the
residual baseline, collapses interchangeable options before training, and
uses a chronological episode split within each teacher cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent
from ml.core.distill import compact_booster


ACTION_TYPES = (
    "ability",
    "attack",
    "bench",
    "boss",
    "end",
    "energy",
    "evolve",
    "hammer",
    "other",
    "retreat",
    "trainer",
    "xerosic",
)
ACTION_TYPE_MAP = {name: index for index, name in enumerate(ACTION_TYPES)}
BASE_CATEGORICAL = {
    "action_type",
    "option_type",
    "candidate_card_id",
    "candidate_attack_id",
    "candidate_target_id",
    "self_active_id",
    "opp_active_id",
    "stadium_id",
    "fallback_action_type",
    "fallback_card_id",
}


def _tree_score(features: list[float], model: dict[str, Any]) -> float:
    total = 0.0
    for tree in model["trees"]:
        node = tree
        while "v" not in node:
            value = features[node["f"]]
            if value != value:
                go_left = node.get("x", True)
            elif node.get("d", "<=") == "==":
                go_left = int(round(value)) in node.get("c", [])
            else:
                go_left = value <= node["t"]
            node = node["l"] if go_left else node["r"]
        total += node["v"]
    if model.get("average_output") and model["trees"]:
        total /= len(model["trees"])
    return total


def _semantic_key(feature: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(feature.get("option_type", -1)),
        int(feature.get("candidate_card_id", -1)),
        int(feature.get("candidate_attack_id", -1)),
        int(feature.get("candidate_target_id", -1)),
        int(feature.get("candidate_target_hp", -1)),
        int(feature.get("candidate_target_max_hp", -1)),
        int(feature.get("candidate_target_energy", -1)),
        int(feature.get("candidate_target_special_energy", -1)),
        int(feature.get("candidate_inplay_area", -1)),
    )


def _safe_policy_scores(
    policy_module: Any,
    observation: dict[str, Any],
    option_count: int,
) -> list[float]:
    try:
        parsed = policy_module.to_observation_class(observation)
        policy = policy_module.AlakazamPolicy(parsed)
        scores = [float(policy._score(option)) for option in parsed.select.option]
        if len(scores) == option_count:
            return scores
    except Exception:
        pass
    return [0.0] * option_count


def _rank_positions(scores: list[float]) -> tuple[list[int], list[int]]:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positions = [0] * len(scores)
    for position, index in enumerate(order):
        positions[index] = position
    return order, positions


def _read_replay(
    row: dict[str, str],
    archives: dict[str, zipfile.ZipFile],
) -> dict[str, Any]:
    storage_path = row["storage_path"]
    if row["storage_type"] == "zip":
        archive = archives.get(storage_path)
        if archive is None:
            archive = zipfile.ZipFile(storage_path)
            archives[storage_path] = archive
        return json.loads(archive.read(row["replay_path"]))
    return json.loads(
        (Path(storage_path) / row["replay_path"]).read_text(
            encoding="utf-8",
        )
    )


def _v29_context(feature: dict[str, Any]) -> dict[str, Any]:
    action_index = int(feature.get("action_type", ACTION_TYPE_MAP["other"]))
    action_name = (
        ACTION_TYPES[action_index]
        if 0 <= action_index < len(ACTION_TYPES)
        else "other"
    )
    return {
        "action_type": action_name,
        "card_id": int(feature.get("candidate_card_id", -1)),
        "target_id": int(feature.get("candidate_target_id", -1)),
        "breaks_current_ko": bool(
            feature.get("breaks_current_ko_estimate", 0)
        ),
        "attack_lethal": bool(
            feature.get("attack_lethal_estimate", 0)
        ),
    }


def _v29_baseline_index(
    features: list[dict[str, Any]],
    scores: list[float],
    deterministic_index: int,
    model: dict[str, Any],
) -> int:
    contexts = [_v29_context(feature) for feature in features]
    fallback_context = contexts[deterministic_index]
    if fallback_context["attack_lethal"]:
        return deterministic_index
    order = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )
    probabilities = _probabilities(
        np.asarray(scores, dtype=np.float32),
        float(model.get("temperature", 1.0)),
    )
    top = order[0]
    confidence = float(probabilities[top])
    second = float(probabilities[order[1]]) if len(order) > 1 else 0.0
    if (
        confidence
        < max(float(model.get("fallback_probability", 0.20)), 0.20)
        or confidence - second < float(model.get("fallback_margin", 0.0))
    ):
        return deterministic_index
    runtime_module = sys.modules["v29_runtime"]
    safety_reason = runtime_module._candidate_safety_reason(
        contexts[top],
        fallback_context,
        features[top],
        attack_is_available=any(
            context["action_type"] == "attack"
            for context in contexts
        ),
    )
    return deterministic_index if safety_reason is not None else top


def _extract_chunk(
    agent_dir: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    agent_path = Path(agent_dir)
    _, _, main_module = load_dir_agent(agent_path)
    deterministic_agent = main_module._fallback_agent
    policy_module = main_module.fallback_policy
    features_module = sys.modules["ml_features"]
    legacy_model = json.loads(
        (agent_path / "legacy_ranker_model.json").read_text(encoding="utf-8")
    )
    v29_model = json.loads(
        (agent_path / "v29_ranker_model.json").read_text(encoding="utf-8")
    )

    feature_batches: list[np.ndarray] = []
    labels: list[int] = []
    row_weights: list[float] = []
    groups: list[int] = []
    splits: list[str] = []
    fallback_correct: list[int] = []
    teacher_action_types: list[int] = []
    episode_ids: list[int] = []
    ranks: list[int] = []
    feature_names: list[str] | None = None
    stats: Counter[str] = Counter()
    archives: dict[str, zipfile.ZipFile] = {}

    for row in rows:
        deterministic_agent({"select": None})
        replay = _read_replay(row, archives)
        seat = int(row["seat_index"])
        rank = int(row["leaderboard_rank"])
        episode_id = int(row["episode_id"])
        steps = replay.get("steps") or []
        final = steps[-1] if steps else []
        own_reward = (
            final[seat].get("reward") if seat < len(final) else None
        )
        other_reward = (
            final[1 - seat].get("reward") if 1 - seat < len(final) else None
        )
        won = (
            own_reward is not None
            and other_reward is not None
            and own_reward > other_reward
        )
        outcome_weight = 1.03 if won else 0.98
        cohort_weight = {
            "current_top": 1.25,
            "majkel_full": 1.15,
            "yushin_full": 0.90,
        }.get(row["source_cohort"], 1.0)
        decision_weight = (
            float(row["teacher_priority"])
            * outcome_weight
            * cohort_weight
        )

        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            recorded = (steps[step_index + 1][seat] or {}).get("action")
            if (
                record.get("status") != "ACTIVE"
                or int(select.get("context", -1)) != 0
                or int(select.get("minCount") or 0) != 1
                or int(select.get("maxCount") or 0) != 1
                or len(options) < 2
                or not isinstance(recorded, list)
                or len(recorded) != 1
                or not isinstance(recorded[0], int)
                or not 0 <= recorded[0] < len(options)
            ):
                continue

            try:
                deterministic = list(deterministic_agent(observation))
            except Exception:
                stats["fallback_error"] += 1
                continue
            if (
                len(deterministic) != 1
                or not 0 <= deterministic[0] < len(options)
            ):
                stats["fallback_unresolved"] += 1
                continue

            current = observation.get("current") or {}
            base_state = features_module.state_features(current)
            base_state.update(
                features_module.observation_features(observation)
            )
            raw_features: list[dict[str, Any]] = []
            for option in options:
                feature = dict(
                    features_module.option_features(
                        current,
                        select,
                        option,
                        base_state=base_state,
                    )
                )
                feature["action_type"] = ACTION_TYPE_MAP.get(
                    str(feature.get("action_type") or "other"),
                    ACTION_TYPE_MAP["other"],
                )
                raw_features.append(feature)

            teacher_key = _semantic_key(raw_features[recorded[0]])
            fallback_key = _semantic_key(raw_features[deterministic[0]])
            policy_scores = _safe_policy_scores(
                policy_module, observation, len(options)
            )
            policy_order, policy_positions = _rank_positions(policy_scores)
            policy_peak = policy_scores[policy_order[0]]

            legacy_scores = []
            for feature in raw_features:
                legacy_row = [
                    float(feature.get(name, -1))
                    for name in legacy_model["feature_names"]
                ]
                legacy_scores.append(_tree_score(legacy_row, legacy_model))
            legacy_order, legacy_positions = _rank_positions(legacy_scores)
            legacy_peak = legacy_scores[legacy_order[0]]
            fallback_action_type = int(
                raw_features[deterministic[0]]["action_type"]
            )
            fallback_card_id = int(
                raw_features[deterministic[0]].get("candidate_card_id", -1)
            )
            fallback_legacy_agree = int(
                deterministic[0] == legacy_order[0]
            )

            # First reconstruct the exact residual features consumed by v29.
            for index, feature in enumerate(raw_features):
                feature.update({
                    "fallback_selected": int(
                        _semantic_key(feature) == fallback_key
                    ),
                    "fallback_action_type": fallback_action_type,
                    "fallback_card_id": fallback_card_id,
                    "fallback_policy_score": max(
                        -10_000_000.0, min(10_000_000.0, policy_scores[index])
                    ),
                    "fallback_policy_score_gap": max(
                        -10_000_000.0,
                        min(10_000_000.0, policy_scores[index] - policy_peak),
                    ),
                    "fallback_policy_rank": policy_positions[index],
                    "legacy_ranker_score": legacy_scores[index],
                    "legacy_ranker_score_gap": (
                        legacy_scores[index] - legacy_peak
                    ),
                    "legacy_ranker_rank": legacy_positions[index],
                    "legacy_ranker_selected": int(index == legacy_order[0]),
                    "fallback_legacy_agree": fallback_legacy_agree,
                })

            v29_scores = []
            for feature in raw_features:
                v29_row = [
                    float(feature.get(name, -1))
                    for name in v29_model["feature_names"]
                ]
                v29_scores.append(_tree_score(v29_row, v29_model))
            v29_order, v29_positions = _rank_positions(v29_scores)
            v29_peak = v29_scores[v29_order[0]]
            baseline_index = _v29_baseline_index(
                raw_features,
                v29_scores,
                deterministic[0],
                v29_model,
            )
            baseline_key = _semantic_key(
                raw_features[baseline_index]
            )

            # Interchangeable copies are one ranking candidate in v30.
            representatives: list[int] = []
            seen_keys: set[tuple[Any, ...]] = set()
            for index, feature in enumerate(raw_features):
                key = _semantic_key(feature)
                if key in seen_keys:
                    stats["duplicate_candidates_collapsed"] += 1
                    continue
                seen_keys.add(key)
                representatives.append(index)

            decision_rows: list[list[float]] = []
            decision_labels: list[int] = []
            for index in representatives:
                feature = raw_features[index]
                feature.update({
                    "v29_selected": int(
                        _semantic_key(feature) == baseline_key
                    ),
                    "v29_ranker_score": v29_scores[index],
                    "v29_ranker_score_gap": (
                        v29_scores[index] - v29_peak
                    ),
                    "v29_ranker_rank": v29_positions[index],
                    "v29_ranker_raw_selected": int(index == v29_order[0]),
                    "v29_deterministic_agree": int(
                        baseline_key == fallback_key
                    ),
                })
                if feature_names is None:
                    feature_names = list(feature.keys())
                if list(feature.keys()) != feature_names:
                    raise RuntimeError("v30 feature order changed within corpus")
                decision_rows.append(
                    [float(feature[name]) for name in feature_names]
                )
                decision_labels.append(
                    int(_semantic_key(feature) == teacher_key)
                )

            if not any(decision_labels):
                stats["label_unresolved"] += 1
                continue
            feature_batches.append(
                np.asarray(decision_rows, dtype=np.float32)
            )
            labels.extend(decision_labels)
            row_weights.extend(
                [decision_weight] * len(representatives)
            )
            groups.append(len(representatives))
            splits.append(row["split"])
            fallback_correct.append(int(baseline_key == teacher_key))
            teacher_action_types.append(
                int(raw_features[recorded[0]]["action_type"])
            )
            episode_ids.append(episode_id)
            ranks.append(rank)
            stats["decisions"] += 1
            stats["candidate_rows"] += len(representatives)
            stats[f"rank_{rank}_decisions"] += 1
            stats[f"cohort_{row['source_cohort']}_decisions"] += 1
            stats["wins"] += int(won)

    if feature_names is None:
        raise RuntimeError("No teacher decisions extracted")
    for archive in archives.values():
        archive.close()
    return {
        "features": np.concatenate(feature_batches, axis=0),
        "labels": np.asarray(labels, dtype=np.int8),
        "weights": np.asarray(row_weights, dtype=np.float32),
        "groups": np.asarray(groups, dtype=np.int32),
        "splits": splits,
        "fallback_correct": np.asarray(fallback_correct, dtype=np.int8),
        "teacher_action_types": np.asarray(teacher_action_types, dtype=np.int8),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "ranks": np.asarray(ranks, dtype=np.int16),
        "feature_names": feature_names,
        "stats": dict(stats),
    }


def _decision_row_ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    return starts, ends


def _select_decisions(
    arrays: dict[str, Any],
    decision_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    starts, ends = _decision_row_ranges(arrays["groups"])
    row_indices = np.concatenate([
        np.arange(starts[index], ends[index], dtype=np.int64)
        for index in decision_indices
    ])
    return (
        arrays["features"][row_indices],
        arrays["labels"][row_indices],
        arrays["weights"][row_indices],
        arrays["groups"][decision_indices].astype(int).tolist(),
    )


def _group_ranges(groups: list[int]) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(np.asarray(groups, dtype=np.int64))
    return np.r_[0, ends[:-1]], ends


def _probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores.astype(float) / max(temperature, 1e-6)
    scaled -= scaled.max()
    values = np.exp(np.clip(scaled, -50, 50))
    return values / max(values.sum(), 1e-12)


def _fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: list[int],
) -> float:
    starts, ends = _group_ranges(groups)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in np.geomspace(0.2, 4.0, 41):
        losses = []
        for start, end in zip(starts, ends):
            probabilities = _probabilities(scores[start:end], temperature)
            positive = labels[start:end] == 1
            losses.append(
                -math.log(max(float(probabilities[positive].sum()), 1e-12))
            )
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def _evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: list[int],
    fallback_correct: np.ndarray,
    teacher_actions: np.ndarray,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    starts, ends = _group_ranges(groups)
    top1 = top2 = top3 = oracle = 0
    confidences = []
    margins = []
    model_correct_values = []
    action_stats: dict[int, Counter[str]] = {}
    for decision, (start, end) in enumerate(zip(starts, ends)):
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        order = np.argsort(-group_scores, kind="stable")
        model_correct = bool(group_labels[order[0]] == 1)
        top1 += int(model_correct)
        top2 += int(bool(np.any(group_labels[order[:2]] == 1)))
        top3 += int(bool(np.any(group_labels[order[:3]] == 1)))
        oracle += int(model_correct or bool(fallback_correct[decision]))
        probabilities = _probabilities(group_scores, temperature)
        confidence = float(probabilities[order[0]])
        second = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        confidences.append(confidence)
        margins.append(confidence - second)
        model_correct_values.append(model_correct)
        action = int(teacher_actions[decision])
        stats = action_stats.setdefault(action, Counter())
        stats["count"] += 1
        stats["correct"] += int(model_correct)

    count = len(groups)
    fallback_rate = float(np.mean(fallback_correct)) if count else 0.0
    metrics = {
        "decisions": count,
        "semantic_top1": top1 / count if count else 0.0,
        "semantic_top2": top2 / count if count else 0.0,
        "semantic_top3": top3 / count if count else 0.0,
        "fallback_semantic": fallback_rate,
        "fallback_or_model_oracle": oracle / count if count else 0.0,
        "temperature": temperature,
        "by_teacher_action": {
            ACTION_TYPES[action]: {
                "count": int(stats["count"]),
                "semantic_top1": stats["correct"] / stats["count"],
            }
            for action, stats in sorted(action_stats.items())
        },
    }
    gates = []
    model_correct_array = np.asarray(model_correct_values, dtype=bool)
    confidence_array = np.asarray(confidences)
    margin_array = np.asarray(margins)
    fallback_array = fallback_correct.astype(bool)
    for probability_threshold in np.arange(0.0, 0.81, 0.05):
        for margin_threshold in np.arange(0.0, 0.31, 0.05):
            selected = (
                (confidence_array >= probability_threshold)
                & (margin_array >= margin_threshold)
            )
            correct = np.where(selected, model_correct_array, fallback_array)
            gates.append({
                "probability": round(float(probability_threshold), 2),
                "margin": round(float(margin_threshold), 2),
                "semantic": float(correct.mean()),
                "model_selection_rate": float(selected.mean()),
            })
    gates.sort(
        key=lambda row: (row["semantic"], -row["model_selection_rate"]),
        reverse=True,
    )
    metrics["best_gates"] = gates[:15]
    return metrics, gates


def _make_model(n_estimators: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=741,
        n_jobs=4,
        verbosity=-1,
    )


def _fit(
    arrays: dict[str, Any],
    feature_names: list[str],
    decision_indices: np.ndarray,
    *,
    n_estimators: int,
    validation_indices: np.ndarray | None = None,
) -> lgb.LGBMRanker:
    x, y, weights, groups = _select_decisions(arrays, decision_indices)
    categorical = [
        index
        for index, name in enumerate(feature_names)
        if name in BASE_CATEGORICAL or name.endswith("_id")
    ]
    model = _make_model(n_estimators)
    kwargs: dict[str, Any] = {
        "X": x,
        "y": y,
        "group": groups,
        "sample_weight": weights,
        "feature_name": feature_names,
        "categorical_feature": categorical,
    }
    if validation_indices is not None and len(validation_indices):
        vx, vy, _, vgroups = _select_decisions(arrays, validation_indices)
        kwargs.update({
            "eval_set": [(vx, vy)],
            "eval_group": [vgroups],
            "callbacks": [lgb.early_stopping(45, verbose=False)],
        })
    model.fit(**kwargs)
    return model


def _predict_for_decisions(
    model: lgb.LGBMRanker,
    arrays: dict[str, Any],
    decision_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    x, y, _, groups = _select_decisions(arrays, decision_indices)
    scores = model.predict(x).astype(np.float32)
    return scores, y, groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--n-estimators", type=int, default=650)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.teacher_index.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        replay_rows = list(csv.DictReader(handle))
    if (
        args.max_trajectories is not None
        and len(replay_rows) > args.max_trajectories
    ):
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in replay_rows:
            grouped.setdefault(row["source_cohort"], []).append(row)
        limited: list[dict[str, str]] = []
        while (
            len(limited) < args.max_trajectories
            and any(grouped.values())
        ):
            for cohort in sorted(grouped):
                if grouped[cohort]:
                    limited.append(grouped[cohort].pop(0))
                    if len(limited) >= args.max_trajectories:
                        break
        replay_rows = limited
    if not replay_rows:
        raise RuntimeError("No matching teacher replays")

    by_cohort: dict[str, list[int]] = {}
    for row in replay_rows:
        by_cohort.setdefault(row["source_cohort"], []).append(
            int(row["episode_id"])
        )
    split_by_episode: dict[int, str] = {}
    for episodes in by_cohort.values():
        ordered = sorted(set(episodes))
        train_end = max(1, int(len(ordered) * 0.70))
        validation_end = max(
            train_end + 1,
            int(len(ordered) * 0.80),
        )
        for index, episode_id in enumerate(ordered):
            split_by_episode[episode_id] = (
                "train"
                if index < train_end
                else "validation"
                if index < validation_end
                else "test"
            )
    for row in replay_rows:
        row["split"] = split_by_episode[int(row["episode_id"])]

    worker_count = min(max(1, args.workers), len(replay_rows))
    chunks = [
        replay_rows[index::worker_count]
        for index in range(worker_count)
    ]
    if worker_count == 1:
        parts = [
            _extract_chunk(
                str(args.agent_dir.resolve()),
                chunks[0],
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            parts = list(
                executor.map(
                    _extract_chunk,
                    [str(args.agent_dir.resolve())] * worker_count,
                    chunks,
                )
            )

    feature_names = parts[0]["feature_names"]
    if any(part["feature_names"] != feature_names for part in parts):
        raise RuntimeError("Worker feature schemas differ")
    arrays: dict[str, Any] = {
        key: np.concatenate([part[key] for part in parts])
        for key in (
            "features",
            "labels",
            "weights",
            "groups",
            "fallback_correct",
            "teacher_action_types",
            "episode_ids",
            "ranks",
        )
    }
    arrays["splits"] = sum((part["splits"] for part in parts), [])
    split_values = np.asarray(arrays["splits"])
    train_indices = np.flatnonzero(split_values == "train")
    validation_indices = np.flatnonzero(split_values == "validation")
    test_indices = np.flatnonzero(split_values == "test")

    model = _fit(
        arrays,
        feature_names,
        train_indices,
        n_estimators=args.n_estimators,
        validation_indices=validation_indices,
    )
    best_iteration = int(model.best_iteration_ or args.n_estimators)
    validation_scores, validation_labels, validation_groups = (
        _predict_for_decisions(model, arrays, validation_indices)
    )
    temperature = _fit_temperature(
        validation_scores, validation_labels, validation_groups
    )
    validation_metrics, validation_gates = _evaluate(
        validation_scores,
        validation_labels,
        validation_groups,
        arrays["fallback_correct"][validation_indices],
        arrays["teacher_action_types"][validation_indices],
        temperature,
    )
    best_gate = validation_gates[0]

    test_scores, test_labels, test_groups = _predict_for_decisions(
        model, arrays, test_indices
    )
    test_metrics, _ = _evaluate(
        test_scores,
        test_labels,
        test_groups,
        arrays["fallback_correct"][test_indices],
        arrays["teacher_action_types"][test_indices],
        temperature,
    )
    # Evaluate the gate selected on validation without retuning on test.
    starts, ends = _group_ranges(test_groups)
    gated_correct = []
    gated_selected = []
    for local_index, (start, end) in enumerate(zip(starts, ends)):
        order = np.argsort(-test_scores[start:end], kind="stable")
        probabilities = _probabilities(test_scores[start:end], temperature)
        confidence = float(probabilities[order[0]])
        second = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        selected = (
            confidence >= float(best_gate["probability"])
            and confidence - second >= float(best_gate["margin"])
        )
        model_correct = bool(test_labels[start:end][order[0]] == 1)
        gated_selected.append(selected)
        gated_correct.append(
            model_correct
            if selected
            else bool(arrays["fallback_correct"][test_indices[local_index]])
        )
    test_metrics["validation_selected_gate"] = {
        **best_gate,
        "test_semantic": float(np.mean(gated_correct)),
        "test_model_selection_rate": float(np.mean(gated_selected)),
    }

    # Retrain the submission artifact on the entire frozen corpus after the
    # episode-held-out estimate has been recorded.
    all_indices = np.arange(len(arrays["groups"]), dtype=np.int64)
    final_model = _fit(
        arrays,
        feature_names,
        all_indices,
        n_estimators=best_iteration,
    )
    compact = compact_booster(final_model.booster_, "ranker")
    compact.update({
        "temperature": temperature,
        "fallback_probability": float(best_gate["probability"]),
        "fallback_margin": float(best_gate["margin"]),
        "action_type_map": ACTION_TYPE_MAP,
        "legal_option_only": True,
        "runtime_scope": "v30_expanded_coherent_teacher_main_policy",
        "training_decisions": int(len(arrays["groups"])),
        "training_candidate_rows": int(len(arrays["labels"])),
        "teacher_ranks": sorted({
            int(row["leaderboard_rank"])
            for row in replay_rows
        }),
        "teacher_trajectories": len(replay_rows),
        "teacher_cohorts": dict(Counter(
            row["source_cohort"] for row in replay_rows
        )),
        "baseline": "v29_runtime_choice_and_raw_ranker_score",
    })
    model_path = args.agent_dir / "ranker_model.json"
    model_path.write_text(
        json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    extraction_stats: Counter[str] = Counter()
    for part in parts:
        extraction_stats.update(part["stats"])
    report = {
        "agent_dir": str(args.agent_dir.resolve()),
        "teacher_index": str(args.teacher_index.resolve()),
        "deck_hash": "cc38cb450b86770a",
        "teacher_ranks": sorted({
            int(row["leaderboard_rank"])
            for row in replay_rows
        }),
        "teacher_cohorts": dict(Counter(
            row["source_cohort"] for row in replay_rows
        )),
        "teacher_games": len(replay_rows),
        "features": len(feature_names),
        "feature_names": feature_names,
        "best_iteration": best_iteration,
        "split_decisions": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "extraction_stats": dict(extraction_stats),
        "validation": validation_metrics,
        "test": test_metrics,
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "label_definition": (
            "one representative per card/attack/target semantic option"
        ),
        "rank_weight": (
            "teacher_priority * cohort_weight * slight outcome_weight"
        ),
        "split_definition": (
            "chronological 70/10/20 by episode id within each cohort"
        ),
        "final_fit": "all frozen teacher decisions after held-out evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "model": str(model_path),
        "teacher_games": report["teacher_games"],
        "decisions": extraction_stats["decisions"],
        "best_iteration": best_iteration,
        "validation_top1": validation_metrics["semantic_top1"],
        "test_top1": test_metrics["semantic_top1"],
        "test_top3": test_metrics["semantic_top3"],
        "test_fallback": test_metrics["fallback_semantic"],
        "test_gated": test_metrics["validation_selected_gate"]["test_semantic"],
        "model_bytes": report["model_bytes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
