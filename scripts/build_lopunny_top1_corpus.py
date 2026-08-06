"""Build a leakage-audited all-context imitation corpus.

The replay stores an observation at step ``t`` and the action produced from it
on the same seat at step ``t + 1``.  This is the alignment already validated by
the Alakazam v31-v34 pipeline.  Seats are read from ``manifest.csv`` rather than
guessed from team-name order, and every usable replay is checked against both
the expected teacher name and exact 60-card deck before it contributes rows.

Unlike the Alakazam corpus, this dataset keeps every selection context and
supports multiple selected options.  Candidate rows carry binary membership
labels; a separate per-decision matrix is exported for variable pick-count
learning.  Episode-level chronological splits prevent adjacent decisions from
the same game leaking across train, validation, and test.  The optional
``--opponent-deck-hash`` mode learns from the opposite replay seat after exact
header/deck verification, turning saved opponents into sparring teachers.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402


DEFAULT_RUN = (
    ROOT
    / "data"
    / "runs"
    / "leaderboard_top1"
    / "non_alakazam"
    / "20260801_rank01_sub55137818"
)
DEFAULT_AGENT = ROOT / "agents" / "lopunny" / "majkel_lopunny_ml_v1"
EXPECTED_TEACHER = "Majkel1337"
EXPECTED_SUBMISSION = 55137818
EXPECTED_DECK = sorted([
    11, 11, 11, 11,
    13,
    14, 14, 14,
    66, 66, 66, 66,
    174,
    305, 305, 305, 305,
    848, 848, 848, 848,
    849, 849, 849,
    1086, 1086, 1086, 1086,
    1121, 1121, 1121, 1121,
    1122, 1122, 1122, 1122,
    1152, 1152, 1152, 1152,
    1174, 1174, 1174, 1174,
    1182, 1182, 1182,
    1197,
    1225, 1225, 1225, 1225,
    1227, 1227, 1227, 1227,
    1229, 1229, 1229, 1229,
])


def _load_features(agent_dir: Path):
    path = agent_dir / "imitation_features.py"
    spec = importlib.util.spec_from_file_location("lopunny_imitation_features", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _initial_deck(replay: dict[str, Any], seat: int) -> list[int] | None:
    try:
        frames = replay["steps"][0][0].get("visualize") or []
        for frame in frames:
            action = frame.get("action")
            if (
                isinstance(action, list)
                and seat < len(action)
                and isinstance(action[seat], list)
                and len(action[seat]) == 60
            ):
                return sorted(int(card_id) for card_id in action[seat])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def _forced_by_bounds(option_count: int, minimum: int, maximum: int) -> bool:
    minimum = max(0, min(minimum, option_count))
    maximum = max(minimum, min(maximum, option_count))
    return minimum == maximum and minimum in (0, option_count)


def _split_map(episode_ids: list[int], validation_games: int, test_games: int):
    ordered = sorted(set(episode_ids))
    if len(ordered) <= validation_games + test_games:
        raise ValueError(
            f"Need more than {validation_games + test_games} episodes, got {len(ordered)}"
        )
    test = set(ordered[-test_games:])
    validation = set(ordered[-(validation_games + test_games):-test_games])
    return {
        episode_id: (
            "test" if episode_id in test
            else "validation" if episode_id in validation
            else "train"
        )
        for episode_id in ordered
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--teacher", default=EXPECTED_TEACHER)
    parser.add_argument("--submission-id", type=int, default=EXPECTED_SUBMISSION)
    parser.add_argument("--validation-games", type=int, default=40)
    parser.add_argument("--test-games", type=int, default=50)
    parser.add_argument(
        "--opponent-deck-hash",
        help=(
            "Build from the opposite seat of every replay-index row whose "
            "submitted seat used the indexed deck and whose opponent used "
            "this exact deck hash. This turns saved opponents into local "
            "sparring teachers without guessing their seat or team name."
        ),
    )
    parser.add_argument(
        "--source-index",
        type=Path,
        help=(
            "replay_index.csv used with --opponent-deck-hash. Relative "
            "replay_path values are resolved from the index's grandparent."
        ),
    )
    parser.add_argument(
        "--use-episodes-csv",
        action="store_true",
        help=(
            "Derive seats from episodes.csv and use every replay currently "
            "available. Useful while a replay-only incremental pull is active."
        ),
    )
    args = parser.parse_args()

    feature_module = _load_features(args.agent_dir.resolve())
    manifest_path = args.run_dir / "manifest.csv"
    source_manifest = "manifest.csv"
    if args.opponent_deck_hash:
        if args.use_episodes_csv:
            parser.error(
                "--opponent-deck-hash and --use-episodes-csv are mutually exclusive"
            )
        source_index = args.source_index
        if source_index is None:
            source_index = (
                ROOT
                / "data"
                / "kaggle_grimmsnarl_top50"
                / "indexes"
                / "replay_index.csv"
            )
        source_manifest = f"opponent seat from {source_index}"
        with source_index.open(encoding="utf-8-sig", newline="") as handle:
            manifest = list(csv.DictReader(handle))
    elif args.use_episodes_csv:
        source_manifest = "episodes.csv submission-seat derivation"
        episodes_path = args.run_dir / "episodes.csv"
        with episodes_path.open(encoding="utf-8-sig", newline="") as handle:
            episode_rows = list(csv.DictReader(handle))
        manifest = []
        for row in episode_rows:
            seat = ""
            if str(row.get("agent_0_submission_id")) == str(args.submission_id):
                seat = "0"
            elif str(row.get("agent_1_submission_id")) == str(args.submission_id):
                seat = "1"
            manifest.append({
                "submission_id": str(args.submission_id),
                "episode_id": str(row.get("episode_id") or ""),
                "detected_submission_agent_index": seat,
                "error": "",
            })
    else:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            manifest = list(csv.DictReader(handle))

    stats: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    deck_signatures: Counter[str] = Counter()
    trajectories: list[dict[str, Any]] = []
    valid_episode_ids: list[int] = []
    expected_deck = EXPECTED_DECK
    seen_trajectories: set[tuple[str, int]] = set()

    for row in manifest:
        stats["manifest_rows"] += 1
        if args.opponent_deck_hash:
            try:
                indexed_seat = int(row["seat_index"])
                episode_id = int(row["episode_id"])
                replay_path = (
                    source_index.parent.parent
                    / Path(str(row["replay_path"]).replace(chr(92), "/"))
                )
            except (KeyError, TypeError, ValueError):
                stats["invalid_manifest_identity"] += 1
                continue
            if indexed_seat not in (0, 1) or not replay_path.exists():
                stats["missing_replay"] += 1
                continue
            seat = 1 - indexed_seat
            identity = (str(replay_path.resolve()), seat)
            if identity in seen_trajectories:
                stats["duplicate_trajectory"] += 1
                continue
            try:
                header = extract_fast_header_from_file(replay_path)
            except Exception:
                stats["invalid_replay_header"] += 1
                continue
            hashes = header.get("deck_hashes") or ["", ""]
            if len(hashes) < 2 or hashes[seat] != args.opponent_deck_hash:
                stats["opponent_deck_mismatch"] += 1
                continue
            try:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stats["invalid_replay"] += 1
                continue
            deck = _initial_deck(replay, seat)
            if not deck:
                stats["deck_mismatch"] += 1
                continue
            if expected_deck == EXPECTED_DECK:
                # The explicit deck hash is the identity. Its first verified
                # 60-card list becomes the exact-list assertion for every
                # subsequent trajectory in this corpus.
                expected_deck = deck
            signature = " ".join(map(str, deck))
            deck_signatures[signature] += 1
            if deck != expected_deck:
                stats["deck_mismatch"] += 1
                continue
            seen_trajectories.add(identity)
            trajectories.append({
                "episode_id": episode_id,
                "seat": seat,
                "replay": replay,
            })
            valid_episode_ids.append(episode_id)
            stats["validated_trajectories"] += 1
            continue
        if str(row.get("submission_id")) != str(args.submission_id):
            stats["wrong_submission"] += 1
            continue
        if row.get("error"):
            stats["manifest_errors"] += 1
            continue
        try:
            episode_id = int(row["episode_id"])
            seat = int(row["detected_submission_agent_index"])
        except (KeyError, TypeError, ValueError):
            stats["invalid_manifest_identity"] += 1
            continue
        replay_path = (
            args.run_dir
            / "episodes"
            / str(episode_id)
            / "replay"
            / f"episode_{episode_id}.json"
        )
        if not replay_path.exists():
            stats["missing_replay"] += 1
            continue
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["invalid_replay"] += 1
            continue
        teams = (replay.get("info") or {}).get("TeamNames") or []
        if seat not in (0, 1) or seat >= len(teams) or teams[seat] != args.teacher:
            stats["teacher_seat_mismatch"] += 1
            continue
        deck = _initial_deck(replay, seat)
        signature = " ".join(map(str, deck or []))
        deck_signatures[signature] += 1
        if deck != expected_deck:
            stats["deck_mismatch"] += 1
            continue
        trajectories.append({
            "episode_id": episode_id,
            "seat": seat,
            "replay": replay,
        })
        valid_episode_ids.append(episode_id)
        stats["validated_trajectories"] += 1

    split_by_episode = _split_map(
        valid_episode_ids, args.validation_games, args.test_games
    )

    candidate_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    semantic_batches: list[np.ndarray] = []
    groups: list[int] = []
    chosen_counts: list[int] = []
    minimums: list[int] = []
    maximums: list[int] = []
    forced: list[int] = []
    episode_ids: list[int] = []
    seats: list[int] = []
    splits: list[str] = []
    select_types: list[int] = []
    select_contexts: list[int] = []
    won_values: list[int] = []
    count_rows: list[list[float]] = []
    feature_names: list[str] | None = None
    count_feature_names: list[str] | None = None

    for trajectory in sorted(trajectories, key=lambda item: item["episode_id"]):
        replay = trajectory["replay"]
        episode_id = int(trajectory["episode_id"])
        seat = int(trajectory["seat"])
        steps = replay.get("steps") or []
        rewards = replay.get("rewards") or [0, 0]
        won = int(
            seat < len(rewards)
            and 1 - seat < len(rewards)
            and rewards[seat] > rewards[1 - seat]
        )
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                stats["malformed_step"] += 1
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select")
            recorded = (steps[step_index + 1][seat] or {}).get("action")
            if record.get("status") != "ACTIVE" or not isinstance(select, dict):
                continue
            options = list(select.get("option") or [])
            if not options:
                stats["zero_option_decisions"] += 1
                continue
            minimum = max(0, min(int(select.get("minCount") or 0), len(options)))
            maximum = max(
                minimum,
                min(int(select.get("maxCount") or 0), len(options)),
            )
            if (
                not isinstance(recorded, list)
                or any(not isinstance(value, int) for value in recorded)
                or len(set(recorded)) != len(recorded)
                or any(value < 0 or value >= len(options) for value in recorded)
                or not minimum <= len(recorded) <= maximum
            ):
                stats["invalid_recorded_action"] += 1
                continue

            current = observation.get("current") or {}
            base = feature_module.state_features(current)
            base.update(feature_module.observation_features(observation))
            rows: list[list[float]] = []
            semantic: list[list[int]] = []
            for position, option in enumerate(options):
                feature = feature_module.option_features(
                    current,
                    select,
                    option,
                    base_state=base,
                    option_position=position,
                )
                action_name = str(feature.get("action_type") or "other")
                # A stable numeric code is preferable to string/object arrays.
                feature["action_type"] = feature_module.encode_action_type(
                    action_name
                )
                if feature_names is None:
                    feature_names = list(feature.keys())
                    feature_module.assert_no_leakage(feature_names)
                if list(feature.keys()) != feature_names:
                    raise RuntimeError("candidate feature order changed within corpus")
                rows.append([float(feature[name]) for name in feature_names])
                semantic.append([
                    int(round(value))
                    for value in feature_module.semantic_feature_key(feature)
                ])

            count_feature = feature_module.decision_features(observation)
            if count_feature_names is None:
                count_feature_names = list(count_feature.keys())
                feature_module.assert_no_leakage(count_feature_names)
            if list(count_feature.keys()) != count_feature_names:
                raise RuntimeError("count feature order changed within corpus")

            labels = np.zeros(len(options), dtype=np.int8)
            labels[recorded] = 1
            candidate_batches.append(np.asarray(rows, dtype=np.float32))
            label_batches.append(labels)
            semantic_batches.append(np.asarray(semantic, dtype=np.int32))
            groups.append(len(options))
            chosen_counts.append(len(recorded))
            minimums.append(minimum)
            maximums.append(maximum)
            forced_value = _forced_by_bounds(len(options), minimum, maximum)
            forced.append(int(forced_value))
            episode_ids.append(episode_id)
            seats.append(seat)
            splits.append(split_by_episode[episode_id])
            select_type = int(select.get("type", -1))
            select_context = int(select.get("context", -1))
            select_types.append(select_type)
            select_contexts.append(select_context)
            won_values.append(won)
            count_rows.append([
                float(count_feature[name]) for name in count_feature_names
            ])
            context_counts[f"type_{select_type}_context_{select_context}"] += 1
            stats["decisions"] += 1
            stats["candidate_rows"] += len(options)
            stats["forced_decisions"] += int(forced_value)
            stats["nonforced_decisions"] += int(not forced_value)
            stats["multi_pick_decisions"] += int(len(recorded) > 1)
            stats["variable_count_decisions"] += int(minimum < maximum)

    if not candidate_batches or feature_names is None or count_feature_names is None:
        raise RuntimeError("No valid teacher decisions extracted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(candidate_batches, axis=0),
        labels=np.concatenate(label_batches),
        semantics=np.concatenate(semantic_batches, axis=0),
        groups=np.asarray(groups, dtype=np.int32),
        chosen_counts=np.asarray(chosen_counts, dtype=np.int8),
        minimums=np.asarray(minimums, dtype=np.int8),
        maximums=np.asarray(maximums, dtype=np.int8),
        forced=np.asarray(forced, dtype=np.int8),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        seats=np.asarray(seats, dtype=np.int8),
        splits=np.asarray(splits, dtype="U10"),
        select_types=np.asarray(select_types, dtype=np.int8),
        select_contexts=np.asarray(select_contexts, dtype=np.int8),
        won=np.asarray(won_values, dtype=np.int8),
        count_features=np.asarray(count_rows, dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype="U96"),
        count_feature_names=np.asarray(count_feature_names, dtype="U96"),
    )

    split_counts = Counter(splits)
    split_episodes = {
        split: len({
            episode_id for episode_id, value in zip(episode_ids, splits)
            if value == split
        })
        for split in ("train", "validation", "test")
    }
    card_names: dict[int, str] = {}
    cards_path = ROOT / "vendor" / "cg" / "cards.json"
    if cards_path.exists():
        for card in json.loads(cards_path.read_text(encoding="utf-8")):
            card_names[int(card["cardId"])] = str(card["name"])
    deck_counts = Counter(expected_deck)
    report = {
        "run_dir": (
            None if args.opponent_deck_hash else str(args.run_dir.resolve())
        ),
        "source_index": (
            str(source_index.resolve()) if args.opponent_deck_hash else None
        ),
        "output": str(args.output.resolve()),
        "teacher": (
            f"opponent deck {args.opponent_deck_hash}"
            if args.opponent_deck_hash else args.teacher
        ),
        "submission_id": None if args.opponent_deck_hash else args.submission_id,
        "opponent_deck_hash": args.opponent_deck_hash,
        "source_manifest": source_manifest,
        "seat_verification": (
            "opposite of indexed submitted seat + replay header exact deck hash"
            if args.opponent_deck_hash
            else "manifest detected_submission_agent_index + replay TeamNames"
        ),
        "deck_verified_trajectories": stats["validated_trajectories"],
        "deck": [
            {"card_id": card_id, "name": card_names.get(card_id, ""), "count": count}
            for card_id, count in sorted(deck_counts.items())
        ],
        "stats": dict(stats),
        "split_episodes": split_episodes,
        "split_decisions": dict(split_counts),
        "split_policy": {
            "order": "ascending episode_id",
            "validation_games": args.validation_games,
            "test_games": args.test_games,
        },
        "context_counts": dict(context_counts.most_common()),
        "feature_count": len(feature_names),
        "count_feature_count": len(count_feature_names),
        "deck_signature_count": len(deck_signatures),
        "deck_mismatch_count": stats["deck_mismatch"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
