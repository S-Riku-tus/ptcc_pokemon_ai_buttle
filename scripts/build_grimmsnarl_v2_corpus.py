"""Extract the v2 Grimmsnarl imitation corpus: every scorable select context.

v1 extracted MAIN only. Measured against the pinned teacher, MAIN agreement
was 90.5% while the rule policy that still owned the other contexts agreed
39.5% on deck search (about 8 decisions per game), 64.5% on Adrena-Brain
damage placement and 50.0% on counter removal. All-context agreement was
81.4%. Those contexts decide what the deck draws and what it kills, so v2
trains on them too.

A decision is extracted when the context is in ``SCORABLE_CONTEXTS``, the
teacher picked exactly one option, and at least two options are
semantically distinct. Optional selects (``minCount == 0``) are included
because the teachers never once declined one in 3,655 games, so there is no
decline branch to model.

Original v1 notes follow.

Extract the Grimmsnarl imitation corpus from the top-50 replay archive.

Differences from the Alakazam v31-v36 extractor, all deliberate:

* No residual chain. The Alakazam corpus fed a rule policy's score, a legacy
  ranker's score and a v29 ranker's score in as features, so every retrain
  inherited the previous generation's mistakes and the deployed agent needed
  a rule shell to run at all. Here the ranker sees only the observation.
* Multi-teacher by default. ``analyze_grimmsnarl_teacher_corpus.py`` shows the
  21 same-deck top-50 teams behave as one policy, so all of them are teachers
  and ``team_id`` is carried per decision for leave-one-team-out evaluation.
* Chronological split on episode id, which is monotonic in wall-clock time,
  with team-disjoint splits available for the transfer check.

Intra-turn history columns are reproduced from the v33 design: they describe
what the acting player has already been offered and passed over this turn.
They are computed from the agent's own decision stream, so the runtime can
rebuild them exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN_CONTEXT = 0
DEFAULT_AGENT_DIR = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v2"

# Candidate identity used to collapse interchangeable copies and to key the
# intra-turn history. The runtime can compute the same key from raw options.
# Candidate identity. The four ctx_* columns matter outside MAIN: a
# REMOVE_DAMAGE_COUNTER_COUNT select offers {number: 1|2|3} options that carry
# no card at all, so without ctx_number they all collapse into one candidate
# and the decision is dropped as having a single distinct option.
SEMANTIC = (
    "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "candidate_inplay_area",
    "candidate_target_hp", "candidate_target_energy",
    "ctx_card_id", "ctx_area", "ctx_owner_is_self", "ctx_number",
)
TURN_FEATURES = (
    "turn_decision_index",
    "turn_candidate_offer_count",
    "turn_candidate_passed_over",
    "turn_candidate_offered_previous",
    "turn_candidate_first_offer_index",
    "turn_class_passed_over",
    "turn_class_offer_count",
    "turn_new_candidate",
)
CATEGORICAL = {
    "action_type_id", "option_type", "candidate_card_id",
    "candidate_attack_id", "candidate_target_id", "candidate_area",
    "candidate_inplay_area", "self_active_id", "opp_active_id", "stadium_id",
    "select_type", "select_context",
}
LEGACY_DOWNLOAD_STATUSES = {"success"}
VALIDATED_CACHE_STATUS = "skipped_existing"


def select_training_index(
    index: pd.DataFrame,
    *,
    deck_hash_value: str,
    excluded_teams: set[int],
    accepted_download_statuses: set[str],
    limit_per_team: int = 0,
    latest_per_team: int = 0,
) -> pd.DataFrame:
    """Return the exact replay relations eligible for corpus extraction.

    Defaults intentionally reproduce the v4 selection. New corpus versions
    can opt into validated cache hits and a newest-per-team cap without
    silently changing any existing model or processed corpus.
    """

    if limit_per_team and latest_per_team:
        raise ValueError("--limit-per-team and --latest-per-team are exclusive")

    required = {
        "download_status",
        "deck_hash",
        "team_id",
        "submission_id",
        "episode_id",
        "seat_index",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"training index is missing columns: {missing}")

    selected = index[
        index["download_status"].isin(sorted(accepted_download_statuses))
    ].copy()
    if deck_hash_value:
        selected = selected[selected["deck_hash"] == deck_hash_value]
    if excluded_teams:
        selected = selected[~selected["team_id"].isin(excluded_teams)]

    selected = selected.drop_duplicates(subset=["episode_id", "seat_index"])
    selected = selected.sort_values(
        ["episode_id", "seat_index", "submission_id"],
        kind="stable",
    )
    if limit_per_team:
        # Preserve the legacy meaning for exact reproduction of old commands.
        selected = selected.groupby("team_id", group_keys=False).head(
            limit_per_team
        )
    elif latest_per_team:
        selected = selected.groupby("team_id", group_keys=False).tail(
            latest_per_team
        )
    return selected.sort_values(
        ["episode_id", "seat_index", "submission_id"],
        kind="stable",
    ).reset_index(drop=True)


def write_selection_manifest(index: pd.DataFrame, path: Path) -> str:
    """Persist the exact selected replay relations and return their SHA-256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False, encoding="utf-8-sig")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_features(agent_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "grimmsnarl_ml_features", agent_dir / "ml_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _semantic_key(row: dict[str, Any]) -> tuple:
    return tuple(int(row.get(name, -1)) for name in SEMANTIC)


def _extract_chunk(payload: tuple[str, str, list[dict[str, Any]]]) -> dict[str, Any]:
    agent_dir, replay_root, rows = payload
    features_module = _load_features(Path(agent_dir))
    scorable = set(getattr(features_module, "SCORABLE_CONTEXTS", {MAIN_CONTEXT}))
    action_map = {
        name: index
        for index, name in enumerate(features_module.ACTION_TYPES)
    }

    feature_batches: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    episode_ids: list[int] = []
    team_ids: list[int] = []
    submission_ids: list[int] = []
    seats: list[int] = []
    turns: list[int] = []
    contexts: list[int] = []
    won_flags: list[int] = []
    teacher_action_types: list[int] = []
    feature_names: list[str] | None = None
    stats: Counter[str] = Counter()

    for row in rows:
        # Newer fetch_submission_logs runs keep each replay inside its own
        # episode directory instead of flattening everything under
        # ``data_root/replays``.  A frozen selection manifest may therefore
        # carry the exact replay path.  Preserve the legacy fallback so every
        # historical corpus command remains reproducible.
        replay_value = str(row.get("replay_path") or "").strip()
        if replay_value:
            candidate = Path(replay_value)
            path = (
                candidate if candidate.is_absolute()
                # Index paths are rooted at data_root (normally
                # ``replays/episode_*.json``), while the legacy fallback
                # below receives data_root/replays directly.
                else Path(replay_root).parent / candidate
            )
        else:
            path = Path(replay_root) / f"episode_{row['episode_id']}.json"
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats["replay_unreadable"] += 1
            continue
        seat = int(row["seat_index"])
        steps = replay.get("steps") or []
        final = steps[-1] if steps else []
        own = final[seat].get("reward") if seat < len(final) else None
        other = final[1 - seat].get("reward") if 1 - seat < len(final) else None
        won = int(own is not None and other is not None and own > other)
        stats["episodes"] += 1
        stats["wins"] += won

        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            action = (steps[index + 1][seat] or {}).get("action")
            context = int(select.get("context", -1))
            if (
                record.get("status") != "ACTIVE"
                or context not in scorable
                or int(select.get("minCount") or 0) > 1
                or int(select.get("maxCount") or 0) != 1
                or len(options) < 2
                or not isinstance(action, list)
                or len(action) != 1
                or not isinstance(action[0], int)
                or not 0 <= action[0] < len(options)
            ):
                continue

            current = observation.get("current") or {}
            base_state = features_module.state_features(current)
            base_state.update(
                features_module.observation_features(observation)
            )
            raw: list[dict[str, Any]] = []
            for position, option in enumerate(options):
                feature = dict(features_module.option_features(
                    current, select, option,
                    base_state=base_state,
                    option_position=position,
                ))
                feature["action_type_id"] = action_map.get(
                    str(feature.pop("action_type", "other")),
                    action_map["other"],
                )
                raw.append(feature)

            chosen_key = _semantic_key(raw[action[0]])

            # Interchangeable copies are one ranking candidate. Without this
            # the label is arbitrary among identical options and Top-1 is
            # unfairly penalised for picking the other copy.
            representatives: list[int] = []
            seen: set[tuple] = set()
            for position, feature in enumerate(raw):
                key = _semantic_key(feature)
                if key in seen:
                    stats["duplicate_candidates_collapsed"] += 1
                    continue
                seen.add(key)
                representatives.append(position)
            if len(representatives) < 2:
                stats["decisions_single_distinct_option"] += 1
                continue

            if feature_names is None:
                feature_names = sorted(
                    name for name, value in raw[0].items()
                    if isinstance(value, (int, float, bool))
                )
                features_module.assert_no_leakage(feature_names)

            matrix = np.empty(
                (len(representatives), len(feature_names)), dtype=np.float32
            )
            group_labels: list[int] = []
            for slot, position in enumerate(representatives):
                feature = raw[position]
                matrix[slot] = [
                    float(feature.get(name, -1)) for name in feature_names
                ]
                group_labels.append(
                    int(_semantic_key(feature) == chosen_key)
                )
            if sum(group_labels) != 1:
                stats["decisions_label_not_unique"] += 1
                continue

            feature_batches.append(matrix)
            labels.extend(group_labels)
            groups.append(len(representatives))
            episode_ids.append(int(row["episode_id"]))
            team_ids.append(int(row["team_id"]))
            submission_ids.append(int(row["submission_id"]))
            seats.append(seat)
            turns.append(int(current.get("turn", -1)))
            contexts.append(context)
            won_flags.append(won)
            teacher_action_types.append(
                int(raw[action[0]]["action_type_id"])
            )
            stats["decisions"] += 1
            stats[f"decisions_ctx{context}"] += 1

    if not feature_batches:
        return {"empty": True, "stats": dict(stats)}
    return {
        "empty": False,
        "features": np.concatenate(feature_batches),
        "labels": np.asarray(labels, dtype=np.int8),
        "groups": np.asarray(groups, dtype=np.int32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "team_ids": np.asarray(team_ids, dtype=np.int64),
        "submission_ids": np.asarray(submission_ids, dtype=np.int64),
        "seats": np.asarray(seats, dtype=np.int8),
        "turns": np.asarray(turns, dtype=np.int16),
        "contexts": np.asarray(contexts, dtype=np.int16),
        "won": np.asarray(won_flags, dtype=np.int8),
        "teacher_action_types": np.asarray(
            teacher_action_types, dtype=np.int16
        ),
        "feature_names": feature_names,
        "stats": dict(stats),
    }


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def build_turn_state(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    episode_ids: np.ndarray,
    seats: np.ndarray,
    names: list[str],
) -> np.ndarray:
    """Intra-turn offer/pass history, in the order the teacher acted."""
    starts, ends = _ranges(groups)
    sem_cols = [names.index(name) for name in SEMANTIC]
    cls_cols = [names.index(name) for name in
                ("action_type_id", "candidate_card_id")]
    turn_col = names.index("turn")

    extra = np.zeros((len(labels), len(TURN_FEATURES)), dtype=np.float32)
    seen_candidate: dict[tuple, tuple[int, int, int]] = {}
    seen_class: dict[tuple, tuple[int, int]] = {}
    previous_offered: set[tuple] = set()
    current_key: tuple | None = None
    position = 0

    for decision in range(len(groups)):
        start, end = starts[decision], ends[decision]
        key = (
            int(episode_ids[decision]),
            int(seats[decision]),
            int(features[start, turn_col]),
        )
        if key != current_key:
            current_key = key
            seen_candidate = {}
            seen_class = {}
            previous_offered = set()
            position = 0
        chosen = start + int(np.flatnonzero(labels[start:end] == 1)[0])
        chosen_semantic = tuple(features[chosen, sem_cols].tolist())
        chosen_class = tuple(features[chosen, cls_cols].tolist())

        offered_now: dict[tuple, tuple] = {}
        for row in range(start, end):
            semantic = tuple(features[row, sem_cols].tolist())
            klass = tuple(features[row, cls_cols].tolist())
            offered_now.setdefault(semantic, klass)
            offers, passed, first = seen_candidate.get(semantic, (0, 0, -1))
            class_offers, class_passed = seen_class.get(klass, (0, 0))
            extra[row] = (
                position,
                offers,
                passed,
                int(semantic in previous_offered),
                first if first >= 0 else position,
                class_passed,
                class_offers,
                int(offers == 0),
            )
        for semantic, klass in offered_now.items():
            offers, passed, first = seen_candidate.get(semantic, (0, 0, -1))
            seen_candidate[semantic] = (
                offers + 1,
                passed + int(semantic != chosen_semantic),
                first if first >= 0 else position,
            )
            class_offers, class_passed = seen_class.get(klass, (0, 0))
            seen_class[klass] = (
                class_offers + 1,
                class_passed + int(klass != chosen_class),
            )
        previous_offered = set(offered_now)
        position += 1
    return extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    parser.add_argument("--deck-hash", default="9714ab5c3996f6cc")
    parser.add_argument(
        "--exclude-teams", default="",
        help="Comma separated team ids to drop (policy outliers).",
    )
    parser.add_argument("--limit-per-team", type=int, default=0)
    parser.add_argument(
        "--latest-per-team",
        type=int,
        default=0,
        help=(
            "Keep only each team's newest N episode/seat relations. Unlike "
            "legacy --limit-per-team, this selects from the recent end."
        ),
    )
    parser.add_argument(
        "--include-skipped-existing",
        action="store_true",
        help=(
            "Treat collector-validated skipped_existing replay/log rows as "
            "usable. Off by default so historical corpus commands reproduce "
            "their original selection exactly."
        ),
    )
    parser.add_argument(
        "--selection-manifest-in",
        type=Path,
        help=(
            "Use an immutable CSV selection manifest instead of filtering "
            "the live collection index."
        ),
    )
    parser.add_argument(
        "--selection-manifest-out",
        type=Path,
        help="Write the exact selected replay relations to this CSV.",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--validation-fraction", type=float, default=0.12,
        help="Chronologically newest episodes held out before the test block.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    excluded = {
        int(value) for value in args.exclude_teams.split(",") if value.strip()
    }
    accepted_download_statuses = set(LEGACY_DOWNLOAD_STATUSES)
    if args.include_skipped_existing:
        accepted_download_statuses.add(VALIDATED_CACHE_STATUS)

    if args.selection_manifest_in is not None:
        index_source = args.selection_manifest_in
        index = pd.read_csv(index_source)
        # A manifest is already the selected dataset. Only deterministic
        # relation de-duplication and ordering are repeated.
        index = index.drop_duplicates(subset=["episode_id", "seat_index"])
        index = index.sort_values(
            ["episode_id", "seat_index", "submission_id"],
            kind="stable",
        ).reset_index(drop=True)
        selection_mode = "manifest"
        accepted_download_statuses = set(
            str(value) for value in index["download_status"].dropna().unique()
        )
        manifest_sha256 = hashlib.sha256(index_source.read_bytes()).hexdigest()
    else:
        index_source = args.data_root / "indexes" / "episodes.csv"
        raw_index = pd.read_csv(index_source)
        try:
            index = select_training_index(
                raw_index,
                deck_hash_value=args.deck_hash,
                excluded_teams=excluded,
                accepted_download_statuses=accepted_download_statuses,
                limit_per_team=args.limit_per_team,
                latest_per_team=args.latest_per_team,
            )
        except ValueError as exc:
            parser.error(str(exc))
        selection_mode = "live_index"
        manifest_sha256 = ""

    if index.empty:
        raise SystemExit("no replay relations selected")

    if args.selection_manifest_out is not None:
        manifest_sha256 = write_selection_manifest(
            index,
            args.selection_manifest_out,
        )

    # Split on episode id so no game contributes to two splits, and so the
    # test block is strictly later than everything used for fitting.
    unique_episodes = np.sort(index["episode_id"].unique())
    total = len(unique_episodes)
    test_size = int(total * args.test_fraction)
    validation_size = int(total * args.validation_fraction)
    train_end = total - test_size - validation_size
    validation_min = int(unique_episodes[train_end])
    test_min = int(unique_episodes[train_end + validation_size])

    row_columns = [
        "team_id", "submission_id", "episode_id", "seat_index",
    ]
    if "replay_path" in index.columns:
        row_columns.append("replay_path")
    rows = index[row_columns].to_dict("records")
    print(
        f"trajectories={len(rows)} teams={index['team_id'].nunique()} "
        f"validation_min={validation_min} test_min={test_min}",
        flush=True,
    )

    replay_root = str((args.data_root / "replays").resolve())
    agent_dir = str(args.agent_dir.resolve())
    workers = max(1, min(args.workers, len(rows)))
    # Contiguous chunks keep each episode's decision stream inside one worker,
    # which the intra-turn history depends on.
    bounds = np.linspace(0, len(rows), workers + 1).astype(int)
    chunks = [
        rows[bounds[i]:bounds[i + 1]] for i in range(workers)
        if bounds[i + 1] > bounds[i]
    ]
    if len(chunks) == 1:
        parts = [_extract_chunk((agent_dir, replay_root, chunks[0]))]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            parts = list(executor.map(
                _extract_chunk,
                [(agent_dir, replay_root, chunk) for chunk in chunks],
            ))

    parts = [part for part in parts if not part["empty"]]
    if not parts:
        raise SystemExit("no decisions extracted")
    names = parts[0]["feature_names"]
    if any(part["feature_names"] != names for part in parts):
        raise RuntimeError("worker feature schemas differ")

    arrays = {
        key: np.concatenate([part[key] for part in parts])
        for key in (
            "features", "labels", "groups", "episode_ids", "team_ids",
            "submission_ids",
            "seats", "turns", "contexts", "won",
            "teacher_action_types",
        )
    }
    # Restore global chronological order; workers ran contiguous blocks so
    # each episode's stream is already intact inside its block.
    order = np.argsort(arrays["episode_ids"], kind="stable")
    if not np.array_equal(order, np.arange(len(order))):
        starts, ends = _ranges(arrays["groups"])
        row_order = np.concatenate([
            np.arange(starts[i], ends[i]) for i in order
        ])
        arrays["features"] = arrays["features"][row_order]
        arrays["labels"] = arrays["labels"][row_order]
        for key in ("groups", "episode_ids", "team_ids", "submission_ids",
                    "seats", "turns", "contexts",
                    "won", "teacher_action_types"):
            arrays[key] = arrays[key][order]

    extra = build_turn_state(
        arrays["features"], arrays["labels"], arrays["groups"],
        arrays["episode_ids"], arrays["seats"], names,
    )
    features = np.concatenate([arrays["features"], extra], axis=1)
    all_names = list(names) + list(TURN_FEATURES)

    episode_ids = arrays["episode_ids"]
    splits = np.where(
        episode_ids >= test_min, "test",
        np.where(episode_ids >= validation_min, "validation", "train"),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=arrays["labels"],
        groups=arrays["groups"],
        splits=splits,
        episode_ids=episode_ids,
        team_ids=arrays["team_ids"],
        submission_ids=arrays["submission_ids"],
        seats=arrays["seats"],
        turns=arrays["turns"],
        contexts=arrays["contexts"],
        won=arrays["won"],
        teacher_action_types=arrays["teacher_action_types"],
        feature_names=np.asarray(all_names),
        categorical=np.asarray(sorted(CATEGORICAL & set(all_names))),
    )

    stats: Counter[str] = Counter()
    for part in parts:
        stats.update(part["stats"])
    report = {
        "data_root": str(args.data_root.resolve()),
        "index_source": str(index_source.resolve()),
        "selection_mode": selection_mode,
        "accepted_download_statuses": sorted(accepted_download_statuses),
        "limit_per_team": args.limit_per_team,
        "latest_per_team": args.latest_per_team,
        "selection_manifest_in": (
            str(args.selection_manifest_in.resolve())
            if args.selection_manifest_in is not None
            else ""
        ),
        "selection_manifest_out": (
            str(args.selection_manifest_out.resolve())
            if args.selection_manifest_out is not None
            else ""
        ),
        "selection_manifest_sha256": manifest_sha256,
        "agent_dir": agent_dir,
        "deck_hash": args.deck_hash,
        "excluded_teams": sorted(excluded),
        "cache": str(args.output.resolve()),
        "trajectories": len(rows),
        "teams": int(index["team_id"].nunique()),
        "submissions": int(index["submission_id"].nunique()),
        "selected_rows_by_status": {
            str(status): int(count)
            for status, count in sorted(
                index["download_status"].value_counts().items()
            )
        },
        "selected_rows_by_team": {
            str(int(team)): int(count)
            for team, count in sorted(index.groupby("team_id").size().items())
        },
        "selected_rows_by_submission": {
            str(int(submission)): int(count)
            for submission, count in sorted(
                index.groupby("submission_id").size().items()
            )
        },
        "episodes": int(stats["episodes"]),
        "win_rate": round(stats["wins"] / max(1, stats["episodes"]), 4),
        "decisions": int(len(arrays["groups"])),
        "candidate_rows": int(len(arrays["labels"])),
        "features": len(all_names),
        "base_features": len(names),
        "turn_features": list(TURN_FEATURES),
        "validation_min_episode": validation_min,
        "test_min_episode": test_min,
        "split_decisions": {
            split: int(np.count_nonzero(splits == split))
            for split in ("train", "validation", "test")
        },
        "split_episodes": {
            split: int(len(np.unique(episode_ids[splits == split])))
            for split in ("train", "validation", "test")
        },
        "mean_candidates_per_decision": round(
            float(len(arrays["labels"]) / max(1, len(arrays["groups"]))), 3
        ),
        "extraction_stats": dict(stats),
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
