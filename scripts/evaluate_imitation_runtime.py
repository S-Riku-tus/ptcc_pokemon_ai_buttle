"""Teacher-force stored games through the actual submitted imitation shell.

This measures the final agent action after routing, OOD guards, and fallback;
matrix-level model accuracy alone cannot reveal those integration effects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent  # noqa: E402


SEMANTIC = (
    "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "candidate_inplay_area",
    "candidate_target_hp", "candidate_target_energy",
    "ctx_card_id", "ctx_area", "ctx_owner_is_self", "ctx_number",
)


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.959963985
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return [
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    ]


def legal(action: Any, select: dict[str, Any]) -> bool:
    if not isinstance(action, list) or any(not isinstance(value, int) for value in action):
        return False
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    maximum = int(select.get("maxCount") or 0)
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(0 <= value < len(options) for value in action)
    )


def attachment_flags(observation: dict[str, Any], action: Any) -> Counter[str]:
    """Count typed Dragapult attachments in an action on the observed board."""
    out: Counter[str] = Counter()
    if not isinstance(action, list):
        return out
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    hand = mine.get("hand") or []
    options = select.get("option") or []
    for index in action:
        if not isinstance(index, int) or not 0 <= index < len(options):
            continue
        option = options[index]
        if int(option.get("type", -1)) != 8:
            continue
        hand_index = int(option.get("index", -1))
        if not 0 <= hand_index < len(hand) or not isinstance(hand[hand_index], dict):
            continue
        source = int(hand[hand_index].get("id", -1))
        area = int(option.get("inPlayArea", -1))
        target_index = int(option.get("inPlayIndex", -1))
        zone = mine.get("active") if area == 4 else mine.get("bench") if area == 5 else []
        if not isinstance(zone, list) or not 0 <= target_index < len(zone):
            continue
        target = zone[target_index]
        if not isinstance(target, dict):
            continue
        target_id = int(target.get("id", -1))
        energies = [int(value) for value in target.get("energies") or []]
        out["attachments"] += 1
        if source in (2, 5) and target_id in (119, 120, 121):
            out["route_attachments"] += 1
            if source in energies:
                out["duplicate_route_color"] += 1
            else:
                out["useful_route_color"] += 1
                other = 5 if source == 2 else 2
                out["completes_route_colors"] += int(other in energies)
        if source == 7 and target_id == 112:
            out["munkidori_dark"] += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--run-dir", type=Path,
        help="Downloaded submission run; its submitted actions become replay labels.",
    )
    parser.add_argument("--index", type=Path)
    parser.add_argument("--team", type=int)
    parser.add_argument("--min-episode", type=int, default=0)
    parser.add_argument(
        "--split-report", type=Path,
        help="training report containing per-team [validation_min, test_min] boundaries",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"),
        help="evaluate only this chronological split (requires --split-report)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if bool(args.run_dir) == bool(args.data_root):
        raise SystemExit("use exactly one of --data-root or --run-dir")
    if args.run_dir:
        index_path = args.run_dir / "manifest.csv"
        manifest = pd.read_csv(index_path)
        index = pd.DataFrame({
            "download_status": "success",
            "team_id": 0,
            "episode_id": manifest["episode_id"].astype(int),
            "seat_index": manifest["detected_submission_agent_index"].astype(int),
            "replay_path": [
                str((
                    args.run_dir / "episodes" / str(int(episode_id)) / "replay"
                    / f"episode_{int(episode_id)}.json"
                ).resolve())
                for episode_id in manifest["episode_id"]
            ],
        })
        data_root = args.run_dir
    else:
        assert args.data_root is not None
        index_path = args.index or args.data_root / "indexes" / "episodes.csv"
        index = pd.read_csv(index_path)
        data_root = args.data_root
    index = index[index["download_status"].isin(["success", "skipped_existing"])]
    if args.team is not None:
        index = index[index["team_id"] == args.team]
    if args.min_episode:
        index = index[index["episode_id"] >= args.min_episode]
    if bool(args.split_report) != bool(args.split):
        raise SystemExit("--split-report and --split must be used together")
    if args.split_report:
        training = json.loads(args.split_report.read_text(encoding="utf-8"))
        boundaries = training.get("split_boundaries") or {}

        def in_split(row: pd.Series) -> bool:
            boundary = boundaries.get(str(int(row.team_id)))
            if not boundary or len(boundary) != 2:
                return False
            episode = int(row.episode_id)
            validation_min, test_min = map(int, boundary)
            if args.split == "train":
                return episode < validation_min
            if args.split == "validation":
                return validation_min <= episode < test_min
            return episode >= test_min

        index = index[index.apply(in_split, axis=1)]
    index = index.drop_duplicates(subset=["episode_id", "seat_index"])
    index = index.sort_values(["episode_id", "seat_index"])
    if args.limit:
        index = index.head(args.limit)
    if index.empty:
        raise SystemExit("no evaluation episodes")

    _, _, module = load_dir_agent(args.agent_dir)
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        raise SystemExit(f"ranker not loaded: {getattr(module, '_LOAD_ERROR', None)}")
    features = sys.modules["ml_features"]

    def semantic(observation: dict[str, Any], position: int) -> tuple:
        select = observation.get("select") or {}
        option = (select.get("option") or [])[position]
        row = features.option_features(
            observation.get("current") or {}, select, option,
            option_position=position,
        )
        action_map = {name: index for index, name in enumerate(features.ACTION_TYPES)}
        row["action_type_id"] = action_map.get(
            str(row.pop("action_type", "other")), action_map["other"]
        )
        return tuple(int(row.get(name, -1)) for name in SEMANTIC)

    def semantic_action(observation: dict[str, Any], action: list[int]) -> list[tuple]:
        return sorted(semantic(observation, position) for position in action)

    counts: Counter[str] = Counter()
    by_context: dict[int, Counter[str]] = defaultdict(Counter)
    by_route: dict[str, Counter[str]] = defaultdict(Counter)
    ranker_totals: Counter[str] = Counter()
    fallback_totals: Counter[str] = Counter()
    fallback_errors: Counter[str] = Counter()
    guard_totals: Counter[str] = Counter()
    predicted_behaviour: Counter[str] = Counter()
    teacher_behaviour: Counter[str] = Counter()
    latencies: list[float] = []
    for _, relation in index.iterrows():
        replay_value = str(getattr(relation, "replay_path", "") or "").strip()
        if replay_value:
            candidate = Path(replay_value)
            replay_path = candidate if candidate.is_absolute() else data_root / candidate
        else:
            replay_path = data_root / "replays" / f"episode_{relation.episode_id}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        seat = int(relation.seat_index)
        steps = replay.get("steps") or []
        module.diag_reset()
        ranker.teacher_forced = True
        counts["episodes"] += 1

        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = select.get("option") or []
            teacher = (steps[step_index + 1][seat] or {}).get("action")
            if not options or not legal(teacher, select):
                continue
            context = int(select.get("context", -1))
            minimum = int(select.get("minCount") or 0)
            maximum = int(select.get("maxCount") or 0)
            slice_name = (
                "mandatory_single" if minimum == maximum == 1
                else "optional" if minimum == 0 and maximum == 1
                else "multi"
            )
            before_used = ranker.snapshot().get("ranker_used", 0)
            before_guarded = int(getattr(module, "_GUARD_STATS", {}).get("overrides", 0))
            start = time.perf_counter()
            predicted = None
            try:
                predicted = module.agent(observation)
            except Exception:  # noqa: BLE001
                counts["agent_exception"] += 1
            latencies.append(time.perf_counter() - start)
            after_used = ranker.snapshot().get("ranker_used", 0)
            after_guarded = int(getattr(module, "_GUARD_STATS", {}).get("overrides", 0))
            route = (
                "guarded" if after_guarded > before_guarded
                else "ml" if after_used > before_used
                else "fallback"
            )

            is_legal = legal(predicted, select)
            correct = bool(
                is_legal
                and semantic_action(observation, predicted)
                == semantic_action(observation, teacher)
            )
            predicted_behaviour.update(attachment_flags(observation, predicted))
            teacher_behaviour.update(attachment_flags(observation, teacher))
            counts["decisions"] += 1
            counts["legal"] += int(is_legal)
            counts["correct"] += int(correct)
            counts[f"slice::{slice_name}::decisions"] += 1
            counts[f"slice::{slice_name}::correct"] += int(correct)
            counts[f"slice::{slice_name}::legal"] += int(is_legal)
            by_context[context]["decisions"] += 1
            by_context[context]["correct"] += int(correct)
            by_context[context]["legal"] += int(is_legal)
            by_route[route]["decisions"] += 1
            by_route[route]["correct"] += int(correct)
            by_route[route]["legal"] += int(is_legal)
            if len(teacher) == 1:
                module.observe_external(observation, teacher[0])

        episode_diag = module.diag_snapshot()
        ranker_totals.update(episode_diag.get("ml") or {})
        fallback_diag = episode_diag.get("fallback") or {}
        fallback_totals.update({
            key: value for key, value in fallback_diag.items()
            if isinstance(value, (int, float))
        })
        fallback_errors.update(fallback_diag.get("errors") or {})
        guard_totals.update(episode_diag.get("guard") or {})

    def rates(counter: Counter[str]) -> dict[str, Any]:
        total = counter["decisions"]
        return {
            "decisions": total,
            "semantic_top1": round(counter["correct"] / total, 4) if total else None,
            "legal_rate": round(counter["legal"] / total, 4) if total else None,
        }

    latencies.sort()
    total = counts["decisions"]
    report = {
        "agent_dir": str(args.agent_dir.resolve()),
        "index": str(index_path.resolve()),
        "team": args.team,
        "min_episode": args.min_episode,
        "split_report": str(args.split_report.resolve()) if args.split_report else None,
        "split": args.split,
        "episodes": int(counts["episodes"]),
        "decisions": int(total),
        "semantic_agreement": round(counts["correct"] / total, 4),
        "semantic_agreement_wilson95": wilson(counts["correct"], total),
        "legal_rate": round(counts["legal"] / total, 4),
        "by_slice": {
            name: rates(Counter({
                "decisions": counts[f"slice::{name}::decisions"],
                "correct": counts[f"slice::{name}::correct"],
                "legal": counts[f"slice::{name}::legal"],
            }))
            for name in ("mandatory_single", "optional", "multi")
        },
        "by_context": {str(key): rates(value) for key, value in sorted(by_context.items())},
        "by_route": {key: rates(value) for key, value in sorted(by_route.items())},
        "agent_exceptions": int(counts["agent_exception"]),
        "ranker_stats": dict(ranker_totals),
        "diagnostics": {
            "ml": dict(ranker_totals),
            "fallback": {**dict(fallback_totals), "errors": dict(fallback_errors)},
            "guard": dict(guard_totals),
            "load_error": getattr(module, "_LOAD_ERROR", None),
        },
        "behaviour": {
            "predicted": dict(predicted_behaviour),
            "teacher": dict(teacher_behaviour),
        },
        "latency_ms": {
            "mean": round(1000 * sum(latencies) / max(1, len(latencies)), 3),
            "p95": round(1000 * latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3)
            if latencies else None,
            "max": round(1000 * latencies[-1], 3) if latencies else None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if counts["legal"] == total and not counts["agent_exception"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
