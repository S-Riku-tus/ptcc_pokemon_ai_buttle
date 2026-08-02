"""Replay held-out teacher games through the real runtime agent.

The offline number is what the model scores on a feature matrix. The number
that matters is what the shipped ``main.py`` does when the Kaggle engine hands
it an observation, and on the Alakazam line those two differed by 5.16 points
because a safety shell overrode the ranker. So this measures the agent, not
the model: it loads the agent directory exactly as Kaggle would and feeds it
every observation of each trajectory in order.

Evaluation is teacher-forced. The board state comes from the teacher's actual
game, so the intra-turn history is advanced with the teacher's action rather
than the agent's suggestion - the same conditioning the corpus was built with.
Both are reported; the free-running variant would drift away from the replay
after the first disagreement and measure nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path


import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402

MAIN_CONTEXT = 0
SEMANTIC = (
    "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "candidate_inplay_area",
    "candidate_target_hp", "candidate_target_energy",
)


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt(
        (phat * (1 - phat) + z * z / (4 * total)) / total
    )
    return [
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-dir", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v1",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--min-episode", type=int, required=True,
                        help="First test episode id from the corpus report.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "ml" / "grimmsnarl" / "processed"
        / "corpus_v1.npz",
        help="Used only to re-pin the teacher code for comparison runs.",
    )
    parser.add_argument(
        "--pin-team", type=int,
        help="Score as this pilot instead of the one baked into the model. "
             "Without it the shipped model is measured exactly as it will "
             "run on Kaggle.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    index = pd.read_csv(args.data_root / "indexes" / "episodes.csv")
    index = index[
        (index["download_status"] == "success")
        & (index["team_id"] == args.team)
        & (index["episode_id"] >= args.min_episode)
    ].drop_duplicates(subset=["episode_id", "seat_index"])
    index = index.sort_values("episode_id")
    if args.limit:
        index = index.head(args.limit)
    if index.empty:
        raise SystemExit("no test episodes for this team")

    _, _, module = load_dir_agent(args.agent_dir)
    agent = module.agent
    ranker = module._RANKER
    if ranker is None:
        raise SystemExit(f"ranker not loaded: {module._LOAD_ERROR}")
    if args.pin_team is not None:
        import numpy as np

        teams = sorted({
            int(x) for x in
            np.load(args.corpus, allow_pickle=False)["team_ids"]
        })
        ranker.teacher_code = teams.index(args.pin_team)
    features_module = sys.modules["ml_features"]
    print(
        f"episodes={len(index)} teacher_team={args.team} "
        f"pinned_code={ranker.teacher_code}",
        flush=True,
    )

    def semantic(current, select, option, position):
        row = features_module.option_features(
            current, select, option, option_position=position
        )
        return tuple(int(row.get(name, -1)) for name in SEMANTIC)

    counts: Counter[str] = Counter()
    latencies: list[float] = []
    for _, row in index.iterrows():
        path = args.data_root / "replays" / f"episode_{row.episode_id}.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        seat = int(row.seat_index)
        steps = replay.get("steps") or []
        module.diag_reset()
        counts["episodes"] += 1

        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            action = (steps[step_index + 1][seat] or {}).get("action")
            if not options:
                continue

            is_main = (
                int(select.get("context", -1)) == MAIN_CONTEXT
                and int(select.get("minCount") or 0) == 1
                and int(select.get("maxCount") or 0) == 1
                and len(options) >= 2
                and isinstance(action, list)
                and len(action) == 1
                and isinstance(action[0], int)
                and 0 <= action[0] < len(options)
            )
            start = time.perf_counter()
            try:
                answer = agent(observation)
            except Exception:
                counts["agent_exception"] += 1
                continue
            elapsed = time.perf_counter() - start
            if not is_main:
                counts["non_main_decisions"] += 1
                continue

            latencies.append(elapsed)
            counts["main_decisions"] += 1
            if not isinstance(answer, list) or len(answer) != 1:
                counts["illegal_shape"] += 1
                continue
            predicted = answer[0]
            if not isinstance(predicted, int) or not 0 <= predicted < len(options):
                counts["illegal_index"] += 1
                continue

            current = observation.get("current") or {}
            teacher = semantic(current, select, options[action[0]], action[0])
            guess = semantic(current, select, options[predicted], predicted)
            counts["correct"] += int(teacher == guess)
            if teacher == guess and predicted != action[0]:
                counts["correct_via_duplicate"] += 1

            # Teacher forcing: the replay continues from the teacher's move,
            # so the intra-turn history must follow the teacher too.
            ranker.observe_external(observation, action[0])

    total = counts["main_decisions"]
    hits = counts["correct"]
    latencies.sort()
    report = {
        "agent_dir": str(args.agent_dir.resolve()),
        "teacher_team": args.team,
        "pinned_teacher_code": ranker.teacher_code,
        "min_episode": args.min_episode,
        "episodes": counts["episodes"],
        "main_decisions": total,
        "runtime_top1": round(hits / total, 4) if total else None,
        "runtime_top1_wilson95": wilson(hits, total),
        "counts": dict(counts),
        "ranker_stats": ranker.snapshot(),
        "latency_ms": {
            "mean": round(1000 * sum(latencies) / max(1, len(latencies)), 2),
            "p50": round(1000 * latencies[len(latencies) // 2], 2)
            if latencies else None,
            "p95": round(1000 * latencies[int(0.95 * len(latencies))], 2)
            if latencies else None,
            "max": round(1000 * latencies[-1], 2) if latencies else None,
        },
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
