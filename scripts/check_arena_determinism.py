"""Does ``--seed`` actually pair two arena runs? (No.)

The v10 brief asks that v8 and the candidate be compared on "the same seed
and the same seat order". ``scripts/local_arena.py`` takes ``--seed`` and
``scripts/self_play.py`` has ``--reseed-each-game`` "for paired A/B
evaluation",
which both read as if common random numbers were available. They are not: the
shuffle happens inside ``vendor/cg``'s native library, which exposes no seeding
entry point, and ``random.seed`` only touches the Python RNG that the baseline
agents use.

This runs one fixed matchup of deterministic agents twice, in two processes,
with the same seed, and reports whether the per-game results match. If they do
not, every arena number in this line is an *unpaired* sample and has to be read
with a Wilson interval rather than as a paired difference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(agent_a: str, agent_b: str, games: int, seed: int) -> list[str]:
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "local_arena.py"),
            agent_a, agent_b, "--games", str(games), "--seed", str(seed),
        ],
        capture_output=True, text=True, cwd=str(ROOT), check=True,
    )
    return [
        line.split("->", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("game") and "->" in line
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", default="marnies_grimmsnarl_ex_v7")
    parser.add_argument("--agent-b", default="first")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    runs = [
        run(args.agent_a, args.agent_b, args.games, args.seed)
        for _ in range(args.repeats)
    ]
    identical = all(sequence == runs[0] for sequence in runs)
    # A matchup one side sweeps is identical across processes for a reason
    # that has nothing to do with seeding, so it cannot answer the question.
    # The first attempt here picked such a matchup and read "deterministic"
    # off eight identical wins.
    informative = len({result for run_ in runs for result in run_}) > 1
    report = {
        "agent_a": args.agent_a,
        "agent_b": args.agent_b,
        "games": args.games,
        "seed": args.seed,
        "runs": runs,
        "identical_across_processes": identical,
        "informative": informative,
        "conclusion": (
            "inconclusive: one side swept every game, so identical runs "
            "prove "
            "nothing about seeding - rerun on a matchup with variance"
            if not informative else
            "paired comparison is available"
            if identical else
            "the engine shuffle is not seeded from Python; arena results are "
            "unpaired samples and a candidate cannot be compared to v8 on "
            "identical games"
        ),
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
