"""The escalation's firing rate in *live* play, not on v5's stored boards.

The counterfactual probe is teacher-forced: it visits the boards v5 created and
asks v6 what it would do there. v6 plays different boards. That matters for this
change specifically, because refusing the Froslass evolve leaves the Snorunt on
the bench with the Froslass still in hand, so the class can re-fire on later
turns and the decision count can grow rather than shrink.

Plays N alternating-seat games against one opponent and reports the runtime's own
escalation counters as a share of decisions, which is pre-registered ladder gate
3 in RESULTS.md.

Usage:
    python experiments/grimmsnarl_ml_v6/measure_live_escalation.py \
        grimmsnarl_ml_v6 alakazam_ml_v35 --games 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_loader import load_dir_agent  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


def resolve(spec: str) -> Path:
    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        direct = base / spec
        if (direct / "main.py").exists():
            return direct
        for child in sorted(base.glob("*/" + spec)):
            if (child / "main.py").exists():
                return child
    raise SystemExit(f"agent not found: {spec}")


def deck(agent_dir: Path) -> list[int]:
    text = (agent_dir / "deck.csv").read_text(encoding="utf-8-sig")
    return [int(value) for value in text.split()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("opponent")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    candidate_dir, opponent_dir = resolve(args.candidate), resolve(args.opponent)
    candidate, _, module = load_dir_agent(candidate_dir)
    opponent, _, _ = load_dir_agent(opponent_dir)
    decks = [deck(candidate_dir), deck(opponent_dir)]

    totals: Counter[str] = Counter()
    wins = 0
    for game in range(args.games):
        seat = game % 2
        agents = [None, None]
        agents[seat] = candidate
        agents[1 - seat] = opponent
        seat_decks = [None, None]
        seat_decks[seat] = decks[0]
        seat_decks[1 - seat] = decks[1]
        if hasattr(module, "diag_reset"):
            module.diag_reset()
        observation, start = battle_start(seat_decks[0], seat_decks[1])
        if observation is None:
            raise SystemExit(f"battle_start failed: {start.errorPlayer}")
        try:
            for _ in range(2000):
                current = observation["current"]
                if current["result"] >= 0:
                    wins += int(current["result"] == seat)
                    break
                actor = current["yourIndex"]
                observation = battle_select(list(agents[actor](observation)))
        finally:
            battle_finish()
        snapshot = module.diag_snapshot() or {}
        for key, value in (snapshot.get("ml") or {}).items():
            if isinstance(value, (int, float)):
                totals[key] += int(value)

    decisions = totals["main_decisions"] or 1
    report = {
        "candidate": args.candidate,
        "opponent": args.opponent,
        "games": args.games,
        "wins": wins,
        "counters": dict(totals),
        "escalation_offered_share_of_scored_decisions": round(
            totals["escalation_offered"] / decisions, 4
        ),
        "escalation_moved_share_of_offered": round(
            totals["escalation_moved"] / (totals["escalation_offered"] or 1), 4
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
