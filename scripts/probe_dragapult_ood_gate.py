"""Where does the runtime out-of-distribution gate fire on the ladder?

`Ranker._supported` rejects a decision when a candidate carries a card id the
training set never contained.  On teacher replays that essentially never
happens; on the ladder it does, because the field below 1000 plays decks the
1050+ teachers never met.  This walks a downloaded run, calls the ranker on
every decision, and records the select context and the offending card whenever
the gate fires - the breakdown the aggregate counter cannot give.

Usage:
  python scripts/probe_dragapult_ood_gate.py \
      data/submissions/submission_55550682_dragapult_v2 \
      --agent-dir agents/dragapult/dragapult_ml_v2 \
      --report experiments/dragapult_ml_v2/ood_gate_v2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_loader import load_dir_agent_module  # noqa: E402

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
CONTEXTS = {
    0: "MAIN", 3: "search-card", 5: "energy-card", 7: "TO_HAND", 8: "DISCARD",
    13: "DAMAGE_COUNTER", 14: "DAMAGE_COUNTER_ANY (Phantom Dive)",
    21: "ATTACH_FROM", 22: "ATTACH_TO", 30: "DISCARD_ENERGY",
}


def name(card_id: Any) -> str:
    try:
        return str(CARDS.get(int(card_id), {}).get("name") or card_id)
    except (TypeError, ValueError):
        return str(card_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    module = load_dir_agent_module(args.agent_dir.resolve())
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        raise SystemExit("agent has no ranker")
    known = set(ranker.support.get("candidate_card_id") or [])

    by_context: Counter[str] = Counter()
    by_card: Counter[str] = Counter()
    context_totals: Counter[int] = Counter()
    ood_total = 0
    decisions = 0
    games = 0

    for row in csv.DictReader(
        (args.run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
    ):
        seat = row.get("detected_submission_agent_index", "")
        if seat not in ("0", "1"):
            continue
        path = (args.run / "episodes" / str(row["episode_id"]) / "replay"
                / f"episode_{row['episode_id']}.json")
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        games += 1
        module.diag_reset()
        seat_index = int(seat)
        for index, pair in enumerate(steps):
            payload = pair[seat_index]
            if payload.get("status") != "ACTIVE":
                continue
            observation = payload.get("observation") or {}
            if not isinstance(observation, dict):
                continue
            select = observation.get("select")
            action = (steps[index + 1][seat_index].get("action")
                      if index + 1 < len(steps) else None)
            if select is None:
                ranker.reset()
                module._fallback_agent(observation)
                continue
            if not isinstance(action, list) or len(action) != 1:
                continue
            options = select.get("option") or []
            played = int(action[0])
            if not 0 <= played < len(options):
                continue

            context = int(select.get("context", -1))
            context_totals[context] += 1
            decisions += 1
            before = ranker.stats["ood_fallback"]
            ranker.choose(observation)
            if ranker.stats["ood_fallback"] > before:
                ood_total += 1
                by_context[f"{context} {CONTEXTS.get(context, '')}"] += 1
                # Which candidate identities the gate did not recognise.
                try:
                    rows, _ = ranker._rows(observation)
                except Exception:
                    rows = []
                for candidate in rows:
                    card_id = candidate.get("candidate_card_id")
                    if card_id is None:
                        continue
                    if int(card_id) >= 0 and int(card_id) not in known:
                        by_card[name(card_id)] += 1
            ranker.observe_external(observation, played)

    print(f"{games} games, {decisions} decisions")
    print(f"out-of-distribution fallbacks: {ood_total} "
          f"({ood_total / max(1, decisions):.4f})")
    print("\nby context (share of that context's decisions):")
    for key, count in by_context.most_common():
        context = int(key.split()[0])
        total = context_totals[context]
        print(f"  {key:38} {count:>5} / {total:<5} ({count / max(1, total):.3f})")
    print("\nunrecognised candidate identities:")
    for card, count in by_card.most_common(25):
        print(f"  {card:38} {count:>5}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "run": str(args.run), "games": games, "decisions": decisions,
            "ood_fallbacks": ood_total,
            "ood_rate": round(ood_total / max(1, decisions), 4),
            "by_context": dict(by_context), "by_card": dict(by_card),
            "context_totals": {str(k): v for k, v in context_totals.items()},
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
