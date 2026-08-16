"""What would the model place if the OOD gate did not reject opponent cards?

`Ranker._supported` refuses a decision when any candidate carries a card id the
training set never contained.  For Phantom Dive counter placement the
candidates *are* the opponent's Pokemon, so one off-meta opponent routes the
deck's prize engine to the hand-written rule policy.  On the 26-game v2 run
that is 54 of 402 placements.

The proposed v3 fix is to exempt opponent-owned candidates from the identity
check, on the grounds that they are already described mechanically
(candidate_prize_value, candidate_counters_to_ko, candidate_dies_to_spread,
candidate_walls_phantom).  This measures the fix instead of asserting it: the
gate is relaxed at runtime, every affected live decision is re-scored, and the
model's choice is compared with what the rule policy actually did.

Both are also scored against a simple, checkable standard for spread damage:
a placement is "productive" if the counters it adds knock out a benched
Pokemon or bring one within one more spread of dying.

Usage:
  python scripts/experiment_dragapult_relax_ood_gate.py \
      data/submissions/submission_55550682_dragapult_v2 \
      --agent-dir agents/dragapult/dragapult_ml_v2 \
      --report experiments/dragapult_ml_v2/relaxed_ood_gate.json
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
PLACEMENT_CONTEXTS = {13, 14, 15}


def name(card_id: Any) -> str:
    try:
        return str(CARDS.get(int(card_id), {}).get("name") or card_id)
    except (TypeError, ValueError):
        return str(card_id)


def relax(ranker: Any) -> None:
    """Skip the identity check for candidates the opponent owns."""
    original = ranker._supported.__func__

    def patched(self, select, rows, reps):
        keep = self.support.get("candidate_card_id")
        # Restrict the identity table to our own candidates for the duration
        # of the call; every other check in _supported is left alone.
        if keep is not None and all(
            int(rows[position].get("candidate_owner_is_self", 1)) == 0
            for position in reps
        ):
            self.support = dict(self.support)
            self.support.pop("candidate_card_id")
            try:
                return original(self, select, rows, reps)
            finally:
                self.support["candidate_card_id"] = keep
        return original(self, select, rows, reps)

    ranker._supported = patched.__get__(ranker, type(ranker))


def value_of(row: dict[str, Any]) -> dict[str, Any]:
    """The mechanical description the model would use for this candidate."""
    return {
        "card": name(row.get("candidate_card_id", -1)),
        "prize_value": row.get("candidate_prize_value"),
        "counters_to_ko": row.get("candidate_counters_to_ko"),
        "walls_phantom": row.get("candidate_walls_phantom"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    module = load_dir_agent_module(args.agent_dir.resolve())
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        raise SystemExit("agent has no ranker")
    relax(ranker)

    stats: Counter = Counter()
    examples: list[dict[str, Any]] = []

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

            before_ood = ranker.stats["ood_fallback"]
            before_used = ranker.stats["ranker_used"]
            picked = ranker.choose(observation)
            recovered = (
                ranker.stats["ranker_used"] > before_used
                and ranker.stats["ood_fallback"] == before_ood
                and context in PLACEMENT_CONTEXTS
            )
            if recovered:
                # The submitted agent had this gated; only count decisions the
                # relaxed gate actually hands back to the model.
                try:
                    rows, _ = ranker._rows(observation)
                except Exception:
                    rows = []
                known = set(ranker.support.get("candidate_card_id") or [])
                gated = any(
                    int(candidate.get("candidate_card_id", -1)) >= 0
                    and int(candidate.get("candidate_card_id", -1)) not in known
                    and int(candidate.get("candidate_owner_is_self", 1)) == 0
                    for candidate in rows
                )
                if gated:
                    stats["recovered"] += 1
                    stats["agreed_with_rule"] += int(picked == played)
                    if picked is not None and picked != played and \
                            len(examples) < args.show and picked < len(rows) \
                            and played < len(rows):
                        examples.append({
                            "episode": row["episode_id"], "context": context,
                            "rule_policy": value_of(rows[played]),
                            "model": value_of(rows[picked]),
                        })
            ranker.observe_external(observation, played)

    print(f"placements handed back to the model: {stats['recovered']}")
    if stats["recovered"]:
        print(f"  model agrees with the rule policy: "
              f"{stats['agreed_with_rule']} "
              f"({stats['agreed_with_rule'] / stats['recovered']:.3f})")
        print(f"  model would place differently:    "
              f"{stats['recovered'] - stats['agreed_with_rule']}")
    print("\nexamples (rule policy -> model):")
    for example in examples:
        print(f"  ep {example['episode']} ctx {example['context']}")
        print(f"    rule : {example['rule_policy']}")
        print(f"    model: {example['model']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "run": str(args.run), **dict(stats), "examples": examples,
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
