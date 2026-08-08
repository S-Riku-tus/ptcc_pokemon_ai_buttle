"""Where do independent advisors agree against v8, and does it repeat?

Reads the decision dump from ``probe_grimmsnarl_v10_advisors.py`` and answers
the only question a *residual* has to answer: which narrow shapes are (a) a
disagreement between v8 and a consensus of advisors that do not share v8's pin,
(b) repeated across several games rather than one, and (c) concentrated where
v8 actually loses.

Nothing here is evidence that the consensus is *right*. The replay was played
by v8, so "consensus disagrees with the replay" is the same event as "consensus
disagrees with v8" and agreement-with-replay cannot score it. What this can do
is rank shapes by how reproducible and how narrow they are, so the ones worth
gating are chosen before any threshold is fitted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CONTEXT_NAMES = {
    0: "MAIN", 1: "setup_active", 2: "setup_bench", 3: "switch",
    4: "to_active", 5: "mulligan_bench", 7: "to_hand/search",
    8: "unknown8", 13: "damage_counter", 15: "damage_target",
    16: "remove_damage_counter", 21: "attach_from", 22: "attach_to",
    27: "ctx27", 30: "ctx30", 37: "ctx37", 38: "ctx38",
    40: "remove_counter_count", 41: "ctx41", 43: "activate",
}


def family(deck_label: str) -> str:
    return deck_label


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def consensus(row: dict[str, Any], base: str, panel: list[str],
              need: int) -> dict[str, Any] | None:
    """The slot at least ``need`` panel advisors picked, if it is not v8's."""
    chosen = row["advisors"].get(base, {}).get("chosen")
    if chosen is None:
        return None
    votes = Counter()
    scored = 0
    for name in panel:
        entry = row["advisors"].get(name) or {}
        slot = entry.get("chosen")
        if slot is None:
            continue
        scored += 1
        votes[slot] += 1
    if scored < need:
        return None
    slot, count = votes.most_common(1)[0]
    if count < need or slot == chosen:
        return None
    return {"slot": slot, "votes": count, "scored": scored}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--base", default="v8")
    parser.add_argument(
        "--panel", default="v9,t16422241,t16452116,t16561259,t16371703",
    )
    parser.add_argument("--need", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = load(args.decisions)
    panel = [name.strip() for name in args.panel.split(",") if name.strip()]

    per_context: dict[int, Counter[str]] = defaultdict(Counter)
    shapes: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    episodes_seen: set[int] = set()
    cells: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        episodes_seen.add(row["episode_id"])
        context = row["context"]
        base_entry = row["advisors"].get(args.base) or {}
        if base_entry.get("chosen") is None:
            per_context[context]["base_unusable"] += 1
            continue
        per_context[context]["decisions"] += 1
        cell = (
            f"{'won' if row['won'] else 'lost'}/"
            f"{'first' if row['went_first'] else 'second'}"
        )
        cells[cell]["decisions"] += 1
        agreement = consensus(row, args.base, panel, args.need)
        if agreement is None:
            continue
        per_context[context]["overrides"] += 1
        cells[cell]["overrides"] += 1
        slot = agreement["slot"]
        proposed_action = None
        proposed_card = None
        for name in panel:
            entry = row["advisors"].get(name) or {}
            if entry.get("chosen") == slot:
                proposed_action = entry.get("action")
                proposed_card = entry.get("card")
                break
        key = (
            context,
            base_entry.get("action"),
            base_entry.get("card"),
            proposed_action,
            proposed_card,
        )
        shapes[key].append({
            "episode_id": row["episode_id"],
            "turn": row["turn"],
            "won": row["won"],
            "went_first": row["went_first"],
            "opponent_deck_hash": row["opponent_deck_hash"],
            "votes": agreement["votes"],
            "scored": agreement["scored"],
            "base_margin": base_entry.get("margin"),
            "options": row["options"],
        })

    ranked = sorted(
        shapes.items(),
        key=lambda kv: (
            -len({e["episode_id"] for e in kv[1]}), -len(kv[1])
        ),
    )
    report = {
        "decisions_file": str(args.decisions),
        "episodes": len(episodes_seen),
        "rows": len(rows),
        "base": args.base,
        "panel": panel,
        "need": args.need,
        "by_context": {
            str(context): {
                **dict(counts),
                "name": CONTEXT_NAMES.get(context, str(context)),
                "override_rate": (
                    round(counts["overrides"] / counts["decisions"], 4)
                    if counts["decisions"] else None
                ),
            }
            for context, counts in sorted(per_context.items())
        },
        "by_outcome_cell": {
            name: {
                **dict(counts),
                "override_rate": (
                    round(counts["overrides"] / counts["decisions"], 4)
                    if counts["decisions"] else None
                ),
            }
            for name, counts in sorted(cells.items())
        },
        "shapes": [
            {
                "context": key[0],
                "context_name": CONTEXT_NAMES.get(key[0], str(key[0])),
                "v8_action": key[1],
                "v8_card": key[2],
                "panel_action": key[3],
                "panel_card": key[4],
                "count": len(events),
                "episodes": len({e["episode_id"] for e in events}),
                "lost_games": len(
                    {e["episode_id"] for e in events if not e["won"]}
                ),
                "second_games": len(
                    {e["episode_id"] for e in events if not e["went_first"]}
                ),
                "examples": events[:6],
            }
            for key, events in ranked
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"rows={len(rows)} episodes={len(episodes_seen)} "
          f"need={args.need}/{len(panel)}")
    print(f"{'ctx':>4} {'name':22s} {'n':>6} {'ovr':>5} {'rate':>7}")
    for context, counts in sorted(
        per_context.items(), key=lambda kv: -kv[1]["overrides"]
    ):
        if not counts["decisions"]:
            continue
        print(f"{context:>4} {CONTEXT_NAMES.get(context, '?'):22s} "
              f"{counts['decisions']:6d} {counts['overrides']:5d} "
              f"{counts['overrides'] / counts['decisions']:7.3f}")
    print()
    for name, counts in sorted(cells.items()):
        print(f"{name:14s} n={counts['decisions']:5d} "
              f"overrides={counts['overrides']:4d} "
              f"rate={counts['overrides'] / max(1, counts['decisions']):.4f}")
    print()
    print("top shapes by distinct episodes:")
    for shape in report["shapes"][:25]:
        print(f"  ctx{shape['context']:>3} {shape['context_name']:20s} "
              f"v8={shape['v8_action']}/{shape['v8_card']} -> "
              f"panel={shape['panel_action']}/{shape['panel_card']}  "
              f"n={shape['count']:3d} eps={shape['episodes']:2d} "
              f"lost={shape['lost_games']:2d} second={shape['second_games']:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
