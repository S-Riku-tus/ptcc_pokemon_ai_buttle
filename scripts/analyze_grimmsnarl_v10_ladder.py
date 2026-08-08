"""Recompute the v8 ladder run from stored replays, per decision-relevant cut.

The v10 brief hands over a table of ladder numbers. None of them are taken on
trust: this reads ``data/submissions/submission_<id>`` and recomputes every one
from the replays themselves, so a later claim about "the Alakazam losses" is
anchored to episodes that exist rather than to a summary.

Three traps this deliberately avoids, because each one has already produced a
wrong conclusion in this line:

* the validation self-play episode is not a rated game and must be dropped, or
  the record is 51 games and both seats are ours;
* ``seat`` is not turn order. ``current.firstPlayer`` is -1 until the flip, so
  it has to be read from a late step, not the first one;
* the opponent's deck is the 60-card list handed back as the *initial action*
  in step 1, hashed the same way ``ml.core.replay_io.deck_hash`` hashes ours.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}


def deck_hash(card_ids: list[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) * z
    return [
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    ]


def archetype(deck: list[int] | None) -> str:
    if not deck:
        return "unknown"
    pokemon = Counter(
        card_id for card_id in deck
        if CARDS.get(card_id, {}).get("cardType") == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple:
        card_id, count = item
        card = CARDS[card_id]
        return (
            bool(card.get("stage2")),
            bool(card.get("megaEx") or card.get("ex")),
            bool(card.get("stage1")),
            count,
            int(card.get("hp", 0)),
        )

    return CARDS.get(max(pokemon.items(), key=key)[0], {}).get("name", "?")


def deck_label(deck: list[int] | None, top: int = 3) -> str:
    """Name the deck by its heaviest evolution lines, not just one."""
    if not deck:
        return "unknown"
    pokemon = Counter(
        card_id for card_id in deck
        if CARDS.get(card_id, {}).get("cardType") == 0
        and (CARDS[card_id].get("stage1") or CARDS[card_id].get("stage2")
             or CARDS[card_id].get("ex") or CARDS[card_id].get("megaEx"))
    )
    names = [
        CARDS[cid].get("name", str(cid))
        for cid, _ in sorted(
            pokemon.items(),
            key=lambda kv: (-kv[1], -int(CARDS[kv[0]].get("hp", 0))),
        )[:top]
    ]
    return " + ".join(names) if names else archetype(deck)


def rating_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 900:
        return "under_900"
    if score < 1000:
        return "900_999"
    if score < 1100:
        return "1000_1099"
    return "1100_plus"


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(bool(row["won"]) for row in rows)
    return {
        "games": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": round(wins / len(rows), 4) if rows else None,
        "wilson95": wilson(wins, len(rows)),
        "mean_opponent_rating": (
            round(
                sum(r["opponent_score"] for r in rows
                    if r["opponent_score"] is not None)
                / max(1, sum(r["opponent_score"] is not None for r in rows)),
                2,
            )
            if rows else None
        ),
        "episodes": [r["episode_id"] for r in rows],
    }


def load_run(run_dir: Path, submission: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(
        (run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        episode_id = int(raw["episode_id"])
        a0 = raw["agent_0_submission_id"]
        a1 = raw["agent_1_submission_id"]
        rated = (
            raw["episode_type"] == "EPISODE_TYPE_PUBLIC"
            and raw["state"] == "COMPLETED"
            and a0 != a1
        )
        seat = 0 if a0 == submission else 1
        path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        rewards = replay.get("rewards") or [None, None]
        won = None
        if rewards[seat] is not None:
            other = rewards[1 - seat]
            won = bool(rewards[seat] > (other if other is not None else 0))
        steps = replay.get("steps") or []
        decks: list[list[int] | None] = [None, None]
        if len(steps) > 1:
            for s in (0, 1):
                action = (steps[1][s] or {}).get("action")
                if isinstance(action, list) and len(action) == 60:
                    decks[s] = [int(v) for v in action]
        # firstPlayer is -1 until the coin flip resolves, so read it from the
        # last step that carries a board.
        went_first = None
        for step in reversed(steps):
            if seat >= len(step):
                continue
            current = ((step[seat] or {}).get("observation") or {}).get(
                "current"
            )
            if isinstance(current, dict) and current.get("players"):
                first = int(current.get("firstPlayer", -1))
                went_first = (first == seat) if first >= 0 else None
                break

        def score(key: str) -> float | None:
            text = (raw.get(key) or "").strip()
            try:
                return float(text) if text else None
            except ValueError:
                return None

        rows.append({
            "episode_id": episode_id,
            "create_time": raw["create_time"],
            "rated": rated,
            "episode_type": raw["episode_type"],
            "seat": seat,
            "won": won,
            "went_first": went_first,
            "turns": len(steps),
            "opponent_submission": a1 if seat == 0 else a0,
            "opponent_score": score(f"agent_{1 - seat}_initial_score"),
            "our_score_before": score(f"agent_{seat}_initial_score"),
            "our_score_after": score(f"agent_{seat}_updated_score"),
            "our_deck_hash": deck_hash(decks[seat]) if decks[seat] else "",
            "opponent_deck_hash": (
                deck_hash(decks[1 - seat]) if decks[1 - seat] else ""
            ),
            "opponent_archetype": archetype(decks[1 - seat]),
            "opponent_deck_label": deck_label(decks[1 - seat]),
        })
    rows.sort(key=lambda r: r["create_time"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = load_run(args.run_dir, args.submission)
    rated = [r for r in rows if r["rated"] and r["won"] is not None]

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rated:
        by_hash[row["opponent_deck_hash"]].append(row)
        by_bucket[rating_bucket(row["opponent_score"])].append(row)

    ratings = [r["opponent_score"] for r in rated if r["opponent_score"]]
    report = {
        "run_dir": str(args.run_dir),
        "submission": args.submission,
        "episodes_total": len(rows),
        "episodes_rated": len(rated),
        "excluded": [
            {"episode_id": r["episode_id"], "type": r["episode_type"]}
            for r in rows if not r["rated"]
        ],
        "our_deck_hashes": sorted({r["our_deck_hash"] for r in rated}),
        "overall": block(rated),
        "mean_opponent_rating": round(sum(ratings) / len(ratings), 4),
        "rating_first": rated[0]["our_score_before"],
        "rating_last": rated[-1]["our_score_after"],
        "first_10": block(rated[:10]),
        "after_10": block(rated[10:]),
        "by_opponent_rating": {
            k: block(v) for k, v in sorted(by_bucket.items())
        },
        "by_turn_order": {
            "first": block([r for r in rated if r["went_first"] is True]),
            "second": block([r for r in rated if r["went_first"] is False]),
            "unknown": block([r for r in rated if r["went_first"] is None]),
        },
        "by_opponent_deck": {
            k: {
                **block(v),
                "label": v[0]["opponent_deck_label"],
                "archetype": v[0]["opponent_archetype"],
                "mean_opponent_rating": block(v)["mean_opponent_rating"],
            }
            for k, v in sorted(by_hash.items(), key=lambda kv: -len(kv[1]))
        },
        "games": rated,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {k: v for k, v in report.items() if k not in ("games",)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
