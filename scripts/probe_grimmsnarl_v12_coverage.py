"""Counterfactual A/B of two Grimmsnarl agents over one agent's own ladder run.

Generalised from ``probe_grimmsnarl_v11_ladder_overrides.py``: ``--base`` and
``--candidate`` are arbitrary agent directories, and every output key is named
after the role rather than after v8/v11, so the same script can be re-pointed
each iteration instead of copied and half-renamed.

It walks both agents side by side over the *deployed* agent's own ladder games -
the states that agent actually reached - and joins each divergence to the result
of the game it was played in.

The single number to read first is ``reproduction.base_rate`` when ``--base`` is
the deployed artifact: it is the probe's fidelity ceiling. If the deployed
policy does not reproduce its own logged actions, the divergence ledger is
measuring something other than what shipped.

The two agents are teacher-forced on the identical stored action, so any
difference between them is the search layer and nothing else. Per-episode
override counts are then split by win/loss and by opponent archetype, which is
the only cut that can distinguish "the layer is neutral and the rating is noise"
from "the layer costs games".

Traps this avoids, each of which has produced a wrong conclusion in this line:

* the self-play validation episode is not rated and must be dropped;
* ``current.firstPlayer`` is -1 until the flip, so turn order is read from a
  late step rather than the first one;
* an override count per *decision* is not a rate per *game*; a layer that fires
  once in a 90-decision game is invisible in the first denominator.
* multi-pick and optional zero-pick selects are real decisions. Both agents are
  called on them so fallback trackers and external-observation hooks stay in
  the same state as live play.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
MIRROR_HASH = "9714ab5c3996f6cc"


def deck_hash(card_ids: list[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def archetype(deck: list[int] | None) -> str:
    if not deck:
        return "unknown"
    pokemon = Counter(
        cid for cid in deck if CARDS.get(cid, {}).get("cardType") == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple:
        cid, count = item
        card = CARDS[cid]
        return (
            bool(card.get("stage2")),
            bool(card.get("megaEx") or card.get("ex")),
            bool(card.get("stage1")),
            count,
            int(card.get("hp", 0)),
        )

    return CARDS.get(max(pokemon.items(), key=key)[0], {}).get("name", "?")


def load(agent_dir: Path):
    agent, _, module = load_dir_agent(agent_dir)
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        raise SystemExit(f"{agent_dir}: no ranker loaded")
    ranker.teacher_forced = True
    return agent, module



# ``ArithmeticSearch.snapshot`` mixes running counters with the layer's static
# configuration. Summing the whole payload across episodes multiplies every
# constant by the episode count, which reads as a shipped-config discrepancy
# that does not exist. Only these keys are additive.
CONFIG_KEYS = frozenset({
    "enabled", "min_turn", "top_k", "determinizations_per_search",
    "max_rank_margin", "min_mean_utility_gain",
    "default_searches_per_turn", "alakazam_second_searches_per_turn",
    "max_searches_per_turn", "degraded_searches_per_turn",
    "overage_reserve", "max_game_search_seconds",
    "overage_remaining_last", "overage_remaining_min",
    "search_seconds_mean",
    "override_records",
})


def legal_replay_action(action: Any, option_count: int) -> list[int] | None:
    """Return a complete legal replay selection, including multi-picks."""
    if not isinstance(action, list):
        return None
    if not all(
        isinstance(slot, int) and 0 <= slot < option_count for slot in action
    ):
        return None
    return list(action)


def describe_answer(
    answer: Any,
    current: dict[str, Any],
    select: dict[str, Any],
    options: list[dict[str, Any]],
    mf,
) -> dict[str, Any] | None:
    """Describe single- and multi-pick answers without losing compatibility."""
    slots = legal_replay_action(answer, len(options))
    if slots is None:
        return None
    cards = []
    for slot in slots:
        card = mf.resolve_option(current, select, options[slot])[0] or {}
        card_id = int(card.get("id", -1))
        cards.append({
            "slot": slot,
            "card": card_id,
            "name": CARDS.get(card_id, {}).get("name", ""),
        })
    if len(cards) == 1:
        return cards[0]
    return {
        "slots": slots,
        "cards": cards,
        "name": " + ".join(card["name"] or "?" for card in cards),
    }


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    from math import comb

    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) < 0 or n == 0:
        return 1.0

    def prob(x: int) -> float:
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)

    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    obs = prob(a)
    return min(1.0, sum(
        prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-12
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v11",
    )
    parser.add_argument(
        "--candidate", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v12",
    )
    parser.add_argument(
        "--run", action="append", required=True,
        help="RUN_DIR:SUBMISSION_ID, repeatable.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base_agent, base_module = load(args.base)
    cand_agent, cand_module = load(args.candidate)
    mf = sys.modules["ml_features"]

    episodes: list[dict[str, Any]] = []
    for spec in args.run:
        run_dir_text, submission = spec.rsplit(":", 1)
        run_dir = Path(run_dir_text)
        for raw in csv.DictReader(
            (run_dir / "episodes.csv").open(encoding="utf-8-sig")
        ):
            a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
            if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
                continue
            if raw["state"] != "COMPLETED":
                continue
            episode_id = int(raw["episode_id"])
            path = (
                run_dir / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not path.exists():
                continue
            seat = 0 if a0 == submission else 1

            def score(key: str) -> float | None:
                text = (raw.get(key) or "").strip()
                try:
                    return float(text) if text else None
                except ValueError:
                    return None

            episodes.append({
                "episode_id": episode_id,
                "submission": submission,
                "seat": seat,
                "path": path,
                "opponent_score": score(f"agent_{1 - seat}_initial_score"),
                "create_time": raw["create_time"],
            })
    episodes.sort(key=lambda e: e["create_time"])
    if args.limit:
        episodes = episodes[: args.limit]

    games: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    decisions_total = 0
    search_totals: Counter[str] = Counter()
    search_config: dict[str, Any] = {}
    # Fidelity: how often each replayed policy reproduces the action the
    # deployed agent actually played. v11's own number bounds how much of the
    # ladder behaviour this probe can explain at all.
    reproduced = Counter()
    selection_sizes: Counter[int] = Counter()

    for meta in episodes:
        episode_id = meta["episode_id"]
        seat = meta["seat"]
        replay = json.loads(meta["path"].read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        won = bool(rewards[seat] > (other if other is not None else 0))

        decks: list[list[int] | None] = [None, None]
        if len(steps) > 1:
            for s in (0, 1):
                action = (steps[1][s] or {}).get("action")
                if isinstance(action, list) and len(action) == 60:
                    decks[s] = [int(v) for v in action]
        opponent_deck = decks[1 - seat]
        opp_hash = deck_hash(opponent_deck) if opponent_deck else ""
        opp_label = (
            "MIRROR" if opp_hash == MIRROR_HASH else archetype(opponent_deck)
        )

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

        for module in (base_module, cand_module):
            module.diag_reset()

        game_decisions = 0
        game_diffs: list[dict[str, Any]] = []

        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            raw_action = (steps[index + 1][seat] or {}).get("action")
            action = legal_replay_action(raw_action, len(options))
            if not options or action is None:
                continue
            current = observation.get("current") or {}
            if len(current.get("players") or []) < 2:
                continue
            game_decisions += 1
            decisions_total += 1
            selection_sizes[len(action)] += 1
            base_answer = base_agent(observation)
            cand_answer = cand_agent(observation)
            reproduced["decisions"] += 1
            if base_answer == action:
                reproduced["base"] += 1
            if cand_answer == action:
                reproduced["candidate"] += 1
            if base_answer != cand_answer:
                row = {
                    "episode_id": episode_id,
                    "won": won,
                    "opponent": opp_label,
                    "turn": int(current.get("turn", -1)),
                    "context": int(select.get("context", -1)),
                    "options": len(options),
                    "base": describe_answer(
                        base_answer, current, select, options, mf
                    ),
                    "candidate": describe_answer(
                        cand_answer, current, select, options, mf
                    ),
                    "played": action[0] if len(action) == 1 else action,
                }
                game_diffs.append(row)
                ledger.append(row)
            external = action[0] if len(action) == 1 else action
            base_module.observe_external(observation, external)
            cand_module.observe_external(observation, external)

        snapshot = (
            (cand_module.diag_snapshot() or {}).get("arithmetic_search") or {}
        )
        for key, value in snapshot.items():
            if key in CONFIG_KEYS:
                search_config.setdefault(key, value)
                continue
            if isinstance(value, (int, float, bool)):
                search_totals[key] += int(value)

        games.append({
            "episode_id": episode_id,
            "submission": meta["submission"],
            "won": won,
            "went_first": went_first,
            "opponent": opp_label,
            "opponent_score": meta["opponent_score"],
            "decisions": game_decisions,
            "overrides": len(game_diffs),
            "searches": int(snapshot.get("searched", 0) or 0),
            "search_overrides": int(snapshot.get("overrides", 0) or 0),
        })
        print(
            f"{episode_id} {'W' if won else 'L'} vs {opp_label:24s} "
            f"decisions={game_decisions:3d} overrides={len(game_diffs):2d} "
            f"searches={snapshot.get('searched', 0)}",
            flush=True,
        )

    # --- the cut the release gate did not have: override count vs outcome ---
    with_ov = [g for g in games if g["overrides"] > 0]
    without = [g for g in games if g["overrides"] == 0]

    def blk(rows: list[dict[str, Any]]) -> dict[str, Any]:
        wins = sum(bool(r["won"]) for r in rows)
        opps = [r["opponent_score"] for r in rows
                if r["opponent_score"] is not None]
        return {
            "games": len(rows),
            "wins": wins,
            "losses": len(rows) - wins,
            "win_rate": round(wins / len(rows), 4) if rows else None,
            "mean_opponent_rating": (
                round(sum(opps) / len(opps), 1) if opps else None
            ),
            "mean_overrides": (
                round(sum(r["overrides"] for r in rows) / len(rows), 3)
                if rows else None
            ),
        }

    by_opponent: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in games:
        grouped[g["opponent"]].append(g)
    for label, rows in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        ov = [r for r in rows if r["overrides"] > 0]
        no = [r for r in rows if r["overrides"] == 0]
        by_opponent[label] = {
            "all": blk(rows),
            "with_override": blk(ov),
            "without_override": blk(no),
        }

    report = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "episodes": len(games),
        "decisions": decisions_total,
        "overrides": len(ledger),
        "override_rate_per_decision": (
            round(len(ledger) / decisions_total, 6) if decisions_total else None
        ),
        "override_rate_per_game": (
            round(len(ledger) / len(games), 4) if games else None
        ),
        "games_with_override": len(with_ov),
        "games_without_override": len(without),
        "outcome_split": {
            "with_override": blk(with_ov),
            "without_override": blk(without),
            "fisher_p": round(fisher(
                sum(g["won"] for g in with_ov),
                len(with_ov) - sum(g["won"] for g in with_ov),
                sum(g["won"] for g in without),
                len(without) - sum(g["won"] for g in without),
            ), 5),
        },
        "search_counters": dict(search_totals),
        "search_config": search_config,
        "reproduction": {
            "decisions": reproduced["decisions"],
            "base_matches_played": reproduced["base"],
            "candidate_matches_played": reproduced["candidate"],
            "base_rate": (
                round(reproduced["base"] / reproduced["decisions"], 4)
                if reproduced["decisions"] else None
            ),
            "candidate_rate": (
                round(reproduced["candidate"] / reproduced["decisions"], 4)
                if reproduced["decisions"] else None
            ),
            "note": (
                "When --base is the deployed artifact its rate is the "
                "probe's fidelity ceiling. A rate well below 1.0 means "
                "the replayed layer is not reproducing deployed "
                "behaviour and the ledger under-counts."
            ),
        },
        "selection_sizes": dict(sorted(selection_sizes.items())),
        "contexts_changed": dict(
            Counter(row["context"] for row in ledger)
        ),
        "turns_changed": dict(sorted(
            Counter(row["turn"] for row in ledger).items()
        )),
        "cards_base_would_play": dict(Counter(
            (row["base"] or {}).get("name", "?") for row in ledger
        ).most_common()),
        "cards_candidate_plays": dict(Counter(
            (row["candidate"] or {}).get("name", "?") for row in ledger
        ).most_common()),
        "by_opponent": by_opponent,
        "games": games,
        "ledger": ledger,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {k: v for k, v in report.items()
         if k not in ("ledger", "games", "by_opponent")},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
