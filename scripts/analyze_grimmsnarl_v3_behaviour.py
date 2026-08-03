"""Counterfactual behaviour probe for the Grimmsnarl agents.

The deep analysis of the v2 ladder run named four behaviours and measured them
on what v2 *played*. Nothing measured what a modified agent *would* play on the
same boards, so every candidate change had to be judged by a ladder run whose
rating noise is larger than the effect being tested (see the rating-noise
finding: an identical agent scored 842.8 and 804).

This replays stored games decision by decision and asks the candidate agent for
its answer at every select, then advances the game with the action that was
actually taken. Teacher forcing keeps the boards on-distribution and keeps the
runtime's intra-turn history describing the same turn the replay described, so
two runs differing only in a pin or a planner are compared on identical states.

Reported per group:

* what the replay played  (``played_*``)
* what the candidate agent would play on the same state (``agent_*``)
* how often the two agree

The four behaviours are the ones the analysis flagged, computed with the same
definitions as ``.tmp/grimmsnarl_deep_analysis.py``:

* REMOVE_DAMAGE_COUNTER (16): passing over a Grimmsnarl ex with >= 30 damage.
* REMOVE_COUNTER_COUNT  (40): moving fewer than the maximum offered counters,
  which both heals less and deals less.
* DAMAGE (15): the Shadow Bullet Bench-30 target, and whether it took the
  best-prize kill on offer.
* TO_ACTIVE (4) after a Boss's Orders: gusting a body the Bench-30 alone kills.
* MAIN (0): evolving into Froslass while Freezing Shroud is net negative
  for us, and playing Boss while a Bench-30 lethal already exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
# Freezing Shroud reads "each Pokemon that has an Ability (both yours and your
# opponent's), except any Froslass", so the Froslass line is excluded from the
# set it damages. Same definition as the deep analysis.
FROSLASS_ID = 104
ABILITY_POKEMON = {
    card_id
    for card_id, card in CARDS.items()
    if int(card.get("cardType", -1)) == 0
    and bool(card.get("skills"))
    and card_id != FROSLASS_ID
}

MAIN = 0
# Boss's Orders hands the gust target back as a SWITCH select, not TO_ACTIVE;
# TO_ACTIVE is how we promote our own body after a knockout.
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_HAND = 7
CTX_DAMAGE_COUNTER = 13
CTX_DAMAGE = 15
CTX_REMOVE_DAMAGE_COUNTER = 16
CTX_REMOVE_COUNTER_COUNT = 40
GRIMMSNARL_EX_ID = 648
BENCH_SNIPE_DAMAGE = 30.0


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


def board(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        card
        for area in ("active", "bench")
        for card in (player.get(area) or [])
        if isinstance(card, dict)
    ]


def ability_holders(player: dict[str, Any]) -> int:
    return sum(int(c.get("id", -1)) in ABILITY_POKEMON for c in board(player))


class Probe:
    """Accumulates one group's counters."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []

    def add(self, key: str, value: int = 1) -> None:
        self.counts[key] += value


def rate(counts: Counter[str], numerator: str, denominator: str):
    total = counts[denominator]
    if not total:
        return None
    return round(counts[numerator] / total, 4)


def episode_paths(args: argparse.Namespace) -> list[tuple[Path, int, int]]:
    """(replay path, seat, episode id) for the requested source."""
    out: list[tuple[Path, int, int]] = []
    if args.run_dir:
        rows = list(csv.DictReader(
            (args.run_dir / "episodes.csv").open(encoding="utf-8-sig")
        ))
        for row in rows:
            episode_id = int(row["episode_id"])
            seat = 0 if row["agent_0_submission_id"] == args.submission else 1
            path = (
                args.run_dir / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if path.exists():
                out.append((path, seat, episode_id))
    else:
        import pandas as pd

        index = pd.read_csv(args.data_root / "indexes" / "episodes.csv")
        index = index[
            (index["download_status"] == "success")
            & (index["deck_hash"] == args.deck_hash)
        ]
        if args.teams:
            teams = {int(v) for v in args.teams.split(",") if v.strip()}
            index = index[index["team_id"].isin(teams)]
        if args.min_episode:
            index = index[index["episode_id"] >= args.min_episode]
        index = index.drop_duplicates(subset=["episode_id", "seat_index"])
        index = index.sort_values("episode_id")
        for _, row in index.iterrows():
            path = (
                args.data_root / "replays"
                / f"episode_{row.episode_id}.json"
            )
            if path.exists():
                out.append((path, int(row.seat_index), int(row.episode_id)))
    if args.limit:
        out = out[: args.limit]
    return out


def team_code_map(corpus: Path) -> dict[int, int]:
    import numpy as np

    data = np.load(corpus, allow_pickle=False)
    teams = sorted({int(x) for x in data["team_ids"]})
    return {team: index for index, team in enumerate(teams)}


RATING_BUCKETS = (
    ("under_900", 0.0, 900.0),
    ("900_999", 900.0, 1000.0),
    ("1000_1099", 1000.0, 1100.0),
    ("1100_plus", 1100.0, 1e9),
)


def rating_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    for name, low, high in RATING_BUCKETS:
        if low <= score < high:
            return name
    return "unknown"


def archetype(deck: list[int] | None) -> str:
    """Name the opponent's deck by its heaviest evolution line.

    Same rule as the v2 deep analysis, so the per-match-up tables from the two
    reports line up.
    """
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

    best = max(pokemon.items(), key=key)[0]
    return CARDS.get(best, {}).get("name", f"#{best}")


def initial_decks(replay: dict[str, Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    steps = replay.get("steps") or []
    if len(steps) > 1:
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[seat] = [int(v) for v in action]
    return decks


def run_outcomes(
    run_dir: Path,
    submission: str,
) -> dict[int, dict[str, Any]]:
    """Per-episode result, opponent rating at pairing time, and archetype.

    The v2 analysis could only bucket opponents by joining today's public
    leaderboard, which matched 12 of 59 and still could not say what any of
    them was rated when the game was played. ``fetch_submission_logs`` now
    stores ``agent_<n>_initial_score`` from the same EpisodeService response
    that lists the episode, so later runs can be split by opponent strength.
    """
    path = run_dir / "episodes.csv"
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        episode_id = int(row["episode_id"])
        seat = 0 if row.get("agent_0_submission_id") == submission else 1
        raw = row.get(f"agent_{1 - seat}_initial_score") or ""
        try:
            opponent_score = float(raw) if raw.strip() else None
        except ValueError:
            opponent_score = None
        replay_path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        won = went_first = None
        opponent_deck = None
        if replay_path.exists():
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            rewards = replay.get("rewards") or [None, None]
            if seat < len(rewards) and rewards[seat] is not None:
                other = rewards[1 - seat]
                won = bool(rewards[seat] > (other if other is not None else 0))
            steps = replay.get("steps") or []
            for step in reversed(steps):
                current = (
                    (step[seat] or {}).get("observation") or {}
                ).get("current") if seat < len(step) else None
                if isinstance(current, dict) and current.get("players"):
                    went_first = int(current.get("firstPlayer", -1)) == seat
                    break
            opponent_deck = initial_decks(replay)[1 - seat]
        out[episode_id] = {
            "seat": seat,
            "won": won,
            "went_first": went_first,
            "opponent_score": opponent_score,
            "opponent_bucket": rating_bucket(opponent_score),
            "opponent_archetype": archetype(opponent_deck),
        }
    return out


def outcome_summary(outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    played = [row for row in outcomes.values() if row["won"] is not None]
    if not played:
        return {}

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        wins = sum(bool(row["won"]) for row in rows)
        return {
            "games": len(rows),
            "wins": wins,
            "win_rate": round(wins / len(rows), 4),
            "wilson95": wilson(wins, len(rows)),
        }

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in played:
        by_bucket[row["opponent_bucket"]].append(row)
        by_archetype[row["opponent_archetype"]].append(row)
    return {
        "overall": block(played),
        "went_first": block([r for r in played if r["went_first"]]),
        "went_second": block([r for r in played if r["went_first"] is False]),
        "by_opponent_rating": {
            name: block(rows) for name, rows in sorted(by_bucket.items())
        },
        "by_opponent_archetype": {
            name: block(rows) for name, rows in sorted(
                by_archetype.items(), key=lambda kv: -len(kv[1])
            )
        },
        "rating_known": sum(
            int(row["opponent_score"] is not None) for row in played
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dir", type=Path,
        help="Our ladder run dir (episodes.csv + episodes/<id>/...).",
    )
    parser.add_argument("--submission", default="55205556")
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
        help="Teacher archive, used when --run-dir is absent.",
    )
    parser.add_argument("--deck-hash", default="9714ab5c3996f6cc")
    parser.add_argument("--teams", default="")
    parser.add_argument("--min-episode", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "ml" / "grimmsnarl" / "processed"
        / "corpus_v21.npz",
        help="Only read to translate team ids into the dense pin codes.",
    )
    parser.add_argument(
        "--pin-team", type=int,
        help="Score every context as this pilot instead of the baked-in one.",
    )
    parser.add_argument(
        "--pin-context", default="",
        help="context:team pairs, e.g. '16:16371703,4:16371703'. Applies on "
             "top of --pin-team.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    agent, _, module = load_dir_agent(args.agent_dir)
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        raise SystemExit("agent has no ranker loaded")

    observe_hook = getattr(module, "observe_external", None)
    codes = team_code_map(args.corpus)
    baseline_code = ranker.teacher_code
    if args.pin_team is not None:
        baseline_code = codes[args.pin_team]
    context_codes = {
        int(pair.split(":")[0]): codes[int(pair.split(":")[1])]
        for pair in args.pin_context.split(",") if ":" in pair
    }

    groups: dict[str, Probe] = defaultdict(Probe)
    totals: Counter[str] = Counter()
    # diag_reset() clears the planner's counters per episode, so they are
    # harvested before each reset and once more at the end. Without this the
    # report would show only the last game's overrides.
    planner_totals: Counter[str] = Counter()
    episodes = episode_paths(args)
    if not episodes:
        raise SystemExit("no episodes found")
    print(
        f"episodes={len(episodes)} baseline_code={baseline_code} "
        f"context_pins={context_codes}",
        flush=True,
    )

    for path, seat, episode_id in episodes:
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        _accumulate_planner(module, planner_totals)
        module.diag_reset()
        totals["episodes"] += 1
        pending_boss: dict[str, Any] | None = None
        pending_shadow: dict[str, Any] | None = None

        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            action = (steps[index + 1][seat] or {}).get("action")
            if not options or not isinstance(action, list) or len(action) != 1:
                continue
            if not isinstance(action[0], int):
                continue
            if not 0 <= action[0] < len(options):
                continue
            current = observation.get("current") or {}
            players = current.get("players") or [{}, {}]
            if len(players) < 2:
                continue
            me, opponent = players[seat], players[1 - seat]
            context = int(select.get("context", -1))

            ranker.teacher_code = context_codes.get(context, baseline_code)
            try:
                answer = agent(observation)
            except Exception as error:  # noqa: BLE001
                totals["agent_exception"] += 1
                totals[f"exception_{type(error).__name__}"] += 1
                answer = None
            chosen = (
                answer[0]
                if isinstance(answer, list) and len(answer) == 1
                and isinstance(answer[0], int)
                and 0 <= answer[0] < len(options)
                else None
            )
            played = action[0]
            totals["decisions"] += 1
            if chosen is None:
                totals["unusable_answer"] += 1
            else:
                totals["agreements"] += int(chosen == played)

            record_decision(
                groups, context, current, select, options, me, opponent,
                played, chosen, episode_id, pending_boss, pending_shadow,
            )
            pending_boss, pending_shadow = update_pending(
                context, current, select, options, me, opponent,
                played, chosen, episode_id, pending_boss, pending_shadow,
            )

            # Teacher forcing: advance the runtime's intra-turn history with
            # the action the replay actually took, never with ours. v3 exposes
            # a module-level hook so the planner's per-turn heal budget follows
            # the teacher too; v2 has only the ranker's.
            if observe_hook is not None:
                observe_hook(observation, played)
            else:
                ranker.observe_external(observation, played)

    _accumulate_planner(module, planner_totals)
    report = {
        "agent_dir": str(args.agent_dir.resolve()),
        "source": str((args.run_dir or args.data_root).resolve()),
        "teams": args.teams or None,
        "pin_team": args.pin_team,
        "pin_context": args.pin_context or None,
        "baseline_code": baseline_code,
        "totals": dict(totals),
        "agreement_with_replay": rate(totals, "agreements", "decisions"),
        "planner": dict(planner_totals),
        "run_outcomes": outcome_summary(
            run_outcomes(args.run_dir, args.submission)
            if args.run_dir else {}
        ),
        "groups": {
            name: summarise(name, probe)
            for name, probe in sorted(groups.items())
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {"totals": report["totals"],
         "agreement": report["agreement_with_replay"],
         "groups": report["groups"]},
        ensure_ascii=False, indent=2,
    ))
    return 0


def _accumulate_planner(module, totals: Counter[str]) -> None:
    snapshot = getattr(module, "diag_snapshot", None)
    if snapshot is None:
        return
    try:
        planner = (snapshot() or {}).get("planner") or {}
    except Exception:
        return
    for key, value in planner.items():
        if isinstance(value, (int, float)):
            totals[key] += int(value)


def summarise(name: str, probe: Probe) -> dict[str, Any]:
    counts = probe.counts
    out: dict[str, Any] = {"counts": dict(counts)}
    if name == "munkidori_source":
        for who in ("played", "agent"):
            out[f"{who}_pass_rate"] = rate(
                counts, f"{who}_passed_grimmsnarl", "grimmsnarl_offered"
            )
        out["offered_wilson95"] = wilson(
            counts["agent_passed_grimmsnarl"], counts["grimmsnarl_offered"]
        )
    elif name == "counter_count":
        for who in ("played", "agent"):
            out[f"{who}_max_rate"] = rate(
                counts, f"{who}_chose_max", "decisions"
            )
    elif name == "snipe_target":
        for who in ("played", "agent"):
            out[f"{who}_ko_rate"] = rate(
                counts, f"{who}_chose_ko", "ko_available"
            )
            out[f"{who}_best_prize_rate"] = rate(
                counts, f"{who}_chose_best_prize_ko", "ko_available"
            )
    elif name == "boss_target":
        for who in ("played", "agent"):
            out[f"{who}_snipe_lethal_rate"] = rate(
                counts, f"{who}_targeted_snipe_lethal", "snipe_lethal_existed"
            )
    elif name == "froslass_evolve":
        for who in ("played", "agent"):
            out[f"{who}_evolve_rate"] = rate(
                counts, f"{who}_evolved", "opportunities"
            )
            out[f"{who}_negative_evolve_rate"] = rate(
                counts, f"{who}_negative_evolved", "negative_opportunities"
            )
    elif name == "boss_play":
        for who in ("played", "agent"):
            out[f"{who}_play_rate"] = rate(
                counts, f"{who}_played_boss", "offered"
            )
            out[f"{who}_play_rate_with_snipe_lethal"] = rate(
                counts, f"{who}_played_boss_with_snipe_lethal",
                "offered_with_snipe_lethal",
            )
    return out


def snipe_lethals(
    opponent: dict[str, Any], current: dict[str, Any], mf
) -> list[dict[str, Any]]:
    stadium = mf._stadium_id(current)
    shield_ids = {int(c.get("id", -1)) for c in board(opponent)}
    return [
        card
        for card in (opponent.get("bench") or [])
        if isinstance(card, dict)
        and mf.bench_snipe_lands(card, stadium, shield_ids)
        and 0 < float(card.get("hp", 0)) <= BENCH_SNIPE_DAMAGE
    ]


def record_decision(
    groups: dict[str, Probe],
    context: int,
    current: dict[str, Any],
    select: dict[str, Any],
    options: list[dict[str, Any]],
    me: dict[str, Any],
    opponent: dict[str, Any],
    played: int,
    chosen: int | None,
    episode_id: int,
    pending_boss: dict[str, Any] | None,
    pending_shadow: dict[str, Any] | None,
) -> None:
    mf = sys.modules["ml_features"]

    if context == CTX_REMOVE_DAMAGE_COUNTER:
        probe = groups["munkidori_source"]
        resolved = [mf.resolve_option(current, select, o) for o in options]
        damaged_grimms = [
            slot for slot, (card, own, _) in enumerate(resolved)
            if card and int(card.get("id", -1)) == GRIMMSNARL_EX_ID
            and float(card.get("maxHp", 0))
            - float(card.get("hp", 0)) >= 30
        ]
        probe.add("decisions")
        if damaged_grimms:
            probe.add("grimmsnarl_offered")
            for who, slot in (("played", played), ("agent", chosen)):
                if slot is None:
                    continue
                card = resolved[slot][0] or {}
                if int(card.get("id", -1)) != GRIMMSNARL_EX_ID:
                    probe.add(f"{who}_passed_grimmsnarl")
            if chosen is not None and played != chosen:
                probe.add("differed")
                if len(probe.examples) < 40:
                    probe.examples.append({
                        "episode_id": episode_id,
                        "turn": int(current.get("turn", -1)),
                        "played": int(
                            (resolved[played][0] or {}).get("id", -1)),
                        "agent": int(
                            (resolved[chosen][0] or {}).get("id", -1)),
                    })

    elif context == CTX_REMOVE_COUNTER_COUNT:
        probe = groups["counter_count"]
        numbers = [
            int(o.get("number")) if isinstance(o.get("number"), (int, float))
            else -1
            for o in options
        ]
        best = max(numbers) if numbers else -1
        probe.add("decisions")
        for who, slot in (("played", played), ("agent", chosen)):
            if slot is not None and numbers[slot] == best:
                probe.add(f"{who}_chose_max")

    elif context == CTX_DAMAGE and pending_shadow is not None:
        probe = groups["snipe_target"]
        resolved = [
            mf.resolve_option(current, select, o)[0] or {}
            for o in options
        ]
        kos = [
            slot for slot, card in enumerate(resolved)
            if 0 < float(card.get("hp", 0)) <= BENCH_SNIPE_DAMAGE
        ]
        probe.add("decisions")
        if kos:
            probe.add("ko_available")
            best_prize = max(
                mf.prize_value(int(resolved[slot].get("id", -1)))
                for slot in kos
            )
            for who, slot in (("played", played), ("agent", chosen)):
                if slot is None:
                    continue
                if slot in kos:
                    probe.add(f"{who}_chose_ko")
                    if mf.prize_value(
                        int(resolved[slot].get("id", -1))
                    ) == best_prize:
                        probe.add(f"{who}_chose_best_prize_ko")

    elif context in (CTX_SWITCH, CTX_TO_ACTIVE) and pending_boss is not None:
        probe = groups["boss_target"]
        resolved = [
            mf.resolve_option(current, select, o)[0] or {}
            for o in options
        ]
        lethal_keys = {
            (int(c.get("id", -1)), float(c.get("hp", 0)))
            for c in pending_boss["snipe_lethals"]
        }
        probe.add("decisions")
        if lethal_keys:
            probe.add("snipe_lethal_existed")
            for who, slot in (("played", played), ("agent", chosen)):
                if slot is None:
                    continue
                key = (
                    int(resolved[slot].get("id", -1)),
                    float(resolved[slot].get("hp", 0)),
                )
                if key in lethal_keys:
                    probe.add(f"{who}_targeted_snipe_lethal")

    elif context == MAIN:
        actions = [mf.action_type(current, o, select) for o in options]
        cards = [
            int((mf.candidate_card(current, o, select) or {}).get("id", -1))
            for o in options
        ]
        froslass_slots = [
            slot for slot, action in enumerate(actions)
            if action == "evolve" and cards[slot] == FROSLASS_ID
        ]
        if froslass_slots:
            probe = groups["froslass_evolve"]
            net = ability_holders(opponent) - ability_holders(me)
            probe.add("opportunities")
            if net < 0:
                probe.add("negative_opportunities")
            for who, slot in (("played", played), ("agent", chosen)):
                if slot is None:
                    continue
                if slot in froslass_slots:
                    probe.add(f"{who}_evolved")
                    if net < 0:
                        probe.add(f"{who}_negative_evolved")

        boss_slots = [
            slot for slot, action in enumerate(actions)
            if action == "boss"
        ]
        if boss_slots:
            probe = groups["boss_play"]
            lethals = snipe_lethals(opponent, current, mf)
            probe.add("offered")
            if lethals:
                probe.add("offered_with_snipe_lethal")
            for who, slot in (("played", played), ("agent", chosen)):
                if slot is None:
                    continue
                if slot in boss_slots:
                    probe.add(f"{who}_played_boss")
                    if lethals:
                        probe.add(f"{who}_played_boss_with_snipe_lethal")


def update_pending(
    context: int,
    current: dict[str, Any],
    select: dict[str, Any],
    options: list[dict[str, Any]],
    me: dict[str, Any],
    opponent: dict[str, Any],
    played: int,
    chosen: int | None,
    episode_id: int,
    pending_boss: dict[str, Any] | None,
    pending_shadow: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Track what the *replay* did, so later selects know their trigger."""
    mf = sys.modules["ml_features"]
    if context == MAIN:
        action = mf.action_type(current, options[played], select)
        if action == "boss":
            pending_boss = {
                "snipe_lethals": snipe_lethals(opponent, current, mf),
            }
        elif action == "attack":
            pending_shadow = {"turn": int(current.get("turn", -1))}
    elif context in (CTX_SWITCH, CTX_TO_ACTIVE):
        pending_boss = None
    elif context == CTX_DAMAGE:
        pending_shadow = None
    return pending_boss, pending_shadow


if __name__ == "__main__":
    raise SystemExit(main())
