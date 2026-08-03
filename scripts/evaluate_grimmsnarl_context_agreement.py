"""Score the shipped agent against teachers on EVERY select context.

v1 imitated MAIN only and reported 91% there, then rated 871 on the ladder
against a 1048-rated teacher. MAIN is roughly 41 decisions per game; the other
contexts are roughly 47, and they are still resolved by the inherited v7 rule
policy, which was never compared against a single top-50 pilot.

In this deck those contexts are not filler. Punk Up alone (search up to five
Basic {D} Energy out of the deck and attach them anywhere) is two nested
selects, and where those five energies land decides the next two turns.

This measures the real agent - the same directory Kaggle loads - on every
context, teacher-forced against the replay, and reports agreement per context
so the largest losses are visible before any model is retrained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402

MAIN_CONTEXT = 0
AREA_NAMES = {2: "hand", 3: "discard", 4: "active", 5: "bench"}


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


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [c for c in (player.get(area) or []) if isinstance(c, dict)]


def _card_id(player: dict[str, Any], area: Any, index: Any) -> int:
    if not isinstance(index, int):
        return -1
    name = AREA_NAMES.get(int(area) if isinstance(area, int) else -1)
    if name is None:
        return -1
    cards = _cards(player, name)
    return int(cards[index].get("id", -1)) if 0 <= index < len(cards) else -1


# Bodies whose Ability prevents all damage from a Pokemon ex, and the Bench
# shields; kept in step with ml_features' generated tables.
EX_BLOCKERS = {345, 330, 117}
BENCH_SHIELDS = {74, 343}


def _wall_tag(
    current: dict[str, Any],
    options: list[dict[str, Any]],
) -> str | None:
    """Label a MAIN decision by whether a Shadow Bullet does anything.

    Only decisions where attacking is legal are labelled, so the block answers
    "when the swing is available and worthless, do we make the same call the
    teacher makes?" rather than mixing in turns with no attacker.
    """
    if not any(
        int(o.get("type", -1)) == 13
        and int(o.get("attackId", -1)) == 937
        for o in options
    ):
        return None
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    opponent = players[1 - your] if 1 - your < len(players) else {}
    active = _cards(opponent, "active")
    if not active:
        return None
    if int(active[0].get("id", -1)) not in EX_BLOCKERS:
        return "wall_absent"
    bench = _cards(opponent, "bench")
    shielded = any(
        int(c.get("id", -1)) in BENCH_SHIELDS
        for c in (active + bench)
    )
    kills = [
        c for c in bench
        if int(c.get("id", -1)) not in EX_BLOCKERS
        and not shielded
        and 0 < int(c.get("hp", 0) or 0) <= 30
    ]
    return "wall_prize_available" if kills else "wall_dead_swing"


def option_identity(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
) -> tuple:
    """Context-agnostic semantic identity of one option.

    Two options with the same identity are interchangeable, so picking either
    counts as agreement. Anything that cannot be resolved to a card falls back
    to the raw fields rather than collapsing into a shared bucket.
    """
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    owner = option.get("playerIndex")
    owner = int(owner) if isinstance(owner, int) and owner in (0, 1) else your
    actor = players[owner] if owner < len(players) else {}

    kind = int(option.get("type", -1))
    area = option.get("area")
    index = option.get("index")
    in_area = option.get("inPlayArea")
    in_index = option.get("inPlayIndex")

    card = _card_id(actor, area, index)
    if card < 0 and (int(area) if isinstance(area, int) else -1) == 1:
        # Area 1 indexes the select's own deck list, not a board zone.
        deck = [c for c in (select.get("deck") or []) if isinstance(c, dict)]
        if isinstance(index, int) and 0 <= index < len(deck):
            card = int(deck[index].get("id", -1))
    target = _card_id(actor, in_area, in_index)
    target_energy = -1
    name = AREA_NAMES.get(int(in_area) if isinstance(in_area, int) else -1)
    if name is not None and isinstance(in_index, int):
        cards = _cards(actor, name)
        if 0 <= in_index < len(cards):
            target_energy = len(
                cards[in_index].get("energyCards")
                or cards[in_index].get("energies") or []
            )
    resolved = (card >= 0) or (target >= 0) or kind in (12, 13, 14)
    return (
        kind,
        owner == your,
        card,
        int(in_area) if isinstance(in_area, int) else -1,
        target,
        target_energy,
        int(option.get("attackId", -1)),
        # Unresolvable options keep their raw slot so they stay distinct.
        -1 if resolved else (area, index),
    )


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
    parser.add_argument("--team", type=int)
    parser.add_argument("--min-episode", type=int, default=0)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument(
        "--episodes-from", type=Path,
        help="wall_attacks.json; scores its wall_episodes block instead. The "
             "pinned pilot's own holdout contained no wall games at all, so "
             "the wall columns can only be measured across pilots.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    index = pd.read_csv(args.data_root / "indexes" / "episodes.csv")
    index = index[index["download_status"] == "success"]
    if args.episodes_from:
        wanted = json.loads(args.episodes_from.read_text(encoding="utf-8"))
        pairs = {
            (int(e["episode_id"]), int(e["seat_index"]))
            for e in wanted.get("wall_episodes", [])
        }
        index = index[[
            (int(e), int(s)) in pairs
            for e, s in zip(index["episode_id"], index["seat_index"])
        ]]
    else:
        if args.team is None:
            raise SystemExit("pass --team or --episodes-from")
        index = index[
            (index["team_id"] == args.team)
            & (index["episode_id"] >= args.min_episode)
        ]
    index = index.drop_duplicates(subset=["episode_id", "seat_index"])
    index = index.sort_values("episode_id")
    if args.limit:
        index = index.head(args.limit)
    if index.empty:
        raise SystemExit("no episodes selected")

    _, _, module = load_dir_agent(args.agent_dir)
    agent = module.agent
    ranker = getattr(module, "_RANKER", None)
    if ranker is not None and hasattr(ranker, "teacher_forced"):
        # Score our answer, but walk the turn along the teacher's line. Without
        # this the agent's own commit and the evaluator's observe_external both
        # fire and the intra-turn columns advance twice per decision, so the
        # model reads features it was never fitted on.
        ranker.teacher_forced = True
    print(f"episodes={len(index)} team={args.team}", flush=True)

    per_context: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    wall_buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    counts: Counter[str] = Counter()
    mismatch_examples: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for _, row in index.iterrows():
        path = args.data_root / "replays" / f"episode_{row.episode_id}.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        seat = int(row.seat_index)
        steps = replay.get("steps") or []
        if hasattr(module, "diag_reset"):
            module.diag_reset()
        if ranker is not None and hasattr(ranker, "teacher_forced"):
            # diag_reset() intentionally restores live-play defaults, so the
            # evaluator must re-enable teacher forcing for every episode.
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
            options = list(select.get("option") or [])
            action = (steps[step_index + 1][seat] or {}).get("action")
            if not options or not isinstance(action, list):
                continue
            if any(
                not isinstance(x, int) or not 0 <= x < len(options)
                for x in action
            ):
                counts["teacher_action_unusable"] += 1
                continue

            context = int(select.get("context", -1))
            minimum = int(select.get("minCount") or 0)
            maximum = int(select.get("maxCount") or 0)
            is_main = (
                context == MAIN_CONTEXT and minimum == 1 and maximum == 1
                and len(options) >= 2
            )
            try:
                answer = agent(observation)
            except Exception:
                counts["agent_exception"] += 1
                continue
            if not isinstance(answer, list) or any(
                not isinstance(x, int) or not 0 <= x < len(options)
                for x in answer
            ):
                counts["agent_illegal"] += 1
                continue

            current = observation.get("current") or {}
            teacher = sorted(
                option_identity(current, select, options[i]) for i in action
            )
            guess = sorted(
                option_identity(current, select, options[i]) for i in answer
            )
            # Only count decisions the teacher could have got wrong: a forced
            # single option is agreement by construction and would inflate
            # every context that is mostly forced.
            distinct = len({
                option_identity(current, select, option) for option in options
            })
            # Area 6 is the face-down prize zone: the observation exposes no
            # card ids, so those picks are blind for the teacher too and no
            # model can beat chance on them. Bucket by area to keep them out
            # of the learnable totals.
            areas = {
                option.get("area") for option in options
                if option.get("area") is not None
            }
            area_tag = (
                f"a{sorted(a for a in areas)[0]}" if len(areas) == 1
                else ("amix" if areas else "a-")
            )
            bucket = (
                "MAIN" if is_main else f"ctx{context}/{area_tag}",
                minimum, maximum,
            )
            if distinct < 2:
                counts["forced"] += 1
                per_context[bucket + ("forced",)][0] += 1
                per_context[bucket + ("forced",)][1] += 1
            else:
                per_context[bucket][0] += 1
                per_context[bucket][1] += int(teacher == guess)
                counts["scored"] += 1
                counts["scored_correct"] += int(teacher == guess)
                # The spots the ladder logs said we were throwing away: the
                # Active prevents all our damage. v1 and the first v2 had no
                # feature for it, so report them as their own block.
                if is_main:
                    tag = _wall_tag(current, options)
                    if tag:
                        wall_buckets[tag][0] += 1
                        wall_buckets[tag][1] += int(teacher == guess)
                        counts[f"attacked_{tag}"] += int(
                            any(
                                int(options[i].get("type", -1)) == 13
                                and int(options[i].get("attackId", -1)) == 937
                                for i in answer
                            )
                        )
                        counts[f"teacher_attacked_{tag}"] += int(
                            any(
                                int(options[i].get("type", -1)) == 13
                                and int(options[i].get("attackId", -1)) == 937
                                for i in action
                            )
                        )
                if teacher != guess and len(mismatch_examples[context]) < 3:
                    mismatch_examples[context].append({
                        "episode": int(row.episode_id),
                        "turn": int(current.get("turn", -1)),
                        "options": len(options),
                        "distinct": distinct,
                        "teacher": [list(map(str, t)) for t in teacher],
                        "agent": [list(map(str, g)) for g in guess],
                    })

            # Advance the turn history for every context the ranker owns, not
            # just MAIN: v2 scores them all, and the corpus builder advanced
            # the same set when it wrote the training columns.
            if ranker is not None and len(action) == 1:
                ranker.observe_external(observation, action[0])

    rows = []
    for bucket, (total, agree) in sorted(
        per_context.items(), key=lambda kv: -kv[1][0]
    ):
        if len(bucket) == 4:
            continue
        rows.append({
            "context": bucket[0],
            "min": bucket[1],
            "max": bucket[2],
            "decisions": total,
            "agreement": round(agree / total, 4),
            "wilson95": wilson(agree, total),
        })

    scored = counts["scored"]
    report = {
        "agent_dir": str(args.agent_dir.resolve()),
        "team": args.team,
        "min_episode": args.min_episode,
        "episodes": counts["episodes"],
        "all_context_decisions": scored,
        "all_context_agreement": round(
            counts["scored_correct"] / scored, 4
        ) if scored else None,
        "all_context_wilson95": wilson(counts["scored_correct"], scored),
        "per_context": rows,
        "wall_spots": {
            tag: {
                "decisions": total,
                "agreement": round(agree / total, 4),
                "wilson95": wilson(agree, total),
                "agent_attacked": counts.get(f"attacked_{tag}", 0),
                "teacher_attacked": counts.get(f"teacher_attacked_{tag}", 0),
            }
            for tag, (total, agree) in sorted(wall_buckets.items())
        },
        "counts": dict(counts),
        "mismatch_examples": {
            str(k): v for k, v in sorted(mismatch_examples.items())
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nall-context agreement: {report['all_context_agreement']} "
          f"(n={scored})")
    print(f"{'context':<10}{'min':>5}{'max':>5}{'n':>8}{'agree':>9}")
    for entry in rows:
        print(f"{entry['context']:<10}{entry['min']:>5}{entry['max']:>5}"
              f"{entry['decisions']:>8}{entry['agreement']:>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
