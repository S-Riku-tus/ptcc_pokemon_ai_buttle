"""Stage 1: where v8's early game diverges from the field's, against Alakazam.

The tempo table says that going second into Alakazam the field attacks on turn
4 with 5.2 bodies down and we attack on turn 8 with 3.6 - but it says it from
*five* of our games, which is enough to prove the hole exists and nowhere near
enough to say what makes it. This gets the power from the other side: run v8
teacher-forced through the **field's** 162 Alakazam-going-second replays and
measure, decision by decision, where v8 would have played something else.

That turns n=5 outcomes into thousands of decisions with a known-good answer
attached to each one, and it separates the two hypotheses that the outcome data
cannot:

* **Decision defect.** v8 disagrees with the field more in these positions than
  in the control cohorts. Then the disagreements name the fix.
* **State divergence.** v8 agrees with the field on the field's boards, and the
  gap is compounding - we never reach those boards in the first place. Then no
  single-decision override can fix it and Stage 1 has to become a setup-line
  change.

Cohorts, all restricted to the same 60-card deck hash and all read with
``firstPlayer`` (never the seat index):

* ``alakazam_second``   - the matchup and seat that is 0-5 for us, 114-48 for them
* ``alakazam_first``    - same matchup, the seat where we are normal (p = 0.62)
* ``other_second``      - every non-Alakazam opponent, going second

``alakazam_first`` is the within-matchup control and ``other_second`` the
within-seat one; a divergence that is specific to the broken cell has to be
elevated against both. Every row carries its pilot's team id so the comparison
can be re-run pilot-matched, because the cohorts do not draw the same pilots.

Only turns up to ``--max-turn`` are walked (default 8, i.e. our first four turns
on the draw). The tempo gap is entirely inside that window and stopping there is
what makes three cohorts affordable.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"


class Pin:
    """The host's own trees scored as a different pilot.

    ``teacher_team_id`` is a categorical input, so asking "what would pilot X
    play here" costs one extra pass over the trees and no extra model file.
    That is what makes a matchup-conditioned re-pin cheap enough to ship, and
    it is the only way to tell a *representable* policy difference from one the
    ranker cannot express: every pin sees the identical features.

    Reads the ranker's argmax, not the agent's action - the planner and safety
    shell above it belong to the base policy and are applied once.
    """

    def __init__(self, name: str, host: "Probe", code: int):
        self.name = name
        self.host = host
        self.code = code

    def ask(self, observation: dict[str, Any], count: int) -> int | None:
        ranker = self.host.ranker
        select = observation.get("select") or {}
        if not ranker.is_scorable(select):
            return None
        try:
            features, representatives = ranker._rows(observation)
            ranker._turn_state(observation, features)
            if len(representatives) < 2:
                return None
            best, _ = ranker._score(features, representatives, self.code)
        except Exception:  # noqa: BLE001
            return None
        return int(best) if 0 <= int(best) < count else None


class Probe:
    """v8, teacher-forced, advanced with the action the replay actually took."""

    def __init__(self, agent_dir: Path):
        self.agent, _, self.module = load_dir_agent(agent_dir)
        self.ranker = getattr(self.module, "_RANKER", None)
        if self.ranker is None:
            raise SystemExit(f"{agent_dir}: no ranker loaded")
        self.ranker.teacher_forced = True
        self.observe = getattr(self.module, "observe_external", None)

    def reset(self) -> None:
        self.module.diag_reset()

    def ask(self, observation: dict[str, Any], count: int) -> int | None:
        try:
            answer = self.agent(observation)
        except Exception:  # noqa: BLE001
            return None
        if (
            isinstance(answer, list) and len(answer) == 1
            and isinstance(answer[0], int) and 0 <= answer[0] < count
        ):
            return answer[0]
        return None

    def advance(self, observation: dict[str, Any], played: int) -> None:
        if self.observe is not None:
            self.observe(observation, played)
        else:
            self.ranker.observe_external(observation, played)


def replay_meta(
    replay: dict[str, Any], seat: int
) -> dict[str, Any] | None:
    """Deck hash check, opponent family and turn order, from one parse."""
    steps = replay.get("steps") or []
    decks: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for s in (0, 1):
            action = (steps[1][s] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[s] = [int(v) for v in action]
    if decks[seat] is None or deck_hash(decks[seat]) != OUR_DECK_HASH:
        return None
    if decks[1 - seat] is None:
        return None
    went_first = None
    for step in reversed(steps):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(current, dict) and current.get("players"):
            first = int(current.get("firstPlayer", -1))
            went_first = (first == seat) if first >= 0 else None
            break
    rewards = replay.get("rewards") or [None, None]
    if rewards[seat] is None:
        return None
    other = rewards[1 - seat]
    return {
        "won": bool(rewards[seat] > (other if other is not None else 0)),
        "went_first": went_first,
        "opponent_family": family(decks[1 - seat]),
        "opponent_deck_hash": deck_hash(decks[1 - seat]),
    }


def cohort_of(meta: dict[str, Any]) -> str | None:
    alakazam = meta["opponent_family"] == "Alakazam"
    if meta["went_first"] is None:
        return None
    if alakazam:
        return "alakazam_first" if meta["went_first"] else "alakazam_second"
    return None if meta["went_first"] else "other_second"


def walk(
    probe: Probe,
    replay: dict[str, Any],
    seat: int,
    meta: dict[str, Any],
    max_turn: int,
    mf: Any,
    pins: list[Pin] | None = None,
) -> list[dict[str, Any]]:
    """One row per decision the pilot made, up to ``max_turn``."""
    steps = replay.get("steps") or []
    probe.reset()
    rows: list[dict[str, Any]] = []
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
        if not isinstance(action[0], int) or not 0 <= action[0] < len(options):
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) < 2:
            continue
        turn = int(current.get("turn", -1))
        if turn > max_turn:
            break
        played = action[0]

        try:
            actions = [mf.action_type(current, o, select) for o in options]
            cards = [
                int((mf.candidate_card(current, o, select) or {}).get("id", -1))
                for o in options
            ]
        except Exception:  # noqa: BLE001
            actions = ["?"] * len(options)
            cards = [-1] * len(options)

        chosen = probe.ask(observation, len(options))
        row = {
            **meta,
            "turn": turn,
            "context": int(select.get("context", -1)),
            "options": len(options),
            "bodies": len(mf._in_play(players[seat])),
            "hand": len(players[seat].get("hand") or []),
            "played_action": actions[played],
            "played_card": cards[played],
            "v8_action": actions[chosen] if chosen is not None else None,
            "v8_card": cards[chosen] if chosen is not None else None,
            "agrees": None if chosen is None else bool(chosen == played),
        }
        # Every pin is asked on the *same* board, before the history advances,
        # so a disagreement between two pins is a policy difference and never
        # a difference in what they were allowed to see.
        for pin in pins or ():
            slot = pin.ask(observation, len(options))
            row[f"pin_{pin.name}"] = (
                None if slot is None else bool(slot == played)
            )
            row[f"pin_{pin.name}_action"] = (
                actions[slot] if slot is not None else None
            )
            row[f"pin_{pin.name}_card"] = (
                cards[slot] if slot is not None else None
            )
        rows.append(row)
        probe.advance(observation, played)
    return rows


def rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["agrees"] is not None]
    agree = sum(int(r["agrees"]) for r in scored)
    return {
        "decisions": len(rows),
        "scored": len(scored),
        "agree": agree,
        "top1": round(agree / len(scored), 4) if scored else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--agent", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8",
    )
    parser.add_argument("--max-turn", type=int, default=8)
    parser.add_argument(
        "--control-cap", type=int, default=180,
        help="Replays kept per control cohort; the broken cell is not "
             "capped.",
    )
    parser.add_argument(
        "--skip-cohort", action="append", default=[],
        help="Cohort name to drop entirely, repeatable.",
    )
    parser.add_argument(
        "--pin", action="append", default=[],
        help="name=team_id, repeatable. Scores the agent's own trees as that "
             "pilot; one extra tree pass, no extra model file.",
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "ml" / "grimmsnarl" / "processed"
        / "corpus_v5_data_refresh_candidate.npz",
        help="Only read to translate team ids into the dense pin codes.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    # Index pass: classify every same-deck replay before loading the model, so
    # the control cohorts can be sampled instead of walked and thrown away.
    catalogue: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id, seat = int(raw["episode_id"]), int(raw["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = args.data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        catalogue.append({
            "episode_id": episode_id, "seat": seat, "path": path,
            "team": int(raw["team_id"]),
        })
    catalogue.sort(key=lambda r: (r["episode_id"], r["seat"]))

    probe = Probe(args.agent)
    mf = sys.modules["ml_features"]

    pins: list[Pin] = []
    if args.pin:
        import numpy as np

        data = np.load(args.corpus, allow_pickle=False)
        codes = {
            team: index
            for index, team in enumerate(
                sorted({int(x) for x in data["team_ids"]})
            )
        }
        for spec in args.pin:
            name, team = spec.split("=", 1)
            pins.append(Pin(name, probe, codes[int(team)]))
        print("pins: " + ", ".join(f"{p.name}=code {p.code}" for p in pins))

    skip = set(args.skip_cohort)
    kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    walked = Counter()
    with args.rows.open("w", encoding="utf-8") as handle:
        for entry in catalogue:
            if args.limit and written and walked.total() >= args.limit:
                break
            try:
                replay = json.loads(entry["path"].read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            meta = replay_meta(replay, entry["seat"])
            if meta is None:
                continue
            cohort = cohort_of(meta)
            if cohort is None or cohort in skip:
                continue
            if (
                cohort != "alakazam_second"
                and walked[cohort] >= args.control_cap
            ):
                continue
            meta = {
                **meta, "cohort": cohort, "team": entry["team"],
                "episode_id": entry["episode_id"],
            }
            rows = walk(
                probe, replay, entry["seat"], meta, args.max_turn, mf, pins
            )
            walked[cohort] += 1
            kept[cohort].extend(rows)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += len(rows)
            print(
                f"{entry['episode_id']} {cohort:16s} rows={len(rows):3d} "
                f"total={written} walked={dict(walked)}",
                flush=True,
            )

    report: dict[str, Any] = {
        "agent": str(args.agent),
        "max_turn": args.max_turn,
        "games": dict(walked),
        "overall": {},
        "by_turn": {},
        "by_context": {},
        "substitutions": {},
        "played_but_v8_would_not": {},
    }
    for cohort, rows in sorted(kept.items()):
        report["overall"][cohort] = rate(rows)
        report["by_turn"][cohort] = {
            str(turn): rate([r for r in rows if r["turn"] == turn])
            for turn in sorted({r["turn"] for r in rows})
        }
        report["by_context"][cohort] = {
            str(context): rate([r for r in rows if r["context"] == context])
            for context in sorted({r["context"] for r in rows})
        }
        disagreed = [r for r in rows if r["agrees"] is False]
        report["substitutions"][cohort] = [
            {"pilot": pilot, "v8": ours, "n": count}
            for (pilot, ours), count in Counter(
                (r["played_action"], r["v8_action"]) for r in disagreed
            ).most_common(15)
        ]
        # Per own turn: what fraction of the pilot's turns contain an action
        # v8 would not have taken, by the pilot's action type. This is the
        # form the tempo gap would show up in - a play we skip every turn.
        turns = defaultdict(set)
        for row in rows:
            turns[row["episode_id"]].add(row["turn"])
        own_turns = sum(len(v) for v in turns.values()) or 1
        report["played_but_v8_would_not"][cohort] = {
            action: round(count / own_turns, 4)
            for action, count in Counter(
                r["played_action"] for r in disagreed
            ).most_common(15)
        }
        report["played_but_v8_would_not"][cohort]["_own_turns"] = own_turns

    # Bodies in play at each turn, from the same walk, for both cohorts: the
    # tempo table's headline number recomputed on the decisions actually scored.
    report["bodies_by_turn"] = {
        cohort: {
            str(turn): round(statistics.fmean(values), 2)
            for turn, values in sorted((
                (t, [r["bodies"] for r in rows if r["turn"] == t])
                for t in sorted({r["turn"] for r in rows})
            ))
        }
        for cohort, rows in sorted(kept.items())
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"{'cohort':18s} {'games':>6} {'decisions':>10} {'v8 top-1':>9}")
    for cohort, block in report["overall"].items():
        print(f"{cohort:18s} {walked[cohort]:6d} {block['scored']:10d} "
              f"{block['top1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
