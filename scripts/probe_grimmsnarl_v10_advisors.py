"""Dump every v8 ladder decision with what each advisor would have played.

The v10 brief asks for a *residual*: v8 stays the policy and only a narrow,
evidence-backed set of boards is overridden. That needs a decision-level table,
not an aggregate agreement number, because the question is never "is advisor X
better" but "on which boards do several independent advisors disagree with v8
in the same direction, repeatedly, in games v8 lost".

Every advisor is run teacher-forced on the identical state and then advanced
with the action the replay actually took, so all of them see the same board at
every step and none of them drifts onto its own distribution.

One row per decision, JSONL. Columns are chosen so a later gate can be written
against them without re-reading 5 MB replays:

* what was on offer (contexts, action types, card ids, option count)
* what v8 played and what each advisor wanted, as index *and* as semantics
* each advisor's score margin between its top two candidates, which is the only
  calibration signal available at inference time
* the per-episode outcome joins so a disagreement can be conditioned on
  "in games we lost, going second, against Alakazam".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402

MAIN = 0


def deck_hash(card_ids: list[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class Advisor:
    def __init__(self, name: str, agent_dir: Path):
        self.name = name
        self.agent, _, self.module = load_dir_agent(agent_dir)
        self.ranker = getattr(self.module, "_RANKER", None)
        if self.ranker is None:
            raise SystemExit(f"{name}: no ranker loaded")
        self.ranker.teacher_forced = True
        self.observe = getattr(self.module, "observe_external", None)

    def ask(self, observation: dict[str, Any], count: int) -> dict[str, Any]:
        try:
            answer = self.agent(observation)
        except Exception as error:  # noqa: BLE001
            return {"chosen": None, "error": type(error).__name__}
        chosen = (
            answer[0]
            if isinstance(answer, list) and len(answer) == 1
            and isinstance(answer[0], int) and 0 <= answer[0] < count
            else None
        )
        scores = dict(getattr(self.ranker, "last_scores", {}) or {})
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = (
            round(ordered[0][1] - ordered[1][1], 6)
            if len(ordered) >= 2 else None
        )
        return {
            "chosen": chosen,
            "scored": bool(scores),
            "margin": margin,
            "rank_of_played": None,  # filled by the caller
            "order": [int(i) for i, _ in ordered[:6]],
        }

    def advance(self, observation: dict[str, Any], played: int) -> None:
        if self.observe is not None:
            self.observe(observation, played)
        else:
            self.ranker.observe_external(observation, played)


class PinAdvisor:
    """The base agent's own trees, scored as a different pilot.

    ``teacher_team_id`` is a categorical input, so a second opinion costs one
    extra pass over the trees and no extra model file. That matters twice: it
    is the only multi-advisor consensus a residual can afford to compute at
    inference time inside a Kaggle submission, and it keeps every advisor on
    the identical feature space, so a disagreement is a policy difference
    rather than a difference in what the two models can see.

    It reads the *ranker's* argmax, not the agent's final action: the planner
    and rule shell above it belong to the base policy and are applied once.
    """

    def __init__(self, name: str, host: Advisor, code: int):
        self.name = name
        self.host = host
        self.code = code

    def ask(self, observation: dict[str, Any], count: int) -> dict[str, Any]:
        ranker = self.host.ranker
        select = observation.get("select") or {}
        if not ranker.is_scorable(select):
            return {"chosen": None, "scored": False, "margin": None,
                    "order": [], "not_scorable": True}
        try:
            features, representatives = ranker._rows(observation)
            ranker._turn_state(observation, features)
            if len(representatives) < 2:
                return {"chosen": None, "scored": False, "margin": None,
                        "order": [], "collapsed": True}
            best, scores = ranker._score(features, representatives, self.code)
        except Exception as error:  # noqa: BLE001
            return {"chosen": None, "scored": False, "margin": None,
                    "order": [], "error": type(error).__name__}
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return {
            "chosen": int(best),
            "scored": True,
            "margin": (
                round(ordered[0][1] - ordered[1][1], 6)
                if len(ordered) >= 2 else None
            ),
            "order": [int(i) for i, _ in ordered[:6]],
        }

    def advance(self, observation: dict[str, Any], played: int) -> None:
        return  # the host advances the shared history exactly once


def episode_rows(run_dir: Path, submission: str) -> list[dict[str, Any]]:
    rows = []
    for raw in csv.DictReader(
        (run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
            continue
        episode_id = int(raw["episode_id"])
        path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        seat = 0 if a0 == submission else 1
        opponent_raw = (
            raw.get(f"agent_{1 - seat}_initial_score") or ""
        ).strip()
        rows.append({
            "episode_id": episode_id,
            "create_time": raw["create_time"],
            "seat": seat,
            "path": path,
            "opponent_score": float(opponent_raw) if opponent_raw else None,
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
    parser.add_argument(
        "--advisor", action="append", default=[],
        help="name=path, repeatable. The first one is the base policy.",
    )
    parser.add_argument(
        "--pin", action="append", default=[],
        help="name=team_id, repeatable. Scores the first advisor's trees as "
             "that pilot; costs one extra tree pass and no extra model.",
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "ml" / "grimmsnarl" / "processed"
        / "corpus_v5_data_refresh_candidate.npz",
        help="Only read to translate team ids into the dense pin codes.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--only-context", type=int, action="append", default=[],
        help="Score only these contexts. Every decision is still walked and "
             "every advisor still advanced with the played action, so the "
             "intra-turn history is identical to a full run - this only "
             "skips the tree walks that would be thrown away.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.advisor:
        args.advisor = [
            f"v8={ROOT / 'agents' / 'grimmsnarl' / 'grimmsnarl_ml_v8'}",
        ]
    advisors: list[Any] = [
        Advisor(spec.split("=", 1)[0], Path(spec.split("=", 1)[1]))
        for spec in args.advisor
    ]
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
            advisors.append(
                PinAdvisor(name, advisors[0], codes[int(team)])
            )
    mf = sys.modules["ml_features"]
    wanted = {int(c) for c in args.only_context}

    episodes = episode_rows(args.run_dir, args.submission)
    if args.limit:
        episodes = episodes[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for order, meta in enumerate(episodes):
            replay = json.loads(meta["path"].read_text(encoding="utf-8"))
            steps = replay.get("steps") or []
            seat = meta["seat"]
            rewards = replay.get("rewards") or [None, None]
            other = rewards[1 - seat]
            won = bool(rewards[seat] > (other if other is not None else 0))
            decks: list[list[int] | None] = [None, None]
            if len(steps) > 1:
                for s in (0, 1):
                    action = (steps[1][s] or {}).get("action")
                    if isinstance(action, list) and len(action) == 60:
                        decks[s] = [int(v) for v in action]
            went_first = None
            for step in reversed(steps):
                if seat >= len(step):
                    continue
                current = (
                    (step[seat] or {}).get("observation") or {}
                ).get("current")
                if isinstance(current, dict) and current.get("players"):
                    first = int(current.get("firstPlayer", -1))
                    went_first = (first == seat) if first >= 0 else None
                    break
            episode_meta = {
                "episode_id": meta["episode_id"],
                "episode_order": order,
                "seat": seat,
                "won": won,
                "went_first": went_first,
                "opponent_score": meta["opponent_score"],
                "opponent_deck_hash": (
                    deck_hash(decks[1 - seat]) if decks[1 - seat] else ""
                ),
            }
            for advisor in advisors:
                if isinstance(advisor, Advisor):
                    advisor.module.diag_reset()

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
                if not options or not isinstance(action, list):
                    continue
                if len(action) != 1 or not isinstance(action[0], int):
                    continue
                if not 0 <= action[0] < len(options):
                    continue
                current = observation.get("current") or {}
                players = current.get("players") or []
                if len(players) < 2:
                    continue
                played = action[0]
                context = int(select.get("context", -1))

                try:
                    actions = [
                        mf.action_type(current, o, select) for o in options
                    ]
                    cards = [
                        int((mf.candidate_card(current, o, select) or {})
                            .get("id", -1))
                        for o in options
                    ]
                    targets = [
                        int((mf.candidate_target(current, o) or {})
                            .get("id", -1))
                        for o in options
                    ]
                except Exception:  # noqa: BLE001
                    actions = ["?"] * len(options)
                    cards = [-1] * len(options)
                    targets = [-1] * len(options)

                row = {
                    **episode_meta,
                    "step": index,
                    "turn": int(current.get("turn", -1)),
                    "context": context,
                    "min_count": int(select.get("minCount") or 0),
                    "max_count": int(select.get("maxCount") or 0),
                    "options": len(options),
                    "prizes_left": [
                        len(p.get("prize") or []) for p in players[:2]
                    ],
                    "played": played,
                    "played_action": actions[played],
                    "played_card": cards[played],
                    "played_target": targets[played],
                    "offered_actions": sorted(set(actions)),
                    "advisors": {},
                }
                if wanted and context not in wanted:
                    for advisor in advisors:
                        advisor.advance(observation, played)
                    continue
                for advisor in advisors:
                    result = advisor.ask(observation, len(options))
                    slot = result["chosen"]
                    result["action"] = (
                        actions[slot] if slot is not None else None
                    )
                    result["card"] = cards[slot] if slot is not None else None
                    result["target"] = (
                        targets[slot] if slot is not None else None
                    )
                    result["agrees"] = (
                        None if slot is None else bool(slot == played)
                    )
                    result["order"] = result.get("order") or []
                    result["rank_of_played"] = (
                        result["order"].index(played)
                        if played in result["order"] else None
                    )
                    row["advisors"][advisor.name] = result
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

                for advisor in advisors:
                    advisor.advance(observation, played)
            print(
                f"{meta['episode_id']} rows={written}", flush=True,
            )
    print(f"wrote {written} decisions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
