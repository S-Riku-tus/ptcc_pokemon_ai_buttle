"""Which cards do we play more or less often than the teachers, and where.

Context-level agreement says the main phase is our weakest slice.  It does not
say *which* card the disagreement is about.  This walks held-out teacher
episodes, forces the agent onto the trajectory the teacher actually played so
its intra-turn history stays aligned, and at every decision records both what
the teacher chose and what the agent would have chosen.

Two tables come out of it:

* offer/take - for every card, how often it was on offer and how often each
  side took it.  A card we take at half the teacher's rate is a concrete,
  addressable defect; a whole-context agreement number hides it.
* confusion - when we disagree, what we play instead.

Usage:
  python scripts/analyze_dragapult_play_gap.py \
      --agent-dir agents/dragapult/dragapult_ml_v2 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --split-report experiments/dragapult_ml_v2/train_full.json \
      --report experiments/dragapult_ml_v2/play_gap_v2.json
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

# Card names carry accented characters that the Windows console codec rejects.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_loader import load_dir_agent_module  # noqa: E402

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
ATTACKS = {
    int(attack["attackId"]): attack
    for attack in json.loads(
        (ROOT / "vendor" / "cg" / "attacks.json").read_text(encoding="utf-8")
    )
}

MAIN = 0
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE = 7, 8, 9
OPT_ABILITY, OPT_RETREAT, OPT_ATTACK, OPT_END = 10, 12, 13, 14


def name(card_id: int) -> str:
    return str(CARDS.get(int(card_id), {}).get("name") or card_id)


def label(observation: dict[str, Any], option: dict[str, Any]) -> str:
    """A stable, human-readable identity for a main-phase option."""
    option_type = int(option.get("type", -1))
    if option_type == OPT_END:
        return "END"
    if option_type == OPT_RETREAT:
        return "RETREAT"
    if option_type == OPT_ATTACK:
        attack_id = int(option.get("attackId", -1))
        return f"ATTACK:{ATTACKS.get(attack_id, {}).get('name', attack_id)}"

    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    index = int(option.get("index", -1))

    if option_type == OPT_ABILITY:
        area = int(option.get("area", -1))
        zone = {
            2: mine.get("hand"), 3: mine.get("discard"), 4: mine.get("active"),
            5: mine.get("bench"), 7: current.get("stadium"),
        }.get(area) or []
        card = zone[index] if isinstance(zone, list) and 0 <= index < len(zone) else {}
        return f"ABILITY:{name(int((card or {}).get('id', -1)))}"

    hand = mine.get("hand") or []
    card = hand[index] if 0 <= index < len(hand) else {}
    prefix = {OPT_PLAY: "PLAY", OPT_ATTACH: "ATTACH", OPT_EVOLVE: "EVOLVE"}.get(
        option_type, f"OPT{option_type}"
    )
    return f"{prefix}:{name(int((card or {}).get('id', -1)))}"


def choice_of(module: Any, observation: dict[str, Any]) -> int | None:
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        picked = list(module._fallback_agent(observation))
        return picked[0] if len(picked) == 1 else None
    index = ranker.choose(observation)
    if index is None:
        picked = list(module._fallback_agent(observation))
        return picked[0] if len(picked) == 1 else None
    guarded = getattr(module, "_guarded_index", None)
    if guarded is not None:
        replacement = guarded(observation, index)
        if replacement is not None:
            index = replacement
    return index


def walk(module: Any, replay: dict[str, Any], seat: int, sink: dict[str, Any]) -> None:
    steps = replay.get("steps") or []
    module.diag_reset()
    for step_index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        select = observation.get("select")
        actual = (
            steps[step_index + 1][seat].get("action")
            if step_index + 1 < len(steps) else None
        )
        if select is None:
            ranker = getattr(module, "_RANKER", None)
            if ranker is not None:
                ranker.reset()
            module._fallback_agent(observation)
            continue
        if not isinstance(actual, list) or len(actual) != 1:
            continue
        options = select.get("option") or []
        played = int(actual[0])
        if not 0 <= played < len(options):
            continue

        ours = choice_of(module, observation)
        ranker = getattr(module, "_RANKER", None)
        if ranker is not None:
            ranker.observe_external(observation, played)

        if int(select.get("context", -1)) != MAIN:
            continue
        offered = {label(observation, option) for option in options}
        teacher_label = label(observation, options[played])
        our_label = (
            label(observation, options[ours])
            if ours is not None and 0 <= ours < len(options) else "NONE"
        )
        for key in offered:
            sink["offers"][key] += 1
        sink["teacher_takes"][teacher_label] += 1
        sink["our_takes"][our_label] += 1
        sink["decisions"] += 1
        if teacher_label == our_label:
            sink["agree"] += 1
        else:
            sink["confusion"][(teacher_label, our_label)] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--split-report", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    module = load_dir_agent_module(args.agent_dir.resolve())
    boundaries: dict[str, list[int]] = {}
    if args.split_report:
        boundaries = json.loads(
            args.split_report.read_text(encoding="utf-8")
        ).get("split_boundaries") or {}

    sink: dict[str, Any] = {
        "offers": Counter(), "teacher_takes": Counter(), "our_takes": Counter(),
        "confusion": Counter(), "decisions": 0, "agree": 0,
    }
    seen: set[tuple[str, int]] = set()
    episodes = 0
    for row in csv.DictReader(
        args.teacher_index.read_text(encoding="utf-8-sig").splitlines()
    ):
        episode_id = str(row["episode_id"])
        seat = int(row["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        boundary = boundaries.get(str(row.get("team_id")))
        if boundary:
            low, high = int(boundary[0]), int(boundary[1])
            in_split = (
                int(episode_id) > high if args.split == "test"
                else low < int(episode_id) <= high if args.split == "validation"
                else int(episode_id) <= low
            )
            if not in_split:
                continue
        path = Path(row["replay_path"])
        if not path.is_absolute():
            path = args.teacher_index.parent.parent / path
        if not path.exists():
            continue
        walk(module, json.loads(path.read_text(encoding="utf-8")), seat, sink)
        episodes += 1
        if args.limit and episodes >= args.limit:
            break

    rows = []
    for key, offers in sink["offers"].most_common():
        teacher = sink["teacher_takes"][key]
        ours = sink["our_takes"][key]
        rows.append({
            "option": key,
            "offers": offers,
            "teacher_takes": teacher,
            "our_takes": ours,
            "teacher_rate": round(teacher / offers, 4),
            "our_rate": round(ours / offers, 4),
            "delta": round((ours - teacher) / offers, 4),
        })
    report = {
        "agent_dir": str(args.agent_dir),
        "split": args.split,
        "episodes": episodes,
        "main_decisions": sink["decisions"],
        "main_agreement": round(sink["agree"] / max(1, sink["decisions"]), 4),
        "offer_take": rows,
        "confusion": [
            {"teacher": teacher, "ours": ours, "count": count}
            for (teacher, ours), count in sink["confusion"].most_common(40)
        ],
    }
    print(f"episodes {episodes}  MAIN decisions {sink['decisions']}  "
          f"agreement {report['main_agreement']}")
    print(f"\n{'option':38} {'offers':>7} {'teach':>7} {'ours':>7} "
          f"{'t_rate':>7} {'o_rate':>7} {'delta':>7}")
    for row in rows:
        if row["offers"] < 40:
            continue
        print(f"{row['option'][:38]:38} {row['offers']:>7} "
              f"{row['teacher_takes']:>7} {row['our_takes']:>7} "
              f"{row['teacher_rate']:>7.3f} {row['our_rate']:>7.3f} "
              f"{row['delta']:>+7.3f}")
    print("\ntop confusions (teacher -> ours):")
    for item in report["confusion"][:20]:
        print(f"  {item['count']:>5}  {item['teacher'][:34]:34} -> {item['ours'][:34]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
