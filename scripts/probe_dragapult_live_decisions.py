"""Attribute every live Dragapult decision to the component that produced it.

Aggregate agreement numbers cannot say whether a bad live action came from the
learned ranker, from the deterministic guard, or from the fallback that owns the
unrouted contexts.  This walks a downloaded run in order, reproduces the exact
submitted decision path, and labels each decision with its owner.

Usage:
  python scripts/probe_dragapult_live_decisions.py \
      data/submissions/submission_55545828_dragapult_v1 \
      --agent-dir agents/dragapult/dragapult_ml_v1 \
      --report experiments/dragapult_ml_v1/live_decision_attribution.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

DREEPY, DRAKLOAK, DRAGAPULT, MUNKIDORI = 119, 120, 121, 112
FIRE, PSYCHIC, DARK = 2, 5, 7
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_RETREAT, OPT_ATTACK, OPT_END = 7, 8, 9, 12, 13, 14


def name(card_id: int | None) -> str:
    if card_id is None or card_id < 0:
        return "-"
    return str(CARDS.get(int(card_id), {}).get("name") or card_id)


def option_label(observation: dict[str, Any], option: dict[str, Any]) -> str:
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    option_type = int(option.get("type", -1))
    player = int(option.get("playerIndex", your))
    owner = players[player] if player in (0, 1) else {}
    area = int(option.get("area", -1))
    index = int(option.get("index", -1))
    if option_type in (OPT_PLAY, OPT_ATTACH, OPT_EVOLVE):
        area, owner = 2, players[your]
    zones = {
        1: select.get("deck") or [],
        2: owner.get("hand") or [],
        3: owner.get("discard") or [],
        4: owner.get("active") or [],
        5: owner.get("bench") or [],
        6: owner.get("prize") or [],
        7: current.get("stadium") or [],
        12: current.get("looking") or [],
    }
    zone = zones.get(area) or []
    card = zone[index] if isinstance(zone, list) and 0 <= index < len(zone) else None
    card_id = int(card.get("id", -1)) if isinstance(card, dict) else -1
    target_area = int(option.get("inPlayArea", -1))
    target_index = int(option.get("inPlayIndex", -1))
    target_zone = (
        (players[your].get("active") or []) if target_area == 4
        else (players[your].get("bench") or []) if target_area == 5 else []
    )
    target = (
        target_zone[target_index]
        if isinstance(target_zone, list) and 0 <= target_index < len(target_zone)
        else None
    )
    if option_type == OPT_ATTACK:
        attack_id = int(option.get("attackId", -1))
        return f"attack:{ATTACKS.get(attack_id, {}).get('name', attack_id)}"
    if option_type == OPT_END:
        return "end"
    if option_type == OPT_RETREAT:
        return "retreat"
    if option_type in (OPT_ATTACH, OPT_EVOLVE):
        kind = "attach" if option_type == OPT_ATTACH else "evolve"
        energies = [int(v) for v in (target or {}).get("energies") or []]
        target_id = int((target or {}).get("id", -1))
        return f"{kind}:{name(card_id)}->{name(target_id)}{energies}"
    if option_type == OPT_PLAY:
        return f"play:{name(card_id)}"
    return f"opt{option_type}:{name(card_id)}"


def attach_facts(observation: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    if int(option.get("type", -1)) != OPT_ATTACH:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    hand = mine.get("hand") or []
    index = int(option.get("index", -1))
    if not 0 <= index < len(hand) or not isinstance(hand[index], dict):
        return None
    source = int(hand[index].get("id", -1))
    area = int(option.get("inPlayArea", -1))
    target_index = int(option.get("inPlayIndex", -1))
    zone = mine.get("active") if area == 4 else mine.get("bench") if area == 5 else []
    if not isinstance(zone, list) or not 0 <= target_index < len(zone):
        return None
    target = zone[target_index]
    if not isinstance(target, dict):
        return None
    energies = [int(value) for value in target.get("energies") or []]
    target_id = int(target.get("id", -1))
    return {
        "source": source,
        "target_id": target_id,
        "energies": energies,
        "route": source in (FIRE, PSYCHIC) and target_id in (DREEPY, DRAKLOAK, DRAGAPULT),
        "duplicate": source in energies,
    }


def walk(module: Any, replay: dict[str, Any], seat: int) -> list[dict[str, Any]]:
    steps = replay.get("steps") or []
    module.diag_reset()
    out: list[dict[str, Any]] = []
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
            if module._RANKER is not None:
                module._RANKER.reset()
            module._fallback_agent(observation)
            continue
        if not isinstance(actual, list) or not actual or len(actual) == 60:
            continue

        current = observation.get("current") or {}
        options = select.get("option") or []
        fallback = list(module._fallback_agent(observation))
        ml_index = module._RANKER.choose(observation) if module._RANKER is not None else None
        external = fallback[0] if len(fallback) == 1 else None
        reason = None
        # v1.0 as submitted has no guard at all; v1.1 and v2 do. Probing the
        # submitted bundle must not require the attribute to exist.
        guard_reason = getattr(module, "_guard_reason", None)
        if ml_index is not None and guard_reason is not None:
            reason = guard_reason(observation, ml_index, external)
        if ml_index is None:
            owner = "fallback"
            final = fallback
            if external is not None and module._RANKER is not None and not module._RANKER.teacher_forced:
                module._RANKER.observe_external(observation, external)
        elif reason is not None and external is not None:
            owner = f"guard:{reason}"
            final = fallback
            module._RANKER.commit(external)
        else:
            owner = "ranker"
            final = [ml_index]
            module._RANKER.commit(ml_index)

        chosen = final[0] if final else None
        record = {
            "step": step_index,
            "turn": int(current.get("turn") or 0),
            "context": int(select.get("context", -1)),
            "min": int(select.get("minCount") or 0),
            "max": int(select.get("maxCount") or 0),
            "options": len(options),
            "owner": owner,
            "chosen": final,
            "actual": actual,
            "reproduced": final == actual,
            "label": (
                option_label(observation, options[chosen])
                if chosen is not None and 0 <= chosen < len(options) else "-"
            ),
        }
        if chosen is not None and 0 <= chosen < len(options):
            facts = attach_facts(observation, options[chosen])
            if facts:
                record["attach"] = facts
                alternatives = []
                for position, option in enumerate(options):
                    other = attach_facts(observation, option)
                    if other and not other["duplicate"] and other["route"]:
                        alternatives.append(position)
                record["useful_attach_available"] = alternatives
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    module = load_dir_agent_module(args.agent_dir.resolve())
    rows = list(csv.DictReader(
        (args.run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
    ))
    games: list[dict[str, Any]] = []
    owner_counts: Counter[str] = Counter()
    owner_by_context: dict[str, Counter[int]] = defaultdict(Counter)
    duplicates: list[dict[str, Any]] = []
    reproduced = mismatch = 0

    for row in rows:
        episode_id = str(row["episode_id"])
        seat = int(row["detected_submission_agent_index"])
        path = args.run / "episodes" / episode_id / "replay" / f"episode_{episode_id}.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        rewards = replay.get("rewards") or [0, 0]
        decisions = walk(module, replay, seat)
        for decision in decisions:
            owner_counts[decision["owner"]] += 1
            owner_by_context[decision["owner"]][decision["context"]] += 1
            reproduced += int(decision["reproduced"])
            mismatch += int(not decision["reproduced"])
            attach = decision.get("attach")
            if attach and attach["duplicate"]:
                duplicates.append({
                    "episode": episode_id,
                    "turn": decision["turn"],
                    "owner": decision["owner"],
                    "label": decision["label"],
                    "useful_alternative": bool(decision.get("useful_attach_available")),
                    "reproduced": decision["reproduced"],
                })
        games.append({
            "episode_id": int(episode_id),
            "seat": seat,
            "result": (
                "win" if rewards[seat] > rewards[1 - seat]
                else "loss" if rewards[seat] < rewards[1 - seat] else "draw"
            ),
            "decisions": len(decisions),
            "reproduced": sum(d["reproduced"] for d in decisions),
            "rows": decisions,
        })

    total = reproduced + mismatch
    summary = {
        "games": len(games),
        "decisions": total,
        "reproduced": reproduced,
        "reproduction_rate": round(reproduced / total, 4) if total else 0.0,
        "owners": dict(owner_counts.most_common()),
        "owner_share": {
            key: round(value / total, 4) for key, value in owner_counts.most_common()
        } if total else {},
        "duplicate_attachments": len(duplicates),
        "duplicate_by_owner": dict(Counter(item["owner"] for item in duplicates)),
        "duplicate_with_useful_alternative": sum(
            item["useful_alternative"] for item in duplicates
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nowner x context:")
    for owner, counter in owner_by_context.items():
        print(f"  {owner:24} {dict(counter.most_common(8))}")
    print("\nduplicate attachments:")
    for item in duplicates:
        print(f"  {item['episode']} t{item['turn']:>3} {item['owner']:20} {item['label']}"
              f"  useful_alt={item['useful_alternative']} repro={item['reproduced']}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"summary": summary, "duplicates": duplicates, "games": games},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
