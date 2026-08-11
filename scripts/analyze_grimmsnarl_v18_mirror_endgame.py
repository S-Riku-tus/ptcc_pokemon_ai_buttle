"""Audit mirror decisions after each seat's first Shadow Bullet.

The v15 aggregate says that our first attack is early enough, but it does not
say whether a later action was refused or merely unavailable.  This audit
therefore works at the legal-option level.  MAIN actions are collapsed by
game/turn (a card that remains playable through five decisions is one offer),
while effect selections are counted once per selection prompt.

The default corpus combines the refreshed v15 runs with three public pilots
rated above 1,100 that use the exact same 60 cards.  Missing replay files are
allowed so the report can be regenerated while an incremental download is in
progress.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v17"
for path in (ROOT, ROOT / "scripts", AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402

MAIN = 0
OPTION_ATTACK = 13
OUR_DECK = sorted(
    int(value.strip())
    for value in (AGENT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if value.strip()
)

CARD_NAMES = {
    mf.DARK_ENERGY_ID: "darkness",
    mf.FROSLASS_ID: "froslass",
    mf.MUNKIDORI_ID: "munkidori",
    mf.IMPIDIMP_ID: "impidimp",
    mf.MORGREM_ID: "morgrem",
    mf.GRIMMSNARL_EX_ID: "grimmsnarl_ex",
    mf.SNORUNT_ID: "snorunt",
    mf.RARE_CANDY_ID: "rare_candy",
    mf.UNFAIR_STAMP_ID: "unfair_stamp",
    mf.POFFIN_ID: "poffin",
    mf.NIGHT_STRETCHER_ID: "night_stretcher",
    mf.POKEGEAR_ID: "pokegear",
    mf.TOOL_SCRAPPER_ID: "tool_scrapper",
    mf.POKE_PAD_ID: "poke_pad",
    mf.BOSS_ID: "boss",
    mf.PETREL_ID: "petrel",
    mf.LILLIE_ID: "lillie",
    mf.DAWN_ID: "dawn",
    mf.SPIKEMUTH_GYM_ID: "spikemuth_gym",
}

DEFAULT_RUNS = (
    ("v15_a", "deployed", "55404196",
     "data/runs/grimmsnarl/20260811_v18_refresh_sub55404196"),
    ("v15_b", "deployed", "55409394",
     "data/runs/grimmsnarl/20260811_v18_refresh_sub55409394"),
    ("raihan_1151", "teacher", "55177269",
     "data/runs/grimmsnarl/v18_teacher_55177269"),
    ("kd_1116", "teacher", "55187358",
     "data/runs/grimmsnarl/v18_teacher_55187358"),
    ("sixth_1114", "teacher", "55138264",
     "data/runs/grimmsnarl/v18_teacher_55138264"),
)


def card_name(card_id: int) -> str:
    return CARD_NAMES.get(card_id, str(card_id))


def nested_id(value: Any) -> int:
    if isinstance(value, dict):
        if isinstance(value.get("id"), int):
            return int(value["id"])
        for child in value.values():
            found = nested_id(child)
            if found >= 0:
                return found
    elif isinstance(value, list):
        for child in value:
            found = nested_id(child)
            if found >= 0:
                return found
    return -1


def selected_indices(steps: list[Any], index: int, seat: int) -> list[int]:
    if index + 1 >= len(steps) or seat >= len(steps[index + 1]):
        return []
    action = (steps[index + 1][seat] or {}).get("action")
    if not isinstance(action, list):
        return []
    return [int(value) for value in action if isinstance(value, int)]


def deck_at(steps: list[Any], seat: int) -> list[int] | None:
    if len(steps) <= 1 or seat >= len(steps[1]):
        return None
    action = (steps[1][seat] or {}).get("action")
    if not (isinstance(action, list) and len(action) == 60):
        return None
    return sorted(int(value) for value in action)


def option_signature(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
) -> str:
    context = int(select.get("context", -1))
    action = mf.action_type(current, option, select)
    card = mf.candidate_card(current, option, select) or {}
    target = mf.candidate_target(current, option) or {}
    card_id = int(card.get("id", -1))
    target_id = int(target.get("id", -1))
    attack_id = mf._int(option.get("attackId"))

    if context == MAIN:
        if action == "attack":
            return f"attack:{attack_id}"
        if action in {"energy", "retreat"}:
            return f"{action}:{card_name(target_id)}"
        if action == "ability":
            return f"ability:{card_name(card_id)}"
        if action == "end":
            return "end"
        return f"{action}:{card_name(card_id)}"

    effect_id = nested_id(select.get("effect"))
    resolved, owner_is_self, area = mf.resolve_option(current, select, option)
    resolved_id = int((resolved or {}).get("id", -1))
    chosen_id = resolved_id if resolved_id >= 0 else (
        target_id if target_id >= 0 else card_id
    )
    owner = "self" if owner_is_self else "opp"
    return (
        f"ctx{context}:effect_{card_name(effect_id)}:"
        f"{owner}_area{area}:{card_name(chosen_id)}"
    )


def ready_grim_count(player: dict[str, Any]) -> int:
    return sum(
        int(
            int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf._dark_energy_count(card) >= mf.SHADOW_BULLET_COST
        )
        for card in mf._cards(player, "active") + mf._cards(player, "bench")
    )


@dataclass
class SideAudit:
    saw_shadow: bool = False
    first_shadow_turn: int | None = None
    prizes_before_first_shadow: int | None = None
    final_prizes_taken: int = 0
    final_deck: int | None = None
    post_decisions: int = 0
    post_main_turns: set[int] = field(default_factory=set)
    main_offers_by_turn: dict[int, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    main_taken_by_turn: dict[int, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    prompt_offers: Counter[str] = field(default_factory=Counter)
    prompt_taken: Counter[str] = field(default_factory=Counter)
    post_actions: Counter[str] = field(default_factory=Counter)
    post_shadow_attacks: int = 0
    post_turns_with_ready_backup: set[int] = field(default_factory=set)
    post_turns_without_ready_backup: set[int] = field(default_factory=set)
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def finish(self) -> None:
        for offers in self.main_offers_by_turn.values():
            self.prompt_offers.update(offers)
        for taken in self.main_taken_by_turn.values():
            self.prompt_taken.update(taken)

    def summary(self) -> dict[str, Any]:
        after = None
        if self.prizes_before_first_shadow is not None:
            after = self.final_prizes_taken - self.prizes_before_first_shadow
        return {
            "first_shadow_turn": self.first_shadow_turn,
            "prizes_before_first_shadow": self.prizes_before_first_shadow,
            "prizes_after_first_shadow": after,
            "final_prizes_taken": self.final_prizes_taken,
            "final_deck": self.final_deck,
            "post_decisions": self.post_decisions,
            "post_main_turns": len(self.post_main_turns),
            "post_shadow_attacks": self.post_shadow_attacks,
            "post_turns_with_ready_backup": len(
                self.post_turns_with_ready_backup
            ),
            "post_turns_without_ready_backup": len(
                self.post_turns_without_ready_backup
            ),
        }


def audit_side(replay: dict[str, Any], seat: int) -> SideAudit:
    result = SideAudit()
    steps = replay.get("steps") or []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(current, dict) or not current.get("players"):
            continue
        players = current.get("players") or [{}, {}]
        if seat >= len(players):
            continue
        me = players[seat]
        result.final_prizes_taken = 6 - len(me.get("prize") or [])
        result.final_deck = int(me.get("deckCount", 0) or 0)
        if not isinstance(select, dict):
            continue
        options = list(select.get("option") or [])
        chosen = selected_indices(steps, index, seat)
        if not chosen or any(not 0 <= slot < len(options) for slot in chosen):
            continue

        context = int(select.get("context", -1))
        turn = int(current.get("turn", -1))
        chosen_signatures = {
            option_signature(current, select, options[slot]) for slot in chosen
        }
        chosen_shadow = any(
            mf._int(options[slot].get("type")) == OPTION_ATTACK
            and mf._int(options[slot].get("attackId")) == mf.SHADOW_BULLET_ID
            for slot in chosen
        )

        if result.saw_shadow:
            result.post_decisions += 1
            signatures = {
                option_signature(current, select, option)
                for option in options
            }
            if context == MAIN:
                result.post_main_turns.add(turn)
                result.main_offers_by_turn[turn].update(signatures)
                result.main_taken_by_turn[turn].update(chosen_signatures)
                ready = ready_grim_count(me)
                if ready >= 2:
                    result.post_turns_with_ready_backup.add(turn)
                else:
                    result.post_turns_without_ready_backup.add(turn)
            else:
                result.prompt_offers.update(signatures)
                result.prompt_taken.update(chosen_signatures)
            result.post_actions.update(chosen_signatures)
            if chosen_shadow:
                result.post_shadow_attacks += 1
            if len(result.timeline) < 120:
                result.timeline.append({
                    "step": index,
                    "turn": turn,
                    "context": context,
                    "prizes_taken": 6 - len(me.get("prize") or []),
                    "deck": int(me.get("deckCount", 0) or 0),
                    "ready_grim": ready_grim_count(me),
                    "chosen": sorted(chosen_signatures),
                    "offered": sorted(signatures),
                })

        if chosen_shadow and not result.saw_shadow:
            result.saw_shadow = True
            result.first_shadow_turn = turn
            result.prizes_before_first_shadow = 6 - len(me.get("prize") or [])

    result.finish()
    return result


@dataclass(frozen=True)
class RunSpec:
    name: str
    group: str
    submission: str
    directory: Path


def load_games(specs: Iterable[RunSpec]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for spec in specs:
        episodes_csv = spec.directory / "episodes.csv"
        if not episodes_csv.exists():
            continue
        for raw in csv.DictReader(episodes_csv.open(encoding="utf-8-sig")):
            if raw.get("state") != "COMPLETED":
                continue
            a0 = raw.get("agent_0_submission_id", "")
            a1 = raw.get("agent_1_submission_id", "")
            if spec.submission not in (a0, a1) or a0 == a1:
                continue
            seat = 0 if a0 == spec.submission else 1
            episode_id = int(raw["episode_id"])
            replay_path = (
                spec.directory / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not replay_path.exists():
                continue
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            steps = replay.get("steps") or []
            if deck_at(steps, seat) != OUR_DECK:
                continue
            if deck_at(steps, 1 - seat) != OUR_DECK:
                continue
            rewards = replay.get("rewards") or [0, 0]
            won = rewards[seat] > rewards[1 - seat]
            games.append({
                "run": spec.name,
                "group": spec.group,
                "submission": spec.submission,
                "episode_id": episode_id,
                "won": won,
                "us": audit_side(replay, seat),
                "them": audit_side(replay, 1 - seat),
                "initial_rating": float(
                    raw.get(f"agent_{seat}_initial_score") or 0
                ),
                "opponent_initial_rating": float(
                    raw.get(f"agent_{1-seat}_initial_score") or 0
                ),
            })
    return games


def mean(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.fmean(clean), 3) if clean else None


def block(rows: list[tuple[dict[str, Any], SideAudit]]) -> dict[str, Any]:
    offered: Counter[str] = Counter()
    taken: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    summaries: list[dict[str, Any]] = []
    for _, side in rows:
        offered.update(side.prompt_offers)
        taken.update(side.prompt_taken)
        actions.update(side.post_actions)
        summaries.append(side.summary())
    keys = sorted(set(offered) | set(taken))
    uptake = [
        {
            "signature": key,
            "offered": offered[key],
            "taken": taken[key],
            "rate": round(taken[key] / offered[key], 4) if offered[key] else None,
        }
        for key in keys
        if offered[key] >= 2
    ]
    uptake.sort(key=lambda row: (-row["offered"], row["signature"]))
    metric_keys = sorted({key for row in summaries for key in row})
    return {
        "games": len(rows),
        "wins": sum(int(game["won"]) for game, _ in rows),
        "means": {
            key: mean(row.get(key) for row in summaries)
            for key in metric_keys
        },
        "uptake": uptake,
        "post_actions": dict(actions.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    specs = [
        RunSpec(name, group, submission, ROOT / directory)
        for name, group, submission, directory in DEFAULT_RUNS
    ]
    games = load_games(specs)

    groups: dict[str, list[tuple[dict[str, Any], SideAudit]]] = defaultdict(list)
    for game in games:
        groups[f"{game['run']}:submission"].append((game, game["us"]))
        if game["group"] == "deployed":
            groups["deployed:all"].append((game, game["us"]))
            groups[
                "deployed:wins" if game["won"] else "deployed:losses"
            ].append((game, game["us"]))
            if not game["won"]:
                groups["opponents:beat_deployed"].append((game, game["them"]))
        else:
            groups["teachers:all"].append((game, game["us"]))
            groups[
                "teachers:wins" if game["won"] else "teachers:losses"
            ].append((game, game["us"]))

    output = {
        "corpus": {
            "runs": [
                {
                    "name": spec.name,
                    "group": spec.group,
                    "submission": spec.submission,
                    "directory": str(spec.directory.relative_to(ROOT)),
                }
                for spec in specs
            ],
            "mirror_games": len(games),
            "deployed_games": sum(g["group"] == "deployed" for g in games),
            "teacher_games": sum(g["group"] == "teacher" for g in games),
        },
        "blocks": {
            name: block(rows) for name, rows in sorted(groups.items())
        },
        "episodes": [
            {
                "run": game["run"],
                "group": game["group"],
                "episode_id": game["episode_id"],
                "won": game["won"],
                "initial_rating": game["initial_rating"],
                "opponent_initial_rating": game["opponent_initial_rating"],
                "us": game["us"].summary(),
                "them": game["them"].summary(),
                "us_timeline": (
                    game["us"].timeline
                    if game["group"] == "deployed" and not game["won"]
                    else []
                ),
            }
            for game in games
        ],
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
