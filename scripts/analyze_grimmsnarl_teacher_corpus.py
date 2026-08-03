"""Measure how many distinct policies the top-50 Marnie's Grimmsnarl ex pilots run.

The Alakazam line hit a hard ceiling because it imitated one pilot with 2.5k
games. The Grimmsnarl corpus has 21 top-50 teams sharing one identical 60-card
list, which is only worth using if those teams actually play alike. Mixing
conflicting teachers is what made direct multi-teacher Alakazam training fail.

This answers that before any model is trained. Every MAIN decision is reduced
to three signatures of decreasing strictness, and for each one the script
reports:

* self-agreement: one team meeting the same signature twice. This is the
  determinism ceiling; no imitator can beat it.
* cross-agreement: two different teams meeting the same signature. Close to
  self-agreement means one shared policy, far below means separate policies.

Signature levels:

* ``full``   - board, hand multiset, both benches, counts, offered menu.
* ``coarse`` - drops serials, HP detail, deck/discard counts and the opponent
  bench detail; keeps what a player would actually read.
* ``menu``   - turn bucket plus the offered option multiset only. Collides
  often, so it is the level that produces usable pair counts.

Output is JSON so the corpus builder can consume the accepted team list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN_CONTEXT = 0
LEVELS = ("full", "coarse", "menu")
MIN_PAIR_COUNT = 50


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [c for c in (player.get(area) or []) if isinstance(c, dict)]


def _attached_ids(card: dict[str, Any]) -> tuple[int, ...]:
    out: list[int] = []
    for energy in card.get("energyCards") or card.get("energies") or []:
        if isinstance(energy, dict):
            out.append(int(energy.get("id", energy.get("cardId", -1))))
        else:
            try:
                out.append(int(energy))
            except (TypeError, ValueError):
                out.append(-1)
    return tuple(sorted(out))


def _tool_ids(card: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(
        int(tool.get("id", -1))
        for tool in (card.get("tools") or [])
        if isinstance(tool, dict)
    ))


def _card_full(card: dict[str, Any]) -> tuple:
    return (
        int(card.get("id", -1)),
        int(card.get("hp", 0)),
        int(card.get("maxHp", 0)),
        _attached_ids(card),
        _tool_ids(card),
        int(bool(card.get("appearThisTurn"))),
        len(card.get("preEvolution") or []),
    )


def _card_coarse(card: dict[str, Any]) -> tuple:
    hp = int(card.get("hp", 0))
    max_hp = max(1, int(card.get("maxHp", 1)))
    return (
        int(card.get("id", -1)),
        min(4, int(4 * hp / max_hp)),
        len(_attached_ids(card)),
        int(bool(card.get("tools"))),
    )


def _count(player: dict[str, Any], key: str) -> int:
    value = player.get(key)
    if isinstance(value, list):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


# Engine area codes, same table the runtime agents use.
AREA_NAMES = {2: "hand", 3: "discard", 4: "active", 5: "bench"}


def _card_id_at(player: dict[str, Any], area: Any, index: Any) -> int:
    if not isinstance(index, int):
        return -1
    name = AREA_NAMES.get(int(area) if isinstance(area, int) else -1)
    if name is None:
        return -1
    cards = _cards(player, name)
    return int(cards[index].get("id", -1)) if 0 <= index < len(cards) else -1


def option_sig(current: dict[str, Any], option: dict[str, Any]) -> tuple:
    """Semantic identity of a MAIN option, independent of list position."""
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    me = players[your] if your < len(players) else {}
    option_type = int(option.get("type", -1))

    if option_type == 7:
        return (7, _card_id_at(me, 2, option.get("index")))
    if option_type in (8, 9):
        area = option.get("inPlayArea")
        target = option.get("inPlayIndex")
        energy = -1
        name = AREA_NAMES.get(int(area) if isinstance(area, int) else -1)
        if name is not None and isinstance(target, int):
            cards = _cards(me, name)
            if 0 <= target < len(cards):
                energy = len(_attached_ids(cards[target]))
        return (
            option_type,
            _card_id_at(me, 2, option.get("index")),
            int(area) if isinstance(area, int) else -1,
            _card_id_at(me, area, target),
            energy,
        )
    if option_type == 10:
        area = option.get("area")
        card_id = _card_id_at(me, area, option.get("index"))
        # Keep the raw area when it is outside the known table so two
        # unresolved abilities do not collapse into one identity.
        return (10, card_id, -1 if card_id >= 0 else (area, option.get("index")))
    if option_type == 12:
        return (12,)
    if option_type == 13:
        return (13, int(option.get("attackId", -1)))
    if option_type == 14:
        return (14,)
    return (option_type, json.dumps(option, sort_keys=True))


def action_label(sig: tuple) -> str:
    kind = sig[0]
    if kind == 7:
        return f"play:{sig[1]}"
    if kind == 8:
        return f"energy:{sig[2]}:{sig[3]}"
    if kind == 9:
        return f"evolve:{sig[1]}"
    if kind == 10:
        return f"ability:{sig[1]}"
    if kind == 12:
        return "retreat"
    if kind == 13:
        return f"attack:{sig[1]}"
    if kind == 14:
        return "end"
    return f"other:{kind}"


def signatures(
    current: dict[str, Any],
    options: list[dict[str, Any]],
    menu: tuple,
) -> dict[str, tuple]:
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    me = players[your] if your < len(players) else {}
    opp = players[1 - your] if 1 - your < len(players) else {}
    turn = int(current.get("turn", -1))
    flags = (
        int(current.get("turnActionCount", -1)),
        int(bool(current.get("supporterPlayed"))),
        int(bool(current.get("retreated"))),
        int(bool(current.get("energyAttached"))),
        int(current.get("firstPlayer", -1)) == your,
    )
    stadium = tuple(sorted(
        int(card.get("id", -1))
        for card in (current.get("stadium") or [])
        if isinstance(card, dict)
    ))
    hand = tuple(sorted(
        int(card.get("id", -1)) for card in _cards(me, "hand")
    ))
    prizes = (_count(me, "prize"), _count(opp, "prize"))

    full = (
        turn, flags, stadium, hand, prizes,
        tuple(_card_full(c) for c in _cards(me, "active")),
        tuple(sorted(_card_full(c) for c in _cards(me, "bench"))),
        tuple(_card_full(c) for c in _cards(opp, "active")),
        tuple(sorted(_card_full(c) for c in _cards(opp, "bench"))),
        _count(me, "deckCount"), _count(me, "discard"),
        _count(opp, "deckCount"), _count(opp, "discard"),
        _count(opp, "handCount"),
        menu,
    )
    coarse = (
        min(turn, 12), flags, stadium, hand, prizes,
        tuple(_card_coarse(c) for c in _cards(me, "active")),
        tuple(sorted(_card_coarse(c) for c in _cards(me, "bench"))),
        tuple(_card_coarse(c) for c in _cards(opp, "active")),
        tuple(sorted(
            int(c.get("id", -1)) for c in _cards(opp, "bench")
        )),
        menu,
    )
    return {
        "full": full,
        "coarse": coarse,
        "menu": (min(turn, 12), flags[1:], menu),
    }


def _digest(value: tuple) -> bytes:
    """Process-stable key. Python's hash() is salted per process."""
    return hashlib.blake2b(repr(value).encode("utf-8"), digest_size=12).digest()


def _trajectory(replay: dict[str, Any], seat: int) -> list[tuple]:
    steps = replay.get("steps") or []
    out: list[tuple] = []
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
        if (
            int(select.get("context", -1)) != MAIN_CONTEXT
            or int(select.get("minCount") or 0) != 1
            or int(select.get("maxCount") or 0) != 1
            or len(options) < 2
            or not isinstance(action, list)
            or len(action) != 1
            or not isinstance(action[0], int)
            or not 0 <= action[0] < len(options)
        ):
            continue
        current = observation.get("current") or {}
        option_sigs = [option_sig(current, option) for option in options]
        menu = tuple(sorted(option_sigs))
        chosen = option_sigs[action[0]]
        sigs = signatures(current, options, menu)
        out.append((
            {level: _digest(sigs[level]) for level in LEVELS},
            _digest(chosen),
            action_label(chosen),
            int(current.get("turn", -1)),
            len(options),
        ))
    return out


def _worker(payload: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    replay_root, rows = payload
    records: list[tuple] = []
    stats: Counter[str] = Counter()
    per_team_actions: dict[Any, Counter[str]] = defaultdict(Counter)
    for row in rows:
        path = Path(replay_root) / f"episode_{row['episode_id']}.json"
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            stats["replay_unreadable"] += 1
            continue
        seat = int(row["seat_index"])
        steps = replay.get("steps") or []
        final = steps[-1] if steps else []
        own = final[seat].get("reward") if seat < len(final) else None
        other = final[1 - seat].get("reward") if 1 - seat < len(final) else None
        won = own is not None and other is not None and own > other
        stats["episodes"] += 1
        stats["wins"] += int(won)
        team = row["team_id"]
        for sigs, chosen, label, turn, count in _trajectory(replay, seat):
            stats["decisions"] += 1
            per_team_actions[team][label] += 1
            records.append((team, sigs, chosen, turn, count))
    return {
        "records": records,
        "stats": dict(stats),
        "actions": {k: dict(v) for k, v in per_team_actions.items()},
    }


def _agreement(
    by_sig: dict[bytes, list[tuple]],
    cap: int,
) -> dict[str, Any]:
    self_total = self_agree = 0
    cross_total = cross_agree = 0
    pairs: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    self_pairs: dict[Any, list[int]] = defaultdict(lambda: [0, 0])
    shared = 0
    for entries in by_sig.values():
        if len(entries) < 2:
            continue
        if len({team for team, _ in entries}) > 1:
            shared += 1
        entries = entries[:cap]
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                same = int(entries[i][1] == entries[j][1])
                if entries[i][0] == entries[j][0]:
                    self_total += 1
                    self_agree += same
                    self_pairs[entries[i][0]][0] += 1
                    self_pairs[entries[i][0]][1] += same
                else:
                    cross_total += 1
                    cross_agree += same
                    key = tuple(sorted((entries[i][0], entries[j][0])))
                    pairs[key][0] += 1
                    pairs[key][1] += same
    pair_rows = sorted(
        (
            {
                "teams": [int(a), int(b)],
                "collisions": total,
                "agreement": round(agree / total, 4),
            }
            for (a, b), (total, agree) in pairs.items()
            if total >= MIN_PAIR_COUNT
        ),
        key=lambda item: item["agreement"],
    )
    # How well each team agrees with everyone else. A team far below the
    # field is running its own policy and should be dropped from the corpus.
    per_team: dict[Any, list[int]] = defaultdict(lambda: [0, 0])
    for (a, b), (total, agree) in pairs.items():
        per_team[a][0] += total
        per_team[a][1] += agree
        per_team[b][0] += total
        per_team[b][1] += agree
    team_rows = sorted(
        (
            {
                "team_id": int(team),
                "collisions": total,
                "agreement_with_field": round(agree / total, 4),
            }
            for team, (total, agree) in per_team.items()
            if total >= MIN_PAIR_COUNT
        ),
        key=lambda item: item["agreement_with_field"],
    )
    # Per-pilot determinism under this state description. If a pilot's own
    # self-agreement is high but a model fitted to it scores far lower, the
    # gap is model capacity, not pilot noise.
    self_rows = sorted(
        (
            {
                "team_id": int(team),
                "self_pairs": total,
                "self_agreement": round(agree / total, 4),
            }
            for team, (total, agree) in self_pairs.items()
            if total >= MIN_PAIR_COUNT
        ),
        key=lambda item: item["self_agreement"],
    )
    return {
        "team_self_agreement": self_rows,
        "team_agreement_with_field": team_rows,
        "distinct_signatures": len(by_sig),
        "signatures_with_repeats": sum(1 for v in by_sig.values() if len(v) > 1),
        "signatures_shared_across_teams": shared,
        "self_pairs": self_total,
        "self_agreement": round(self_agree / self_total, 4) if self_total else None,
        "cross_pairs": cross_total,
        "cross_agreement": (
            round(cross_agree / cross_total, 4) if cross_total else None
        ),
        "worst_team_pairs": pair_rows[:10],
        "best_team_pairs": pair_rows[-10:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--deck-hash", default="9714ab5c3996f6cc")
    parser.add_argument("--limit-per-team", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pair-cap", type=int, default=60)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v1" / "teacher_corpus.json",
    )
    args = parser.parse_args()

    index = pd.read_csv(args.data_root / "indexes" / "episodes.csv")
    index = index[index["download_status"] == "success"]
    if args.deck_hash:
        index = index[index["deck_hash"] == args.deck_hash]
    # One episode appears twice when two indexed teams played each other;
    # both seats are then genuine teacher trajectories.
    index = index.drop_duplicates(subset=["episode_id", "seat_index"])
    if args.limit_per_team:
        index = index.groupby("team_id", group_keys=False).head(
            args.limit_per_team
        )

    rows = index[[
        "team_id", "episode_id", "seat_index", "leaderboard_rank",
        "submission_score",
    ]].to_dict("records")
    print(
        f"trajectories={len(rows)} teams={index['team_id'].nunique()}",
        flush=True,
    )

    replay_root = str((args.data_root / "replays").resolve())
    workers = max(1, min(args.workers, len(rows)))
    chunks = [rows[i::workers] for i in range(workers)]
    chunks = [chunk for chunk in chunks if chunk]
    if workers == 1:
        parts = [_worker((replay_root, chunks[0]))]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(
                _worker, [(replay_root, chunk) for chunk in chunks]
            ))

    stats: Counter[str] = Counter()
    per_team: Counter[Any] = Counter()
    per_team_actions: dict[Any, Counter[str]] = defaultdict(Counter)
    by_level: dict[str, dict[bytes, list[tuple]]] = {
        level: defaultdict(list) for level in LEVELS
    }
    for part in parts:
        stats.update(part["stats"])
        for team, actions in part["actions"].items():
            per_team_actions[team].update(actions)
        for team, sigs, chosen, _turn, _count in part["records"]:
            per_team[team] += 1
            for level in LEVELS:
                by_level[level][sigs[level]].append((team, chosen))

    meta = index.drop_duplicates("team_id").set_index("team_id")
    teams_detail = [
        {
            "team_id": int(team),
            "leaderboard_rank": int(meta.loc[team, "leaderboard_rank"]),
            "submission_score": float(meta.loc[team, "submission_score"]),
            "decisions": int(count),
            "action_mix": {
                label: round(n / count, 4)
                for label, n in per_team_actions[team].most_common(8)
            },
        }
        for team, count in per_team.most_common()
    ]

    report = {
        "deck_hash": args.deck_hash,
        "trajectories": len(rows),
        "teams": int(index["team_id"].nunique()),
        "episodes": int(stats["episodes"]),
        "win_rate": round(stats["wins"] / max(1, stats["episodes"]), 4),
        "main_decisions": int(stats["decisions"]),
        "levels": {
            level: _agreement(by_level[level], args.pair_cap)
            for level in LEVELS
        },
        "teams_detail": teams_detail,
        "extraction_stats": dict(stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value for key, value in report.items()
        if key not in ("teams_detail",)
    }
    for level in LEVELS:
        summary["levels"][level].pop("worst_team_pairs", None)
        summary["levels"][level].pop("best_team_pairs", None)
        summary["levels"][level].pop("team_agreement_with_field", None)
        summary["levels"][level].pop("team_self_agreement", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
