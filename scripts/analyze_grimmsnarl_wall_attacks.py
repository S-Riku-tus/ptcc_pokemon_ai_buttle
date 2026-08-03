"""Do the top-50 Grimmsnarl pilots attack into a damage-immune wall?

Crustle, Sylveon and Cornerstone Mask Ogerpon ex prevent all damage from a
Pokemon ex, so Grimmsnarl ex's Shadow Bullet does 0 to them in the Active
spot. The Bench-30 still lands, so the attack is not always worthless - it is
worthless when the Bench-30 cannot take a prize either.

`grimmsnarl_ml_v1` was reported to have spent 45 of its 264 ladder attacks on
an Active Crustle. That is only a defect if the teachers do not do it. The
inherited v7 rule policy models the wall (`shadow_damage`, `bench_damage_lands`)
and v1 discards that by returning the ranker's argmax over MAIN with no veto,
while `ml_features.attack_kills_active` computes `0 < hp <= 180` with no
immunity term - the model cannot see that the attack does nothing.

This measures, per source, every MAIN decision where an attack was legal and
the effective damage to the Active was zero, and reports how often the actor
attacked anyway. If the teachers decline and we attack, the fix is a feature
plus a retrain. If the teachers attack too, our model is faithful and the
premise is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

MAIN_CONTEXT = 0
OPTION_ATTACK = 13
OPTION_END = 14
AREA_ACTIVE = 4
AREA_BENCH = 5
SHADOW_BULLET_ID = 937
SHADOW_BULLET_BENCH_DAMAGE = 30

# Abilities that prevent all damage from an attacking Pokemon ex (Grimmsnarl ex
# is one). Resolved from the vendor card database when it is importable, with
# the three known bodies always present so the analysis works without it.
FALLBACK_BLOCKERS = {345, 330, 117}  # Crustle, Sylveon, Cornerstone Ogerpon ex
# Abilities that shield the opposing Bench from our Bench-30.
BENCH_SHIELDS = {74, 343}  # Rabsca (whole bench), Shaymin (non-Rule-Box)
NEUTRALIZATION_ZONE = 1247


def _blockers() -> set[int]:
    blockers = set(FALLBACK_BLOCKERS)
    try:
        from policy_base import card_table  # type: ignore
    except Exception:
        return blockers
    for data in card_table.values():
        for skill in (getattr(data, "skills", None) or []):
            text = getattr(skill, "text", "") or ""
            low = text.lower()
            if "prevent all damage" not in low or "this pok" not in low:
                continue
            if "basic pokémon" in low or "basic pokemon" in low:
                continue
            if "tera" in low or "special energy" in low:
                continue
            if "your bench" in low or "benched pok" in low:
                continue
            if "{ex}" in text or "pokémon ex" in low or "pokemon ex" in low \
                    or "have an ability" in low:
                card_id = getattr(data, "cardId", None)
                if card_id is not None:
                    blockers.add(int(card_id))
    return blockers


def _rule_box(card_id: int) -> bool:
    try:
        from policy_base import card_table  # type: ignore
    except Exception:
        return False
    data = card_table.get(card_id)
    return bool(data and (getattr(data, "ex", False)
                          or getattr(data, "megaEx", False)))


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [c for c in (player.get(area) or []) if isinstance(c, dict)]


def _stadium_id(current: dict[str, Any]) -> int:
    stadium = current.get("stadium")
    if isinstance(stadium, dict):
        return int(stadium.get("id", -1))
    if isinstance(stadium, list) and stadium and isinstance(stadium[0], dict):
        return int(stadium[0].get("id", -1))
    return -1


def _analyse_decision(
    current: dict[str, Any],
    options: list[dict[str, Any]],
    chosen: int,
    blockers: set[int],
) -> dict[str, Any] | None:
    """Classify one MAIN decision where a Shadow Bullet was legal."""
    attack_positions = [
        position for position, option in enumerate(options)
        if int(option.get("type", -1)) == OPTION_ATTACK
        and int(option.get("attackId", -1)) == SHADOW_BULLET_ID
    ]
    if not attack_positions:
        return None

    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    me = players[your] if your < len(players) else {}
    opponent = players[1 - your] if 1 - your < len(players) else {}

    active_list = _cards(opponent, "active")
    if not active_list:
        return None
    active = active_list[0]
    active_id = int(active.get("id", -1))
    active_hp = int(active.get("hp", 0) or 0)

    stadium = _stadium_id(current)
    walled = active_id in blockers or (
        stadium == NEUTRALIZATION_ZONE and not _rule_box(active_id)
    )

    # Does the Bench-30 take a prize this turn?
    bench = _cards(opponent, "bench")
    shield_ids = {
        int(c.get("id", -1))
        for c in (_cards(opponent, "active") + bench)
    }
    bench_prize = False
    for body in bench:
        body_id = int(body.get("id", -1))
        if body_id in blockers:
            continue
        if 74 in shield_ids:
            continue  # Rabsca shields the whole bench
        if 343 in shield_ids and not _rule_box(body_id):
            continue  # Shaymin shields non-Rule-Box bench
        if stadium == NEUTRALIZATION_ZONE and not _rule_box(body_id):
            continue
        if 0 < int(body.get("hp", 0) or 0) <= SHADOW_BULLET_BENCH_DAMAGE:
            bench_prize = True
            break

    active_prize = walled is False and 0 < active_hp <= 180
    attacked = chosen in attack_positions
    # A losing board where any attack is better than none should not count as
    # a mistake: if the only other option is END, attacking is free.
    only_end = all(
        int(option.get("type", -1)) in (OPTION_ATTACK, OPTION_END)
        for option in options
    )
    return {
        "walled": int(walled),
        "active_id": active_id,
        "bench_prize": int(bench_prize),
        "active_prize": int(active_prize),
        "attacked": int(attacked),
        "only_end": int(only_end),
        # The defect the analysis names: zero damage to the Active, no prize
        # from the Bench-30, and something else to do with the turn.
        "dead_attack_available": int(walled and not bench_prize),
        "dead_attack_taken": int(walled and not bench_prize and attacked),
        "my_prizes": len(_cards(me, "prize")),
        "opp_prizes": len(_cards(opponent, "prize")),
    }


def _scan(payload: tuple[str, list[dict[str, Any]], list[int]]) -> dict[str, Any]:
    replay_root, rows, blocker_list = payload
    blockers = set(blocker_list)
    per_source: dict[str, Counter] = defaultdict(Counter)
    wall_bodies: Counter = Counter()
    per_episode: list[dict[str, Any]] = []

    for row in rows:
        path = Path(replay_root) / row["replay_name"]
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seat = int(row["seat_index"])
        source = str(row["source"])
        steps = replay.get("steps") or []
        if not steps:
            continue
        final = steps[-1]
        own = final[seat].get("reward") if seat < len(final) else None
        other = final[1 - seat].get("reward") if 1 - seat < len(final) else None
        won = int(own is not None and other is not None and own > other)

        episode = Counter()
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            if int(select.get("context", -1)) != MAIN_CONTEXT:
                continue
            options = list(select.get("option") or [])
            action = (steps[index + 1][seat] or {}).get("action")
            if not (isinstance(action, list) and len(action) == 1
                    and isinstance(action[0], int)
                    and 0 <= action[0] < len(options)):
                continue
            verdict = _analyse_decision(
                observation.get("current") or {}, options, action[0], blockers
            )
            if verdict is None:
                continue
            for key in ("walled", "bench_prize", "active_prize", "attacked",
                        "dead_attack_available", "dead_attack_taken"):
                episode[key] += verdict[key]
            episode["attack_offers"] += 1
            if verdict["dead_attack_taken"] and verdict["only_end"]:
                episode["dead_attack_only_end"] += 1
            if verdict["walled"]:
                wall_bodies[verdict["active_id"]] += 1

        episode["episodes"] = 1
        episode["wins"] = won
        per_source[source].update(episode)
        if episode["dead_attack_available"]:
            per_episode.append({
                "source": source,
                "episode_id": row.get("episode_id"),
                "seat_index": seat,
                "dead_available": int(episode["dead_attack_available"]),
                "dead_attacks": int(episode["dead_attack_taken"]),
                "attacks": int(episode["attacked"]),
                "won": won,
            })
    return {
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "wall_bodies": dict(wall_bodies),
        "per_episode": per_episode,
    }


def _chunks(rows: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    if workers <= 1:
        return [rows]
    size = max(1, (len(rows) + workers - 1) // workers)
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data/kaggle_grimmsnarl_top50",
        help="top-50 archive with indexes/episodes.csv and replays/",
    )
    parser.add_argument("--deck-hash", default="9714ab5c3996f6cc")
    parser.add_argument(
        "--ladder-run",
        default="data/runs/grimmsnarl/20260803_grimmsnarl_ml_v1_sub55185513",
    )
    parser.add_argument("--ladder-submission", type=int, default=55185513)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    blockers = sorted(_blockers())
    rows: list[dict[str, Any]] = []

    data_root = ROOT / args.data_root
    index_path = data_root / "indexes" / "episodes.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
        index = index[index["download_status"] == "success"]
        if args.deck_hash:
            index = index[index["deck_hash"] == args.deck_hash]
        index = index.drop_duplicates(subset=["episode_id", "seat_index"])
        for record in index.to_dict("records"):
            rows.append({
                "source": f"team_{int(record['team_id'])}",
                "episode_id": int(record["episode_id"]),
                "seat_index": int(record["seat_index"]),
                "replay_name": f"episode_{int(record['episode_id'])}.json",
            })
        teacher_root = str(data_root / "replays")
    else:
        teacher_root = ""
        print(f"no teacher index at {index_path}", file=sys.stderr)

    results: dict[str, Counter] = defaultdict(Counter)
    wall_bodies: Counter = Counter()
    episodes: list[dict[str, Any]] = []

    if rows:
        payloads = [
            (teacher_root, chunk, blockers)
            for chunk in _chunks(rows, args.workers)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for out in pool.map(_scan, payloads):
                for source, counts in out["per_source"].items():
                    results[source].update(counts)
                wall_bodies.update(out["wall_bodies"])
                episodes.extend(out["per_episode"])

    # The ladder run stores each episode as its own directory.
    ladder_root = ROOT / args.ladder_run
    ladder_rows: list[dict[str, Any]] = []
    manifest = ladder_root / "episodes.csv"
    if manifest.exists():
        frame = pd.read_csv(manifest)
        for record in frame.to_dict("records"):
            episode_id = int(record["episode_id"])
            seat = 0 if int(
                record.get("agent_0_submission_id", -1)
            ) == args.ladder_submission else 1
            ladder_rows.append({
                "source": "grimmsnarl_ml_v1_ladder",
                "episode_id": episode_id,
                "seat_index": seat,
                "replay_name":
                    f"{episode_id}/replay/episode_{episode_id}.json",
            })
    if ladder_rows:
        out = _scan((str(ladder_root / "episodes"), ladder_rows, blockers))
        for source, counts in out["per_source"].items():
            results[source].update(counts)
        wall_bodies.update(out["wall_bodies"])
        episodes.extend(out["per_episode"])

    report: dict[str, Any] = {
        "blockers": blockers,
        "wall_bodies_seen": dict(wall_bodies.most_common(20)),
        "sources": {},
    }
    for source, counts in sorted(results.items()):
        games = counts["episodes"] or 1
        available = counts["dead_attack_available"]
        report["sources"][source] = {
            "episodes": counts["episodes"],
            "wins": counts["wins"],
            "main_decisions_with_attack": counts["attack_offers"],
            "attacks": counts["attacked"],
            "attacks_per_game": round(counts["attacked"] / games, 3),
            "walled_offers": counts["walled"],
            "dead_attack_available": available,
            "dead_attack_taken": counts["dead_attack_taken"],
            "dead_attack_rate": (
                round(counts["dead_attack_taken"] / available, 4)
                if available else None
            ),
            "dead_attacks_per_game": round(
                counts["dead_attack_taken"] / games, 3
            ),
            "dead_attack_only_end": counts["dead_attack_only_end"],
        }
    report["worst_episodes"] = sorted(
        episodes, key=lambda e: -e["dead_attacks"]
    )[:25]
    # Every teacher game that presented a dead swing, so the agreement
    # evaluator can be pointed at the block where the wall features matter.
    report["wall_episodes"] = sorted(
        (e for e in episodes if "team_" in str(e["source"])),
        key=lambda e: -e["dead_available"],
    )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    print(text[:6000])


if __name__ == "__main__":
    main()
