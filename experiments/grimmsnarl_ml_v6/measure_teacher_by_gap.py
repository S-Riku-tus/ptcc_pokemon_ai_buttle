"""Per-pilot behaviour on the three gaps a conditional teacher escalation could fix.

v5's remaining divergences from the field are all *preferences on routed
contexts*, so the only lever that has ever moved one is the pilot the ranker is
asked to imitate. Picking the escalation teacher by rating alone is wrong: the
model reproduces the 1220.2 pilot at 0.703 Top-1 and the 1151.0 pilot at 0.813
(see [[grimmsnarl-imitability-vs-rating]]), so a teacher whose *own* rate is
only slightly better but who is reproduced far more faithfully can move the
deployed behaviour further.

This reports, per team, the pilot's own rate on:

* the Froslass evolve, at MAIN decisions where it is offered (which is the
  granularity the ranker imitates) and per own turn (which is the behaviour),
  split by mirror and by whether the Freezing Shroud ledger is net negative;
* the Petrel search, taking an Unfair Stamp that cannot be played this turn;
* the once-per-turn Dark Energy attachment, made when legal.

Usage:
    python experiments/grimmsnarl_ml_v6/measure_teacher_by_gap.py --out <json>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v5"))

import ml_features as mf  # noqa: E402

FROSLASS_ID = 104
SNORUNT_ID = 860
STAMP_ID = 1080
PETREL_ID = 1219
BOSS_ID = 1182
DARK_ID = 7
MARNIE_LINE = {646, 647, 648}
DECK_HASH = "9714ab5c3996f6cc"
MAIN = 0
CTX_TO_HAND = 7


def nested_id(value) -> int:
    if isinstance(value, dict):
        if "id" in value or "cardId" in value:
            try:
                return int(value.get("id", value.get("cardId", -1)))
            except Exception:
                return -1
        for item in value.values():
            found = nested_id(item)
            if found >= 0:
                return found
    elif isinstance(value, list):
        for item in value:
            found = nested_id(item)
            if found >= 0:
                return found
    return -1


def ability_holders(player: dict) -> int:
    """Bodies Freezing Shroud would put a counter on. Same as the v3 probe."""
    return len(mf.shroud_targets(mf._in_play(player)))


def scan(payload: tuple[str, int]) -> Counter:
    path, seat = payload
    counts: Counter = Counter()
    try:
        replay = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return counts
    steps = replay.get("steps") or []
    mirror = False
    turn_offered: dict[int, bool] = {}
    turn_taken: dict[int, bool] = {}
    turn_mirror: dict[int, bool] = {}
    turn_negative: dict[int, bool] = {}
    attach_offered: dict[int, bool] = {}
    attach_taken: dict[int, bool] = {}
    opp_prize_by_turn: dict[int, int] = {}

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if not select:
            continue
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", seat))
        if len(players) < 2 or your >= len(players):
            continue
        me, opponent = players[your], players[1 - your]
        action = (steps[index + 1][seat] or {}).get("action")
        if not isinstance(action, list) or not action:
            continue
        if any(
            int(card.get("id", -1)) in MARNIE_LINE
            for card in mf._in_play(opponent)
        ):
            mirror = True
        context = int(select.get("context", -1))
        options = list(select.get("option") or [])
        turn = int(current.get("turn", -1))
        opp_prize = len(opponent.get("prize") or [])
        if turn not in opp_prize_by_turn:
            opp_prize_by_turn[turn] = opp_prize

        if context == MAIN:
            actions = [mf.action_type(current, o, select) for o in options]
            cards = [
                int((mf.candidate_card(current, o, select) or {}).get("id", -1))
                for o in options
            ]
            froslass = [
                slot for slot, act in enumerate(actions)
                if act == "evolve" and cards[slot] == FROSLASS_ID
            ]
            if froslass:
                negative = ability_holders(opponent) - ability_holders(me) < 0
                took = int(action[0] in froslass)
                band = "mirror" if mirror else "other"
                counts["fros_dec_offered"] += 1
                counts["fros_dec_taken"] += took
                counts[f"fros_dec_offered_{band}"] += 1
                counts[f"fros_dec_taken_{band}"] += took
                if negative:
                    counts["fros_dec_offered_negative"] += 1
                    counts["fros_dec_taken_negative"] += took
                turn_offered[turn] = True
                turn_mirror[turn] = mirror
                turn_negative[turn] = turn_negative.get(turn, False) or negative
                if took:
                    turn_taken[turn] = True

            energy = [
                slot for slot, act in enumerate(actions)
                if act == "energy" and cards[slot] == DARK_ID
            ]
            if energy:
                attach_offered[turn] = True
                if action[0] in energy:
                    attach_taken[turn] = True

        if context == CTX_TO_HAND and nested_id(select.get("effect")) == PETREL_ID:
            ids = []
            for option in options:
                card, _, _ = mf.resolve_option(current, select, option)
                ids.append(int((card or {}).get("id", -1)))
            picked = {
                ids[slot] for slot in action
                if isinstance(slot, int) and 0 <= slot < len(ids)
            }
            hand = Counter()
            for card in me.get("hand") or []:
                try:
                    hand[int(card.get("id", -1))] += 1
                except Exception:
                    pass
            if STAMP_ID in ids and not hand[STAMP_ID]:
                earlier = [t for t in opp_prize_by_turn if t < turn]
                prior = opp_prize_by_turn[max(earlier)] if earlier else 6
                playable = opp_prize < prior
                key = "live" if playable else "dead"
                counts[f"stamp_offered_{key}"] += 1
                counts[f"stamp_taken_{key}"] += int(STAMP_ID in picked)
            if BOSS_ID in ids:
                counts["petrel_boss_offered"] += 1
                counts["petrel_boss_taken"] += int(BOSS_ID in picked)

    for turn in turn_offered:
        band = "mirror" if turn_mirror.get(turn) else "other"
        counts["fros_turn_offered"] += 1
        counts["fros_turn_taken"] += int(turn in turn_taken)
        counts[f"fros_turn_offered_{band}"] += 1
        counts[f"fros_turn_taken_{band}"] += int(turn in turn_taken)
        if turn_negative.get(turn):
            counts["fros_turn_offered_negative"] += 1
            counts["fros_turn_taken_negative"] += int(turn in turn_taken)
    for turn in attach_offered:
        counts["attach_turn_offered"] += 1
        counts["attach_turn_taken"] += int(turn in attach_taken)
    counts["games"] += 1
    return counts


def corpus_jobs(manifest: Path | None):
    """(path, seat, team, score) for every same-deck replay in the archive."""
    index = ROOT / "data" / "kaggle_grimmsnarl_top50" / "indexes" / "replay_index.csv"
    base = ROOT / "data" / "kaggle_grimmsnarl_top50"
    allowed: set[tuple[str, str]] | None = None
    if manifest is not None:
        allowed = {
            (row["episode_id"], row["seat_index"])
            for row in csv.DictReader(open(manifest, encoding="utf-8-sig"))
        }
    seen: set[tuple[str, str]] = set()
    for row in csv.DictReader(open(index, encoding="utf-8-sig")):
        if row["deck_hash"] != DECK_HASH:
            continue
        key = (row["episode_id"], row["seat_index"])
        if key in seen or (allowed is not None and key not in allowed):
            continue
        seen.add(key)
        path = base / Path(row["replay_path"].replace(chr(92), "/"))
        if not path.exists():
            continue
        yield (
            str(path), int(row["seat_index"]), int(row["team_id"]),
            float(row["submission_score"]), row["team_name"],
        )


def rate(counts: Counter, num: str, den: str) -> float | None:
    return round(counts[num] / counts[den], 4) if counts[den] else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v5"
        / "data_refresh_selection.csv",
        help="Restrict to the frozen training selection; blank for all.",
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    manifest = args.manifest if args.manifest and str(args.manifest) else None
    jobs = list(corpus_jobs(manifest))
    print(f"replays={len(jobs)}", flush=True)
    per_team: dict[int, Counter] = defaultdict(Counter)
    names: dict[int, str] = {}
    scores: dict[int, float] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(
            scan, [(path, seat) for path, seat, _, _, _ in jobs], chunksize=16
        )
        for (_, _, team, score, name), counts in zip(jobs, results):
            per_team[team].update(counts)
            names[team] = name
            scores[team] = score

    report = {}
    for team, counts in per_team.items():
        report[str(team)] = {
            "team_name": names[team],
            "submission_score": scores[team],
            "games": counts["games"],
            "froslass": {
                "decisions_offered": counts["fros_dec_offered"],
                "decision_take_rate": rate(counts, "fros_dec_taken", "fros_dec_offered"),
                "decision_take_rate_mirror": rate(
                    counts, "fros_dec_taken_mirror", "fros_dec_offered_mirror"),
                "decision_take_rate_negative": rate(
                    counts, "fros_dec_taken_negative", "fros_dec_offered_negative"),
                "turns_offered": counts["fros_turn_offered"],
                "turn_take_rate": rate(counts, "fros_turn_taken", "fros_turn_offered"),
                "turn_take_rate_mirror": rate(
                    counts, "fros_turn_taken_mirror", "fros_turn_offered_mirror"),
                "turn_take_rate_negative": rate(
                    counts, "fros_turn_taken_negative", "fros_turn_offered_negative"),
                "turns_offered_mirror": counts["fros_turn_offered_mirror"],
                "turns_offered_negative": counts["fros_turn_offered_negative"],
            },
            "stamp": {
                "dead_offered": counts["stamp_offered_dead"],
                "dead_take_rate": rate(counts, "stamp_taken_dead", "stamp_offered_dead"),
                "live_offered": counts["stamp_offered_live"],
                "live_take_rate": rate(counts, "stamp_taken_live", "stamp_offered_live"),
            },
            "attachment": {
                "turns_legal": counts["attach_turn_offered"],
                "made_when_legal": rate(
                    counts, "attach_turn_taken", "attach_turn_offered"),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ordered = sorted(report.items(), key=lambda kv: -kv[1]["submission_score"])
    header = (
        f"{'team':>9} {'score':>7} {'games':>5} | "
        f"{'fros/dec':>9} {'fros/turn':>9} {'mirror':>7} {'neg':>7} {'n':>5} | "
        f"{'stampdead':>9} {'n':>5} | {'attach':>7}"
    )
    print(header)
    for team, row in ordered:
        fros, stamp, attach = row["froslass"], row["stamp"], row["attachment"]

        def show(value, width=7):
            return f"{value:>{width}.3f}" if value is not None else " " * (width - 1) + "-"

        print(
            f"{team:>9} {row['submission_score']:>7.1f} {row['games']:>5} | "
            f"{show(fros['decision_take_rate'], 9)} "
            f"{show(fros['turn_take_rate'], 9)} "
            f"{show(fros['turn_take_rate_mirror'])} "
            f"{show(fros['turn_take_rate_negative'])} "
            f"{fros['turns_offered']:>5} | "
            f"{show(stamp['dead_take_rate'], 9)} {stamp['dead_offered']:>5} | "
            f"{show(attach['made_when_legal'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
