"""Compare the prize race, not the action rates: our run against the teachers.

Every Dragapult analysis so far measures fidelity - did we take the action the
teacher took.  None of them measures the thing the ladder scores, which is who
takes six prizes first.  This walks the prize counters in the replay and
reports the shape of the race: when each side takes its first prize, how big
each knock-out was (1 = non-ex, 2 = ex, 3+ = a Phantom Dive multi-knock-out),
and what the board looked like in the games we lost.

``current.players[i].prize`` is the number of prize cards player i has left, so
prizes taken = 6 - remaining.  The knock-out that ends the game is never
observed, so every count here is censored the same way for both cohorts and the
two are comparable to each other but one prize short of the truth for a winner.

Usage:
  python scripts/analyze_dragapult_prize_race.py \
      --run data/submissions/submission_55550682_dragapult_v2 \
      --teacher-index data/kaggle_dragapult_exact/indexes/episodes.csv \
      --report experiments/dragapult_ml_v2/prize_race_v2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DREEPY, DRAKLOAK, DRAGAPULT, MUNKIDORI, FEZANDIPITI = 119, 120, 121, 112, 108
PHANTOM_DIVE = 154
OPT_ATTACK = 13


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0


def analyse(path: Path, seat: int) -> dict[str, Any] | None:
    replay = load(path)
    steps = replay.get("steps") or []
    rewards = replay.get("rewards") or [0, 0]
    if not steps:
        return None

    own_turn = 0
    seen_turns: set[int] = set()
    our_remaining = opp_remaining = 6
    our_events: list[dict[str, int]] = []
    opp_events: list[dict[str, int]] = []
    started = False
    last_turn = 0
    phantom_dives = 0
    ready_turns = 0
    attack_turns = 0
    turns_seen = 0

    for pair in steps:
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        current = observation.get("current")
        if not isinstance(current, dict):
            continue
        your = int(current.get("yourIndex", seat))
        players = current.get("players") or [{}, {}]
        mine = players[your] if your in (0, 1) else {}
        theirs = players[1 - your] if your in (0, 1) else {}
        counts = [mine.get("prize"), theirs.get("prize")]
        if not all(isinstance(value, list) for value in counts):
            continue
        ours, opps = len(counts[0]), len(counts[1])
        # The pre-deal observation reports zero prizes for both seats.
        if not started:
            if ours == 6 and opps == 6:
                started = True
            else:
                continue

        turn = int(current.get("turn") or 0)
        last_turn = max(last_turn, turn)
        if turn not in seen_turns:
            seen_turns.add(turn)
            own_turn += 1
            turns_seen += 1
            bodies = list(mine.get("active") or []) + list(mine.get("bench") or [])
            armed = any(
                isinstance(card, dict) and int(card.get("id", -1)) == DRAGAPULT
                and 2 in [int(v) for v in card.get("energies") or []]
                and 5 in [int(v) for v in card.get("energies") or []]
                for card in bodies
            )
            ready_turns += int(armed)

        if ours < our_remaining:
            # Our prize counter falls when *we* take prizes.
            our_events.append({"size": our_remaining - ours, "own_turn": own_turn})
            our_remaining = ours
        if opps < opp_remaining:
            opp_events.append({"size": opp_remaining - opps, "own_turn": own_turn})
            opp_remaining = opps

        select = observation.get("select") or {}
        for option in select.get("option") or []:
            del option
            break

    # Attacks are counted from the actions actually taken, not from the offers.
    for index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        select = observation.get("select")
        if not isinstance(select, dict):
            continue
        action = (
            steps[index + 1][seat].get("action")
            if index + 1 < len(steps) else None
        )
        if not isinstance(action, list) or len(action) != 1:
            continue
        options = select.get("option") or []
        chosen = int(action[0])
        if not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        if int(option.get("type", -1)) != OPT_ATTACK:
            continue
        attack_turns += 1
        if int(option.get("attackId", -1)) == PHANTOM_DIVE:
            phantom_dives += 1

    if not started:
        return None
    result = "win" if rewards[seat] > rewards[1 - seat] else "loss"
    return {
        "episode": path.stem,
        "seat": seat,
        "result": result,
        "own_turns": turns_seen,
        "shared_turns": last_turn,
        "our_prizes": 6 - our_remaining,
        "opp_prizes": 6 - opp_remaining,
        "differential": (6 - our_remaining) - (6 - opp_remaining),
        "our_events": our_events,
        "opp_events": opp_events,
        "our_first_prize_own_turn": our_events[0]["own_turn"] if our_events else None,
        "opp_first_prize_own_turn": opp_events[0]["own_turn"] if opp_events else None,
        "phantom_dives": phantom_dives,
        "attack_turns": attack_turns,
        "ready_turns": ready_turns,
    }


# Win rate conditional on how many Phantom Dives the game yielded, measured on
# the 1,392-game exact-list teacher corpus.  Applying these to any agent's own
# histogram gives an "implied win rate" that has reproduced every version's
# observed win rate to within a point, so it is a far tighter strength estimate
# than a 25-game record.
DIVE_WIN_RATE = {0: 0.108, 1: 0.231, 2: 0.731}


def dive_histogram(games: list[dict[str, Any]]) -> dict[str, Any]:
    dives = [game["phantom_dives"] for game in games]
    total = len(dives) or 1
    share = {
        "p_zero": round(sum(1 for value in dives if value == 0) / total, 4),
        "p_one": round(sum(1 for value in dives if value == 1) / total, 4),
        "p_two_plus": round(sum(1 for value in dives if value >= 2) / total, 4),
        "p_four_plus": round(sum(1 for value in dives if value >= 4) / total, 4),
    }
    share["implied_win_rate"] = round(
        share["p_zero"] * DIVE_WIN_RATE[0]
        + share["p_one"] * DIVE_WIN_RATE[1]
        + share["p_two_plus"] * DIVE_WIN_RATE[2], 4)
    # Win rate observed within each bucket in *this* cohort, which says whether
    # the conditionals themselves still hold on the new sample.
    for name, keep in (("wr_zero", lambda value: value == 0),
                       ("wr_one", lambda value: value == 1),
                       ("wr_two_plus", lambda value: value >= 2),
                       ("wr_four_plus", lambda value: value >= 4)):
        block = [game for game in games if keep(game["phantom_dives"])]
        share[name] = (
            round(sum(1 for game in block if game["result"] == "win")
                  / len(block), 4) if block else None
        )
        share[f"n_{name[3:]}"] = len(block)
    return share


def summarise(games: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not games:
        return {"label": label, "games": 0}
    wins = [game for game in games if game["result"] == "win"]
    losses = [game for game in games if game["result"] == "loss"]

    def sizes(key: str) -> dict[str, float]:
        counter: Counter[int] = Counter()
        for game in games:
            for event in game[key]:
                counter[event["size"]] += 1
        total = sum(counter.values()) or 1
        return {
            "events_per_game": round(sum(counter.values()) / len(games), 3),
            "size_1": round(counter[1] / total, 3),
            "size_2": round(counter[2] / total, 3),
            "size_3_plus": round(
                sum(count for size, count in counter.items() if size >= 3) / total, 3
            ),
            "prizes_per_event": round(
                sum(size * count for size, count in counter.items()) / total, 3
            ),
        }

    def firsts(key: str) -> dict[str, Any]:
        values = [game[key] for game in games if game[key] is not None]
        return {
            "rate": round(len(values) / len(games), 3),
            "mean_own_turn": mean(values),
        }

    race_won = sum(
        1 for game in games
        if game["our_first_prize_own_turn"] is not None
        and (game["opp_first_prize_own_turn"] is None
             or game["our_first_prize_own_turn"] <= game["opp_first_prize_own_turn"])
    )
    return {
        "label": label,
        "games": len(games),
        "record": f"{len(wins)}-{len(losses)}",
        "win_rate": round(len(wins) / len(games), 4),
        "our_prizes_mean": mean([game["our_prizes"] for game in games]),
        "opp_prizes_mean": mean([game["opp_prizes"] for game in games]),
        "differential_mean": mean([game["differential"] for game in games]),
        "own_turns_mean": mean([game["own_turns"] for game in games]),
        "prizes_per_own_turn": round(
            sum(game["our_prizes"] for game in games)
            / max(1, sum(game["own_turns"] for game in games)), 3),
        "conceded_per_own_turn": round(
            sum(game["opp_prizes"] for game in games)
            / max(1, sum(game["own_turns"] for game in games)), 3),
        "our_knockouts": sizes("our_events"),
        "opp_knockouts": sizes("opp_events"),
        "our_first_prize": firsts("our_first_prize_own_turn"),
        "opp_first_prize": firsts("opp_first_prize_own_turn"),
        "took_the_first_prize": round(race_won / len(games), 3),
        "phantom_dives_per_game": mean([game["phantom_dives"] for game in games]),
        "phantom_dive_histogram": dive_histogram(games),
        "attack_turns_per_game": mean([game["attack_turns"] for game in games]),
        "loss_shape": {
            "games": len(losses),
            "our_prizes_mean": mean([game["our_prizes"] for game in losses]),
            "blown_out_0_to_2": round(
                sum(1 for game in losses if game["our_prizes"] <= 2)
                / max(1, len(losses)), 3),
            "close_4_or_5": round(
                sum(1 for game in losses if game["our_prizes"] >= 4)
                / max(1, len(losses)), 3),
            "own_turns_mean": mean([game["own_turns"] for game in losses]),
            "phantom_dives_mean": mean([game["phantom_dives"] for game in losses]),
        },
        "win_shape": {
            "games": len(wins),
            "opp_prizes_mean": mean([game["opp_prizes"] for game in wins]),
            "own_turns_mean": mean([game["own_turns"] for game in wins]),
            "phantom_dives_mean": mean([game["phantom_dives"] for game in wins]),
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    if not summary.get("games"):
        return
    print(f"\n=== {summary['label']}  {summary['record']} "
          f"({summary['win_rate']:.3f}) over {summary['games']} games")
    print(f"  prizes taken / conceded (censored)  "
          f"{summary['our_prizes_mean']} / {summary['opp_prizes_mean']}  "
          f"diff {summary['differential_mean']}")
    print(f"  per own turn                        "
          f"{summary['prizes_per_own_turn']} / {summary['conceded_per_own_turn']}")
    print(f"  own turns / attacks / Phantom Dives "
          f"{summary['own_turns_mean']} / {summary['attack_turns_per_game']} / "
          f"{summary['phantom_dives_per_game']}")
    print(f"  first prize: us {summary['our_first_prize']} "
          f"them {summary['opp_first_prize']}")
    print(f"  took the first prize                {summary['took_the_first_prize']}")
    dive = summary["phantom_dive_histogram"]
    print(f"  Phantom Dive  P(0) {dive['p_zero']:.3f}  P(1) {dive['p_one']:.3f}  "
          f"P(2+) {dive['p_two_plus']:.3f}  P(4+) {dive['p_four_plus']:.3f}  "
          f"=> implied wr {dive['implied_win_rate']:.3f}")
    print(f"    observed wr in bucket   0:{dive['wr_zero']} (n={dive['n_zero']})  "
          f"1:{dive['wr_one']} (n={dive['n_one']})  "
          f"2+:{dive['wr_two_plus']} (n={dive['n_two_plus']})  "
          f"4+:{dive['wr_four_plus']} (n={dive['n_four_plus']})")
    for side in ("our_knockouts", "opp_knockouts"):
        stat = summary[side]
        print(f"  {side:16} per game {stat['events_per_game']:>6}  "
              f"1-prize {stat['size_1']:.3f}  2-prize {stat['size_2']:.3f}  "
              f"3+ {stat['size_3_plus']:.3f}  mean {stat['prizes_per_event']}")
    loss, win = summary["loss_shape"], summary["win_shape"]
    print(f"  losses  n={loss['games']:<4} prizes {loss['our_prizes_mean']}  "
          f"blown out (<=2) {loss['blown_out_0_to_2']}  close (>=4) "
          f"{loss['close_4_or_5']}  turns {loss['own_turns_mean']}  "
          f"PD {loss['phantom_dives_mean']}")
    print(f"  wins    n={win['games']:<4} conceded {win['opp_prizes_mean']}  "
          f"turns {win['own_turns_mean']}  PD {win['phantom_dives_mean']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--split-report", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    for run in args.run:
        games = []
        rows = list(csv.DictReader(
            (run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
        ))
        for row in rows:
            episode_id = int(row["episode_id"])
            path = (
                run / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            game = analyse(path, int(row["detected_submission_agent_index"]))
            if game:
                games.append(game)
        summary = summarise(games, f"live {run.name}")
        report[run.name] = summary
        print_summary(summary)

    if args.teacher_index:
        boundaries: dict[str, list[int]] = {}
        if args.split_report:
            boundaries = load(args.split_report).get("split_boundaries") or {}
        seen: set[tuple[str, int]] = set()
        games = []
        by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in csv.DictReader(
            args.teacher_index.read_text(encoding="utf-8-sig").splitlines()
        ):
            episode_id = str(row["episode_id"])
            seat = int(row["seat_index"])
            if (episode_id, seat) in seen:
                continue
            seen.add((episode_id, seat))
            boundary = boundaries.get(str(row.get("team_id")))
            if boundary and int(episode_id) <= int(boundary[1]):
                continue
            path = Path(row["replay_path"])
            if not path.is_absolute():
                path = args.teacher_index.parent.parent / path
            game = analyse(path, seat)
            if game:
                games.append(game)
                by_team[str(row.get("team_id"))].append(game)
        report["teachers"] = summarise(games, "teachers")
        print_summary(report["teachers"])
        report["teachers_by_team"] = {
            team: summarise(rows, f"teacher {team}")
            for team, rows in sorted(by_team.items())
        }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
