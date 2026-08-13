"""Per-own-turn behaviour tables: v21 vs the same-deck field vs the top-5 pilots.

Every rate here has its denominator stated in the row: ``turns`` is the number
of *own* turns that reached that ordinal, ``games`` the number of episodes.
Nothing is divided by a decision count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
TOP5 = ["55167115", "55171940", "55177269", "55090635", "55170504"]
TOP10 = TOP5 + ["55109984", "55138264", "55168467", "55160202", "55132728"]


def cohorts(games: pd.DataFrame, turns: pd.DataFrame) -> dict[str, pd.Index]:
    """Episode keys per cohort."""
    key = games.apply(lambda r: f"{r.source}|{r.episode_id}|{r.seat}", axis=1)
    games = games.assign(key=key)
    out = {
        "v21": games[games.source == "v21"],
        "field": games[games.source == "field"],
        "top5": games[games.pilot.isin(TOP5)],
        "top10": games[games.pilot.isin(TOP10)],
    }
    return {name: frame for name, frame in out.items()}


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "   -  "
    return f"{value:.{digits}f}"


def rate_table(
    turns: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    column: str,
    max_turn: int = 8,
    binary: bool = True,
) -> pd.DataFrame:
    rows = []
    for name, frame in groups.items():
        keys = set(zip(frame.source, frame.episode_id, frame.seat))
        sub = turns[
            [k in keys for k in zip(turns.source, turns.episode_id, turns.seat)]
        ]
        row: dict[str, object] = {"cohort": name}
        for turn in range(1, max_turn + 1):
            block = sub[sub.own_turn == turn]
            n = len(block)
            if n == 0:
                row[f"t{turn}"] = "   -  "
                continue
            if binary:
                hits = int((block[column] > 0).sum())
                low, high = wilson(hits, n)
                row[f"t{turn}"] = f"{hits / n:.3f}[{low:.2f},{high:.2f}] n={n}"
            else:
                row[f"t{turn}"] = f"{block[column].mean():.3f} n={n}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("cohort")


def per_turn_scalar(
    turns: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    column: str,
    binary: bool,
    max_turn: int,
) -> pd.DataFrame:
    rows = []
    for name, frame in groups.items():
        keys = set(zip(frame.source, frame.episode_id, frame.seat))
        sub = turns[
            [k in keys for k in zip(turns.source, turns.episode_id, turns.seat)]
        ]
        row: dict[str, object] = {"cohort": name}
        for turn in range(1, max_turn + 1):
            block = sub[sub.own_turn == turn]
            n = len(block)
            row[f"n{turn}"] = n
            if n == 0:
                row[f"t{turn}"] = np.nan
                continue
            row[f"t{turn}"] = (
                (block[column] > 0).mean() if binary else block[column].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("cohort")


def overall(
    turns: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    columns: list[str],
    max_turn: int = 99,
) -> pd.DataFrame:
    rows = []
    for name, frame in groups.items():
        keys = set(zip(frame.source, frame.episode_id, frame.seat))
        sub = turns[
            [k in keys for k in zip(turns.source, turns.episode_id, turns.seat)]
        ]
        sub = sub[(sub.own_turn >= 1) & (sub.own_turn <= max_turn)]
        row: dict[str, object] = {
            "cohort": name,
            "games": len(frame),
            "own_turns": len(sub),
            "turns_per_game": round(len(sub) / max(1, len(frame)), 2),
            "win_rate": round(frame.won.mean(), 4),
        }
        for column in columns:
            hits = int((sub[column] > 0).sum())
            low, high = wilson(hits, len(sub))
            row[column] = f"{hits / max(1, len(sub)):.3f} [{low:.3f},{high:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("cohort")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-turn", type=int, default=8)
    parser.add_argument("--exclude-mirror", action="store_true")
    parser.add_argument("--only-first", action="store_true")
    parser.add_argument("--only-second", action="store_true")
    args = parser.parse_args()

    games = pd.read_parquet(OUT / "games.parquet")
    turns = pd.read_parquet(OUT / "turns.parquet")
    if args.exclude_mirror:
        games = games[games.opponent_family != "Grimmsnarl (mirror)"]
    if args.only_first:
        games = games[games.went_first == 1]
    if args.only_second:
        games = games[games.went_first == 0]

    groups = cohorts(games, turns)
    print("=== cohort sizes / outcome ===")
    for name, frame in groups.items():
        wins = int(frame.won.sum())
        low, high = wilson(wins, len(frame))
        first = frame[frame.went_first == 1]
        second = frame[frame.went_first == 0]
        fw, sw = int(first.won.sum()), int(second.won.sum())
        print(
            f"{name:8s} games={len(frame):5d} win={wins/max(1,len(frame)):.3f} "
            f"[{low:.3f},{high:.3f}]  first={fw}/{len(first)} "
            f"({fw/max(1,len(first)):.3f}) second={sw}/{len(second)} "
            f"({sw/max(1,len(second)):.3f}) pilots={frame.pilot.nunique()}"
        )

    behaviour = [
        "take_attack", "energy_attached_flag", "supporter_flag", "take_adrena",
        "take_grim_evolve", "take_morgrem_evolve", "take_froslass_evolve",
        "retreat_flag", "take_boss", "take_stamp", "take_rare_candy",
        "take_poffin", "take_gym", "take_bench", "take_item", "idle",
    ]
    print("\n=== per own turn (turns 1..%d), rate of turns with >=1 ===" % args.max_turn)
    print(overall(turns, groups, behaviour, args.max_turn).to_string())

    print("\n=== per-turn curves ===")
    for column, binary in [
        ("take_attack", True),
        ("take_shadow", True),
        ("energy_attached_flag", True),
        ("supporter_flag", True),
        ("take_adrena", True),
        ("take_grim_evolve", True),
        ("take_froslass_evolve", True),
        ("retreat_flag", True),
        ("idle", True),
        ("main_decisions", False),
        ("board_grim_ex_on_board", True),
        ("board_grim_ex_ready", True),
        ("board_bench", False),
        ("board_bodies", False),
        ("board_energy_in_play", False),
        ("board_dark_in_play", False),
        ("board_prize_left", False),
        ("board_opp_prize_left", False),
        ("board_hand", False),
        ("board_deck_left", False),
    ]:
        table = per_turn_scalar(turns, groups, column, binary, args.max_turn)
        print(f"\n--- {column} ({'rate' if binary else 'mean'}) ---")
        show = table[[f"t{i}" for i in range(1, args.max_turn + 1)]].round(3)
        show.columns = [f"own{i}" for i in range(1, args.max_turn + 1)]
        print(show.to_string())
        ns = table[[f"n{i}" for i in range(1, args.max_turn + 1)]]
        print("  n:", ns.loc["v21"].tolist(), "| top5:", ns.loc["top5"].tolist())

    print("\n=== offer vs take inside MAIN (turns 1..%d) ===" % args.max_turn)
    rows = []
    for name, frame in groups.items():
        keys = set(zip(frame.source, frame.episode_id, frame.seat))
        sub = turns[
            [k in keys for k in zip(turns.source, turns.episode_id, turns.seat)]
        ]
        sub = sub[(sub.own_turn >= 1) & (sub.own_turn <= args.max_turn)]
        for kind in ["attack", "energy", "supporter", "ability", "evolve",
                     "retreat", "boss", "stamp", "item", "bench", "stadium"]:
            offered = sub[sub[f"offer_{kind}"] > 0]
            if len(offered) == 0:
                continue
            taken = int((offered[f"take_{kind}"] > 0).sum())
            low, high = wilson(taken, len(offered))
            rows.append({
                "cohort": name, "kind": kind,
                "offer_rate": round(len(offered) / max(1, len(sub)), 3),
                "take_when_offered": round(taken / len(offered), 3),
                "ci": f"[{low:.3f},{high:.3f}]",
                "offered_turns": len(offered),
            })
    table = pd.DataFrame(rows).pivot(
        index="kind", columns="cohort",
        values=["offer_rate", "take_when_offered"],
    )
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
