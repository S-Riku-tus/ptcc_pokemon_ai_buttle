"""Significance and confounder checks on the v21-vs-field behaviour gaps.

Three things the raw tables cannot say on their own:

* whether a per-own-turn gap survives N=59 games (Fisher / Welch);
* whether the gap is an opponent-mix artefact (the field corpus is 2026-07/08
  and ours is 2026-08-13, so the archetype mixes differ);
* whether v21's going-second deficit is larger than the field's, which is an
  interaction test, not two separate win rates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
TOP5 = ["55167115", "55171940", "55177269", "55090635", "55170504"]
TOP10 = TOP5 + ["55109984", "55138264", "55168467", "55160202", "55132728"]


def keys(frame: pd.DataFrame) -> set:
    return set(zip(frame.source, frame.episode_id, frame.seat))


def subset(turns: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    k = keys(frame)
    return turns[[x in k for x in zip(turns.source, turns.episode_id, turns.seat)]]


def main() -> int:
    games = pd.read_parquet(OUT / "games.parquet")
    turns = pd.read_parquet(OUT / "turns.parquet")

    v21 = games[games.source == "v21"]
    field = games[games.source == "field"]
    top5 = games[games.pilot.isin(TOP5)]
    top10 = games[games.pilot.isin(TOP10)]

    print("=== opponent family mix ===")
    mix = pd.DataFrame({
        "v21": v21.opponent_family.value_counts(normalize=True),
        "field": field.opponent_family.value_counts(normalize=True),
        "top5": top5.opponent_family.value_counts(normalize=True),
    }).fillna(0).round(3)
    mix["v21_n"] = v21.opponent_family.value_counts()
    print(mix.sort_values("field", ascending=False).to_string())

    print("\n=== turn-order interaction (is v21's second-seat deficit worse?) ===")
    for name, other in [("field", field), ("top5", top5), ("top10", top10)]:
        table = np.array([
            [int(v21[v21.went_first == 1].won.sum()),
             len(v21[v21.went_first == 1]) - int(v21[v21.went_first == 1].won.sum())],
            [int(v21[v21.went_first == 0].won.sum()),
             len(v21[v21.went_first == 0]) - int(v21[v21.went_first == 0].won.sum())],
        ])
        other_table = np.array([
            [int(other[other.went_first == 1].won.sum()),
             len(other[other.went_first == 1]) - int(other[other.went_first == 1].won.sum())],
            [int(other[other.went_first == 0].won.sum()),
             len(other[other.went_first == 0]) - int(other[other.went_first == 0].won.sum())],
        ])
        our_or = (table[0, 0] * table[1, 1]) / max(1e-9, table[0, 1] * table[1, 0])
        their_or = (
            (other_table[0, 0] * other_table[1, 1])
            / max(1e-9, other_table[0, 1] * other_table[1, 0])
        )
        # Breslow-Day style: compare the two log odds ratios.
        se = np.sqrt(sum(1 / max(1, x) for x in table.flatten())
                     + sum(1 / max(1, x) for x in other_table.flatten()))
        z = (np.log(our_or) - np.log(their_or)) / se
        print(
            f"  v21 OR(first vs second)={our_or:.3f}  {name} OR={their_or:.3f}  "
            f"z={z:.2f} p={2 * (1 - stats.norm.cdf(abs(z))):.3f}"
        )
        first_p = stats.fisher_exact([
            [int(v21[v21.went_first == 1].won.sum()),
             len(v21[v21.went_first == 1]) - int(v21[v21.went_first == 1].won.sum())],
            [int(other[other.went_first == 1].won.sum()),
             len(other[other.went_first == 1]) - int(other[other.went_first == 1].won.sum())],
        ])[1]
        second_p = stats.fisher_exact([
            [int(v21[v21.went_first == 0].won.sum()),
             len(v21[v21.went_first == 0]) - int(v21[v21.went_first == 0].won.sum())],
            [int(other[other.went_first == 0].won.sum()),
             len(other[other.went_first == 0]) - int(other[other.went_first == 0].won.sum())],
        ])[1]
        print(f"    v21 vs {name}: first p={first_p:.3f}, second p={second_p:.3f}")

    print("\n=== own-turn gaps, v21 vs top5 / field (Fisher for rates, Welch for means) ===")
    binary_cols = [
        "take_attack", "take_shadow", "energy_attached_flag", "supporter_flag",
        "board_grim_ex_on_board", "board_grim_ex_ready", "take_adrena",
        "retreat_flag", "take_boss", "take_rare_candy", "take_poffin",
        "take_froslass_evolve", "take_stamp", "take_gym",
    ]
    mean_cols = [
        "board_energy_in_play", "board_bench", "board_deck_left", "board_hand",
        "board_prize_left", "main_decisions",
    ]
    rows = []
    for cohort_name, cohort in [("top5", top5), ("field", field)]:
        ours = subset(turns, v21)
        theirs = subset(turns, cohort)
        for turn in range(1, 7):
            a = ours[ours.own_turn == turn]
            b = theirs[theirs.own_turn == turn]
            if len(a) < 5:
                continue
            for column in binary_cols:
                ha, hb = int((a[column] > 0).sum()), int((b[column] > 0).sum())
                p = stats.fisher_exact(
                    [[ha, len(a) - ha], [hb, len(b) - hb]]
                )[1]
                rows.append({
                    "vs": cohort_name, "own_turn": turn, "metric": column,
                    "v21": round(ha / len(a), 3), "them": round(hb / len(b), 3),
                    "delta": round(ha / len(a) - hb / len(b), 3),
                    "n_v21": len(a), "n_them": len(b), "p": round(p, 4),
                })
            for column in mean_cols:
                t, p = stats.ttest_ind(
                    a[column].astype(float), b[column].astype(float),
                    equal_var=False,
                )
                rows.append({
                    "vs": cohort_name, "own_turn": turn, "metric": column,
                    "v21": round(a[column].mean(), 3),
                    "them": round(b[column].mean(), 3),
                    "delta": round(a[column].mean() - b[column].mean(), 3),
                    "n_v21": len(a), "n_them": len(b), "p": round(p, 4),
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "gap_tests.csv", index=False)
    strong = table[(table.p < 0.05)].sort_values("p")
    print(strong.to_string(index=False))

    print("\n=== same, restricted to going second ===")
    rows = []
    for cohort_name, cohort in [("top5", top5), ("field", field)]:
        ours = subset(turns, v21[v21.went_first == 0])
        theirs = subset(turns, cohort[cohort.went_first == 0])
        for turn in range(1, 7):
            a = ours[ours.own_turn == turn]
            b = theirs[theirs.own_turn == turn]
            if len(a) < 5:
                continue
            for column in binary_cols + mean_cols:
                if column in binary_cols:
                    ha, hb = int((a[column] > 0).sum()), int((b[column] > 0).sum())
                    p = stats.fisher_exact([[ha, len(a) - ha], [hb, len(b) - hb]])[1]
                    va, vb = ha / len(a), hb / len(b)
                else:
                    _, p = stats.ttest_ind(
                        a[column].astype(float), b[column].astype(float),
                        equal_var=False,
                    )
                    va, vb = a[column].mean(), b[column].mean()
                rows.append({
                    "vs": cohort_name, "own_turn": turn, "metric": column,
                    "v21": round(va, 3), "them": round(vb, 3),
                    "delta": round(va - vb, 3), "n_v21": len(a), "n_them": len(b),
                    "p": round(p, 4),
                })
    second = pd.DataFrame(rows)
    second.to_csv(OUT / "gap_tests_second.csv", index=False)
    print(second[second.p < 0.05].sort_values("p").to_string(index=False))

    print("\n=== within-field: does behaviour at own turn 2 predict winning? ===")
    field_turns = subset(turns, field)
    t2 = field_turns[field_turns.own_turn == 2].merge(
        field[["source", "episode_id", "seat", "won", "went_first"]],
        on=["source", "episode_id", "seat"],
    )
    for column in ["board_grim_ex_on_board", "take_attack", "supporter_flag",
                   "energy_attached_flag"]:
        yes = t2[t2[column] > 0]
        no = t2[t2[column] == 0]
        lo1, hi1 = wilson(int(yes.won.sum()), len(yes))
        lo0, hi0 = wilson(int(no.won.sum()), len(no))
        p = stats.fisher_exact([
            [int(yes.won.sum()), len(yes) - int(yes.won.sum())],
            [int(no.won.sum()), len(no) - int(no.won.sum())],
        ])[1]
        print(
            f"  {column:26s} yes={yes.won.mean():.3f}[{lo1:.3f},{hi1:.3f}] n={len(yes)} "
            f"| no={no.won.mean():.3f}[{lo0:.3f},{hi0:.3f}] n={len(no)}  p={p:.2e}"
        )
    print("\n  same, field going second only:")
    t2s = t2[t2.went_first == 0]
    for column in ["board_grim_ex_on_board", "take_attack"]:
        yes = t2s[t2s[column] > 0]
        no = t2s[t2s[column] == 0]
        lo1, hi1 = wilson(int(yes.won.sum()), len(yes))
        lo0, hi0 = wilson(int(no.won.sum()), len(no))
        print(
            f"  {column:26s} yes={yes.won.mean():.3f}[{lo1:.3f},{hi1:.3f}] n={len(yes)} "
            f"| no={no.won.mean():.3f}[{lo0:.3f},{hi0:.3f}] n={len(no)}"
        )

    print("\n=== energy in play at own turn 2, by pilot rating band ===")
    field_turns2 = field_turns[field_turns.own_turn == 2]
    bands = pd.cut(field_turns2.rating, [1000, 1075, 1125, 1300])
    print(field_turns2.groupby(bands, observed=True).agg(
        turns=("board_energy_in_play", "size"),
        energy=("board_energy_in_play", "mean"),
        grim=("board_grim_ex_on_board", "mean"),
        attack=("take_attack", "mean"),
        supporter=("supporter_flag", "mean"),
    ).round(3).to_string())
    ours2 = subset(turns, v21)
    ours2 = ours2[ours2.own_turn == 2]
    print(f"  v21: turns={len(ours2)} energy={ours2.board_energy_in_play.mean():.3f} "
          f"grim={(ours2.board_grim_ex_on_board > 0).mean():.3f} "
          f"attack={(ours2.take_attack > 0).mean():.3f} "
          f"supporter={(ours2.supporter_flag > 0).mean():.3f}")

    print("\n=== per-pilot own-turn-2 board rate (is v21 below every pilot?) ===")
    per = field_turns2.groupby(["pilot", "rating"]).agg(
        turns=("board_grim_ex_on_board", "size"),
        grim2=("board_grim_ex_on_board", "mean"),
        energy2=("board_energy_in_play", "mean"),
        attack2=("take_attack", "mean"),
    ).round(3).sort_values("rating", ascending=False)
    print(per.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
