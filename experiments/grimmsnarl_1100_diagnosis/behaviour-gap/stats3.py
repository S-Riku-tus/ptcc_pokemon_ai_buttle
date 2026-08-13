"""Turn-order-stratified version of the gap table.

v21 went first in 36/59 (0.610) and the field in 2161/4204 (0.514), and several
metrics are only legal in one seat (the player going first may not play a
supporter on their own turn 1, and cannot attack on it either).  Comparing
pooled rates therefore mixes a seat effect into every difference, so every
number here is computed inside a turn-order stratum and, where a pooled figure
is wanted, re-weighted to the field's 51.4/48.6 seat mix.

Also splits ``take_ability`` into Munkidori's Adrena-Brain and Spikemuth Gym's
search: the Gym sits in stadium area 7, which ``candidate_card`` cannot resolve,
so its ability lands in the ``ability`` bucket with card id -1.
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

BINARY = [
    "take_attack", "take_shadow", "supporter_flag", "energy_attached_flag",
    "take_rare_candy", "board_grim_ex_on_board", "take_gym", "gym_ability",
    "take_adrena", "take_boss", "take_poffin", "take_stamp", "retreat_flag",
    "take_froslass_evolve", "take_petrel", "take_lillie", "take_dawn",
    "take_bench", "idle",
]
MEAN = [
    "board_energy_in_play", "board_bench", "board_deck_left", "board_hand",
    "main_decisions", "board_prize_left",
]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet(OUT / "games.parquet")
    turns = pd.read_parquet(OUT / "turns.parquet")
    turns = turns.assign(
        gym_ability=(turns.take_ability - turns.take_adrena).clip(lower=0)
    )
    merged = turns.merge(
        games[["source", "episode_id", "seat", "won", "went_first"]],
        on=["source", "episode_id", "seat"],
    )
    return games, merged


def main() -> int:
    games, merged = load()
    field = merged[merged.source == "field"]
    top5 = merged[merged.pilot.isin(TOP5)]
    ours = merged[merged.source == "v21"]

    for seat_label, seat_value in [("FIRST", 1), ("SECOND", 0)]:
        print(f"\n########## going {seat_label} ##########")
        f = field[field.went_first == seat_value]
        t = top5[top5.went_first == seat_value]
        o = ours[ours.went_first == seat_value]
        print(f"  games: v21={o.episode_id.nunique()} "
              f"top5={t.episode_id.nunique()} field={f.episode_id.nunique()}")
        rows = []
        for turn in (1, 2, 3, 4):
            a = o[o.own_turn == turn]
            b = t[t.own_turn == turn]
            c = f[f.own_turn == turn]
            if len(a) < 5:
                continue
            for column in BINARY:
                ha = int((a[column] > 0).sum())
                hb = int((b[column] > 0).sum())
                hc = int((c[column] > 0).sum())
                p = stats.fisher_exact(
                    [[ha, len(a) - ha], [hc, len(c) - hc]]
                )[1]
                per = c.groupby(["pilot", "rating"])[column].apply(
                    lambda s: float((s > 0).mean())
                ).reset_index()
                rho, prho = stats.spearmanr(per.rating, per[column])
                rate = ha / len(a)
                rows.append({
                    "turn": turn, "metric": column,
                    "v21": round(rate, 3), "top5": round(hb / max(1, len(b)), 3),
                    "field": round(hc / max(1, len(c)), 3),
                    "n_v21": len(a),
                    "rank/23": int((per[column] < rate).sum() + 1),
                    "pmin": round(float(per[column].min()), 3),
                    "pmax": round(float(per[column].max()), 3),
                    "p_vs_field": round(float(p), 4),
                    "rho_rating": round(float(rho), 3),
                    "p_rho": round(float(prho), 4),
                })
            for column in MEAN:
                _, p = stats.ttest_ind(
                    a[column].astype(float), c[column].astype(float),
                    equal_var=False,
                )
                per = c.groupby(["pilot", "rating"])[column].mean().reset_index()
                rho, prho = stats.spearmanr(per.rating, per[column])
                rows.append({
                    "turn": turn, "metric": column,
                    "v21": round(float(a[column].mean()), 3),
                    "top5": round(float(b[column].mean()), 3),
                    "field": round(float(c[column].mean()), 3),
                    "n_v21": len(a),
                    "rank/23": int((per[column] < a[column].mean()).sum() + 1),
                    "pmin": round(float(per[column].min()), 3),
                    "pmax": round(float(per[column].max()), 3),
                    "p_vs_field": round(float(p), 4),
                    "rho_rating": round(float(rho), 3),
                    "p_rho": round(float(prho), 4),
                })
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"stratified_{seat_label.lower()}.csv", index=False)
        show = table[
            (table.p_vs_field < 0.10) | (table["rank/23"].isin([1, 2, 22, 23]))
        ]
        print(show.to_string(index=False))

    print("\n########## seat-mix-standardised pooled rates ##########")
    weight = {
        1: float((games[games.source == "field"].went_first == 1).mean()),
        0: float((games[games.source == "field"].went_first == 0).mean()),
    }
    print(f"  field seat mix: first={weight[1]:.3f} second={weight[0]:.3f}; "
          f"v21 raw first={(games[games.source=='v21'].went_first==1).mean():.3f}")
    rows = []
    for column in BINARY + MEAN:
        entry = {"metric": column}
        for label, frame in [("v21", ours), ("top5", top5), ("field", field)]:
            total = 0.0
            for seat_value, w in weight.items():
                block = frame[
                    (frame.went_first == seat_value)
                    & (frame.own_turn >= 1) & (frame.own_turn <= 6)
                ]
                if len(block) == 0:
                    continue
                value = (
                    float((block[column] > 0).mean()) if column in BINARY
                    else float(block[column].mean())
                )
                total += w * value
            entry[label] = round(total, 3)
        entry["delta_vs_top5"] = round(entry["v21"] - entry["top5"], 3)
        rows.append(entry)
    print(pd.DataFrame(rows).sort_values("delta_vs_top5").to_string(index=False))

    print("\n########## gym ability + stadium, per own turn ##########")
    for label, frame in [("v21", ours), ("top5", top5), ("field", field)]:
        line = []
        for turn in range(1, 7):
            block = frame[frame.own_turn == turn]
            line.append(
                f"t{turn}:{float((block.gym_ability > 0).mean()):.3f}/"
                f"{float((block.take_gym > 0).mean()):.3f}"
            )
        print(f"  {label:6s} (ability/play) " + "  ".join(line))

    print("\n########## Spikemuth Gym: field outcome, stratified ##########")
    block = field[field.own_turn <= 2].groupby(
        ["pilot", "episode_id", "seat", "won", "went_first"]
    ).agg(gym=("take_gym", "max"), ability=("gym_ability", "max")).reset_index()
    for column in ("gym", "ability"):
        yes = block[block[column] > 0]
        no = block[block[column] == 0]
        lo1, hi1 = wilson(int(yes.won.sum()), len(yes))
        lo0, hi0 = wilson(int(no.won.sum()), len(no))
        p = stats.fisher_exact([
            [int(yes.won.sum()), len(yes) - int(yes.won.sum())],
            [int(no.won.sum()), len(no) - int(no.won.sum())],
        ])[1]
        print(f"  by own turn 2 {column}: yes {yes.won.mean():.3f}"
              f"[{lo1:.3f},{hi1:.3f}] n={len(yes)} | no {no.won.mean():.3f}"
              f"[{lo0:.3f},{hi0:.3f}] n={len(no)}  p={p:.2e}")
    ours_block = ours[ours.own_turn <= 2].groupby(
        ["episode_id", "seat", "won"]
    ).agg(gym=("take_gym", "max"), ability=("gym_ability", "max")).reset_index()
    print(f"  v21: gym by own turn 2 in {int((ours_block.gym > 0).sum())}/"
          f"{len(ours_block)} games; field "
          f"{int((block.gym > 0).sum())}/{len(block)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
