"""Is any own-turn-2 gap actually a lever?

Two tests this project insists on before a gap is called a defect:

1. **rating gradient** - Spearman of the pilot's own rate against the pilot's
   ladder rating over the 22 same-deck pilots.  A gap that does not order the
   field cannot explain a rating gap (see the Froslass retraction).
2. **within-pilot outcome** - inside the field corpus, does a game where the
   behaviour happened win more than one where it did not, holding the pilot
   fixed (Mantel-Haenszel over pilots) and holding turn order fixed?
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

METRICS = [
    ("t1_gym", 1, "take_gym"),
    ("t1_attack", 1, "take_attack"),
    ("t1_supporter", 1, "supporter_flag"),
    ("t1_energy", 1, "energy_attached_flag"),
    ("t1_poffin", 1, "take_poffin"),
    ("t2_candy", 2, "take_rare_candy"),
    ("t2_grim", 2, "board_grim_ex_on_board"),
    ("t2_attack", 2, "take_attack"),
    ("t2_supporter", 2, "supporter_flag"),
    ("t2_energy_flag", 2, "energy_attached_flag"),
    ("t2_gym", 2, "take_gym"),
    ("t3_attack", 3, "take_attack"),
    ("t3_boss", 3, "take_boss"),
    ("t4_retreat", 4, "retreat_flag"),
]
MEANS = [
    ("t2_energy_in_play", 2, "board_energy_in_play"),
    ("t3_energy_in_play", 3, "board_energy_in_play"),
    ("t2_deck_left", 2, "board_deck_left"),
    ("t2_hand", 2, "board_hand"),
]


def main() -> int:
    games = pd.read_parquet(OUT / "games.parquet")
    turns = pd.read_parquet(OUT / "turns.parquet")
    merged = turns.merge(
        games[["source", "episode_id", "seat", "won", "went_first",
               "opponent_family"]],
        on=["source", "episode_id", "seat"],
    )
    field = merged[merged.source == "field"]
    ours = merged[merged.source == "v21"]

    print("=== per-pilot rate vs pilot rating (Spearman over 22 pilots) ===")
    rows = []
    for name, turn, column in METRICS + MEANS:
        block = field[field.own_turn == turn]
        per = block.groupby(["pilot", "rating"]).agg(
            value=(column, "mean"), n=(column, "size")
        ).reset_index()
        rho, p = stats.spearmanr(per.rating, per.value)
        mine = ours[ours.own_turn == turn][column]
        rows.append({
            "metric": name,
            "v21": round(float(mine.mean()), 3),
            "field": round(float(block[column].mean()), 3),
            "pilot_min": round(float(per.value.min()), 3),
            "pilot_max": round(float(per.value.max()), 3),
            "v21_rank_of_23": int((per.value < mine.mean()).sum() + 1),
            "rho_vs_rating": round(float(rho), 3),
            "p": round(float(p), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== within-field outcome association, stratified by pilot ===")
    out = []
    for name, turn, column in METRICS:
        block = field[field.own_turn == turn].copy()
        block["flag"] = (block[column] > 0).astype(int)
        # Mantel-Haenszel odds ratio over pilots x turn order strata.
        num = den = 0.0
        for _, group in block.groupby(["pilot", "went_first"]):
            a = int(((group.flag == 1) & (group.won == 1)).sum())
            b = int(((group.flag == 1) & (group.won == 0)).sum())
            c = int(((group.flag == 0) & (group.won == 1)).sum())
            d = int(((group.flag == 0) & (group.won == 0)).sum())
            n = a + b + c + d
            if n == 0:
                continue
            num += a * d / n
            den += b * c / n
        mh = num / den if den else np.nan
        yes = block[block.flag == 1]
        no = block[block.flag == 0]
        p = stats.fisher_exact([
            [int(yes.won.sum()), len(yes) - int(yes.won.sum())],
            [int(no.won.sum()), len(no) - int(no.won.sum())],
        ])[1]
        out.append({
            "metric": name, "n_yes": len(yes), "n_no": len(no),
            "win_yes": round(float(yes.won.mean()), 3),
            "win_no": round(float(no.won.mean()), 3),
            "MH_odds_ratio": round(float(mh), 3),
            "p_unstratified": f"{p:.2e}",
        })
    print(pd.DataFrame(out).to_string(index=False))

    print("\n=== field: energy in play at own turn 2, bucketed, vs win ===")
    block = field[field.own_turn == 2]
    buckets = pd.cut(block.board_energy_in_play, [-0.5, 0.5, 1.5, 2.5, 3.5, 20])
    print(block.groupby(buckets, observed=True).agg(
        n=("won", "size"), win=("won", "mean")
    ).round(3).to_string())
    mine = ours[ours.own_turn == 2]
    b2 = pd.cut(mine.board_energy_in_play, [-0.5, 0.5, 1.5, 2.5, 3.5, 20])
    print("v21 distribution:", mine.groupby(b2, observed=True).size().to_dict())
    print("field distribution:",
          block.groupby(buckets, observed=True).size().to_dict())

    print("\n=== how the v21 own-turn-2 shortfall splits ===")
    for label, frame in [("v21", ours), ("field", field)]:
        block = frame[frame.own_turn == 2]
        print(
            f"  {label}: turns={len(block)} "
            f"candy={float((block.take_rare_candy > 0).mean()):.3f} "
            f"grim_on_board={float((block.board_grim_ex_on_board > 0).mean()):.3f} "
            f"attack={float((block.take_attack > 0).mean()):.3f} "
            f"attack|grim={float((block[block.board_grim_ex_on_board > 0].take_attack > 0).mean()):.3f} "
            f"attack|no_grim={float((block[block.board_grim_ex_on_board == 0].take_attack > 0).mean()):.3f}"
        )

    print("\n=== v21 own-turn-2 outcome split (does it cost us the games?) ===")
    block = ours[ours.own_turn == 2]
    for column in ["board_grim_ex_on_board", "take_attack", "supporter_flag"]:
        yes = block[block[column] > 0]
        no = block[block[column] == 0]
        lo1, hi1 = wilson(int(yes.won.sum()), len(yes))
        lo0, hi0 = wilson(int(no.won.sum()), len(no))
        print(
            f"  {column:24s} yes {int(yes.won.sum())}/{len(yes)} "
            f"={yes.won.mean():.3f}[{lo1:.3f},{hi1:.3f}]  "
            f"no {int(no.won.sum())}/{len(no)} ={no.won.mean():.3f}"
            f"[{lo0:.3f},{hi0:.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
