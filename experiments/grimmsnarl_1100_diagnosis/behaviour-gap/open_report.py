"""Offer-side vs take-side for the own-turn-2 gap, plus the setup board.

The hand multiset is read at the FIRST MAIN decision of each own turn, so
"held Rare Candy at own turn 2" is a fact about the draw, and
"held it and did not play it" is a fact about the policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
TOP5 = ["55167115", "55171940", "55177269", "55090635", "55170504"]
NAMES = {
    7: "DarkEnergy", 104: "Froslass", 112: "Munkidori", 646: "Impidimp",
    647: "Morgrem", 648: "GrimmsnarlEX", 860: "Snorunt", 1079: "RareCandy",
    1080: "UnfairStamp", 1086: "Poffin", 1097: "NightStretcher",
    1122: "Pokegear", 1137: "ToolScrapper", 1152: "PokePad", 1182: "Boss",
    1219: "Petrel", 1227: "Lillie", 1231: "Dawn", 1259: "SpikemuthGym",
}


def block(frame: pd.DataFrame, column: str) -> str:
    hits = int((frame[column] > 0).sum())
    low, high = wilson(hits, len(frame))
    return f"{hits / max(1, len(frame)):.3f}[{low:.3f},{high:.3f}] n={len(frame)}"


def main() -> int:
    games = pd.read_parquet(OUT / "open_games.parquet")
    rows = pd.read_parquet(OUT / "open_turns.parquet")
    rows = rows.merge(
        games[["source", "episode_id", "seat", "won", "went_first"]],
        on=["source", "episode_id", "seat"],
    )
    groups = {
        "v21": rows[rows.source == "v21"],
        "top5": rows[rows.pilot.isin(TOP5)],
        "field": rows[rows.source == "field"],
    }
    gmeta = {
        "v21": games[games.source == "v21"],
        "top5": games[games.pilot.isin(TOP5)],
        "field": games[games.source == "field"],
    }

    print("=== setup board (state at the first observation of shared turn 1) ===")
    for name, frame in gmeta.items():
        counts = frame.setup_active_id.value_counts(normalize=True)
        top = ", ".join(
            f"{NAMES.get(int(k), k)}={v:.3f}" for k, v in counts.head(5).items()
        )
        print(f"  {name:6s} n={len(frame)} active: {top}")
        print(f"         bench_size={frame.setup_bench_size.mean():.3f} "
              f"impidimp_total={frame.setup_impidimp_total.mean():.3f} "
              f"bench_snorunt={frame.setup_bench_snorunt.mean():.3f} "
              f"bench_munkidori={frame.setup_bench_munkidori.mean():.3f}")

    print("\n=== own turn 2: hold vs play, Rare Candy / Grimmsnarl ex ===")
    for seat_label, seat_value in [("all", None), ("first", 1), ("second", 0)]:
        print(f"  -- {seat_label} --")
        for name, frame in groups.items():
            f = frame[frame.own_turn == 2]
            if seat_value is not None:
                f = f[f.went_first == seat_value]
            hold_candy = f[f.hand_1079 > 0]
            hold_both = f[(f.hand_1079 > 0) & (f.hand_648 > 0)]
            offered = f[f.offer_1079 > 0]
            played = f[f.take_1079 > 0]
            print(
                f"    {name:6s} turns={len(f):5d} "
                f"hold_candy={len(hold_candy) / max(1, len(f)):.3f} "
                f"hold_candy+grim={len(hold_both) / max(1, len(f)):.3f} "
                f"candy_offered={len(offered) / max(1, len(f)):.3f} "
                f"candy_played={len(played) / max(1, len(f)):.3f} "
                f"play|offered={len(played) / max(1, len(offered)):.3f} "
                f"play|hold_both={int((hold_both.take_1079 > 0).sum()) / max(1, len(hold_both)):.3f}"
            )

    print("\n=== own turn 1 and 2: key cards HELD (mean copies in hand) ===")
    for turn in (1, 2, 3):
        print(f"  -- own turn {turn} --")
        table = {}
        for name, frame in groups.items():
            f = frame[frame.own_turn == turn]
            table[name] = {
                NAMES[cid]: round(float(f[f"hand_{cid}"].mean()), 3)
                for cid in NAMES
            }
            table[name]["hand_size"] = round(float(f.hand_size.mean()), 3)
            table[name]["deck_left"] = round(float(f.deck_left.mean()), 3)
        frame = pd.DataFrame(table)
        frame["v21-top5"] = (frame["v21"] - frame["top5"]).round(3)
        print(frame.sort_values("v21-top5").to_string())

    print("\n=== own turn 1 and 2: Spikemuth Gym hold / offer / play ===")
    for turn in (1, 2):
        for name, frame in groups.items():
            f = frame[frame.own_turn == turn]
            print(
                f"  t{turn} {name:6s} hold={float((f.hand_1259 > 0).mean()):.3f} "
                f"offered={float((f.offer_1259 > 0).mean()):.3f} "
                f"played={float((f.take_1259 > 0).mean()):.3f} "
                f"play|offered={float((f[f.offer_1259 > 0].take_1259 > 0).mean()):.3f} "
                f"n={len(f)}"
            )

    print("\n=== own turn 1 and 2: draw supporters hold / offer / play ===")
    for cid in (1227, 1219, 1231):
        for turn in (1, 2):
            line = []
            for name, frame in groups.items():
                f = frame[frame.own_turn == turn]
                offered = f[f[f"offer_{cid}"] > 0]
                line.append(
                    f"{name}: hold={float((f[f'hand_{cid}'] > 0).mean()):.3f} "
                    f"off={float((f[f'offer_{cid}'] > 0).mean()):.3f} "
                    f"play={float((f[f'take_{cid}'] > 0).mean()):.3f} "
                    f"p|o={float((offered[f'take_{cid}'] > 0).mean()) if len(offered) else float('nan'):.3f}"
                )
            print(f"  {NAMES[cid]:12s} t{turn}  " + " | ".join(line))

    print("\n=== field: outcome by own-turn-2 Rare Candy, split by whether it was held ===")
    f = groups["field"]
    t2 = f[f.own_turn == 2]
    held = t2[t2.hand_1079 > 0]
    print(f"  held candy at t2: {len(held)}/{len(t2)} = {len(held)/len(t2):.3f}, "
          f"won {held.won.mean():.3f}; not held won {t2[t2.hand_1079 == 0].won.mean():.3f}")
    o = groups["v21"]
    t2o = o[o.own_turn == 2]
    heldo = t2o[t2o.hand_1079 > 0]
    p = stats.fisher_exact([
        [len(heldo), len(t2o) - len(heldo)],
        [len(held), len(t2) - len(held)],
    ])[1]
    print(f"  v21 held candy at t2: {len(heldo)}/{len(t2o)} = "
          f"{len(heldo)/max(1,len(t2o)):.3f}  p_vs_field={p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
