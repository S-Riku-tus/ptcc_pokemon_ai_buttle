"""Decompose the own-turn-2 gap into setup, draw and decision components."""

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


def ci(hits: int, n: int) -> str:
    low, high = wilson(hits, n)
    return f"{hits}/{n}={hits / max(1, n):.3f}[{low:.3f},{high:.3f}]"


def main() -> int:
    games = pd.read_parquet(OUT / "open_games.parquet")
    rows = pd.read_parquet(OUT / "open_turns.parquet").merge(
        games[["source", "episode_id", "seat", "won", "went_first"]],
        on=["source", "episode_id", "seat"],
    )
    v21g = games[games.source == "v21"]
    fieldg = games[games.source == "field"]
    top5g = games[games.pilot.isin(TOP5)]

    print("=== setup board composition, with tests and rating gradient ===")
    checks = [
        ("active_is_impidimp", lambda f: (f.setup_active_id == 646).astype(int)),
        ("active_is_munkidori", lambda f: (f.setup_active_id == 112).astype(int)),
        ("active_is_snorunt", lambda f: (f.setup_active_id == 860).astype(int)),
        ("impidimp_total>=1", lambda f: (f.setup_impidimp_total >= 1).astype(int)),
        ("impidimp_total>=2", lambda f: (f.setup_impidimp_total >= 2).astype(int)),
        ("bench_size>=1", lambda f: (f.setup_bench_size >= 1).astype(int)),
        ("bench_munkidori>=1", lambda f: (f.setup_bench_munkidori >= 1).astype(int)),
        ("bench_snorunt>=1", lambda f: (f.setup_bench_snorunt >= 1).astype(int)),
    ]
    out = []
    for name, fn in checks:
        a, b, c = fn(v21g), fn(top5g), fn(fieldg)
        p = stats.fisher_exact([
            [int(a.sum()), len(a) - int(a.sum())],
            [int(c.sum()), len(c) - int(c.sum())],
        ])[1]
        per = fieldg.assign(value=c).groupby(["pilot", "rating"]).value.mean()
        per = per.reset_index()
        rho, prho = stats.spearmanr(per.rating, per.value)
        out.append({
            "metric": name, "v21": ci(int(a.sum()), len(a)),
            "top5": round(float(b.mean()), 3), "field": round(float(c.mean()), 3),
            "rank/23": int((per.value < a.mean()).sum() + 1),
            "pmin": round(float(per.value.min()), 3),
            "pmax": round(float(per.value.max()), 3),
            "p_vs_field": round(float(p), 4),
            "rho_rating": round(float(rho), 3), "p_rho": round(float(prho), 4),
        })
    print(pd.DataFrame(out).to_string(index=False))

    print("\n=== does the setup board predict winning inside the field? ===")
    for name, fn in checks:
        flag = fn(fieldg)
        yes = fieldg[flag == 1]
        no = fieldg[flag == 0]
        p = stats.fisher_exact([
            [int(yes.won.sum()), len(yes) - int(yes.won.sum())],
            [int(no.won.sum()), len(no) - int(no.won.sum())],
        ])[1]
        print(f"  {name:22s} yes {ci(int(yes.won.sum()), len(yes))} | "
              f"no {ci(int(no.won.sum()), len(no))}  p={p:.2e}")

    print("\n=== own-turn-2 Rare Candy legality, decomposed ===")
    t2 = rows[rows.own_turn == 2]
    for label, frame in [
        ("v21", t2[t2.source == "v21"]),
        ("top5", t2[t2.pilot.isin(TOP5)]),
        ("field", t2[t2.source == "field"]),
    ]:
        impidimp = frame[frame.play_646 > 0]
        hold = frame[frame.hand_1079 > 0]
        both = frame[(frame.hand_1079 > 0) & (frame.play_646 > 0)]
        print(
            f"  {label:6s} n={len(frame):5d} "
            f"impidimp_in_play={float((frame.play_646 > 0).mean()):.3f} "
            f"hold_candy={float((frame.hand_1079 > 0).mean()):.3f} "
            f"both={len(both) / max(1, len(frame)):.3f} "
            f"candy_legal|both={float((both.offer_1079 > 0).mean()):.3f} "
            f"play|legal={float((frame[frame.offer_1079 > 0].take_1079 > 0).mean()):.3f}"
        )
    print("  (grim ex must also be in hand or reachable; hold_candy+grim+impidimp:)")
    for label, frame in [
        ("v21", t2[t2.source == "v21"]),
        ("top5", t2[t2.pilot.isin(TOP5)]),
        ("field", t2[t2.source == "field"]),
    ]:
        trio = frame[
            (frame.hand_1079 > 0) & (frame.play_646 > 0) & (frame.hand_648 > 0)
        ]
        print(f"    {label:6s} trio={len(trio)}/{len(frame)}="
              f"{len(trio) / max(1, len(frame)):.3f} "
              f"played={float((trio.take_1079 > 0).mean()) if len(trio) else float('nan'):.3f}")

    print("\n=== Lillie's Determination: take-when-offered, per own turn ===")
    for turn in (1, 2, 3, 4):
        line = []
        for label, frame in [
            ("v21", rows[(rows.source == "v21") & (rows.own_turn == turn)]),
            ("top5", rows[rows.pilot.isin(TOP5) & (rows.own_turn == turn)]),
            ("field", rows[(rows.source == "field") & (rows.own_turn == turn)]),
        ]:
            offered = frame[frame.offer_1227 > 0]
            taken = int((offered.take_1227 > 0).sum())
            line.append(f"{label}: {ci(taken, len(offered))}")
        print(f"  t{turn}  " + " | ".join(line))
        offered = rows[(rows.source == "field") & (rows.own_turn == turn)
                       & (rows.offer_1227 > 0)]
        per = offered.groupby(["pilot", "rating"]).apply(
            lambda g: float((g.take_1227 > 0).mean()), include_groups=False
        ).reset_index(name="value")
        rho, p = stats.spearmanr(per.rating, per.value)
        ours = rows[(rows.source == "v21") & (rows.own_turn == turn)
                    & (rows.offer_1227 > 0)]
        rate = float((ours.take_1227 > 0).mean()) if len(ours) else float("nan")
        print(f"      rho_vs_rating={rho:.3f} p={p:.4f}  pilot range "
              f"[{per.value.min():.3f},{per.value.max():.3f}]  v21 rank="
              f"{int((per.value < rate).sum() + 1)}/23")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
