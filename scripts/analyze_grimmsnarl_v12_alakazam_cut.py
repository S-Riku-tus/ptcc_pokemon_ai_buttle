"""The one matchup that is still a defect: Alakazam, going second."""

from __future__ import annotations

import json
from pathlib import Path

ROWS = [json.loads(line) for line in
        (Path(__file__).resolve().parents[1] / "experiments"
         / "grimmsnarl_ml_v12" / "tempo_rows.jsonl").read_text("utf-8").splitlines()
        if line.strip()]
OURS = {"v8", "v9", "v11", "v11_a", "v11_b"}


def mean_at(rows, key, t):
    v = [r[key].get(str(t)) for r in rows if r[key].get(str(t)) is not None]
    return sum(v) / len(v) if v else None


def line(rows, key, label):
    cells = [f"{(mean_at(rows, key, t) or 0):5.2f}" if mean_at(rows, key, t) is not None
             else "    -" for t in range(1, 8)]
    return f"    {label:<24}" + " ".join(cells)


def block(rows, label):
    if not rows:
        print(f"  {label}: n=0")
        return
    k = sum(r["won"] for r in rows)
    print(f"  {label:<40} {k}-{len(rows)-k} ({k/len(rows):.3f})  n={len(rows)}")
    for key, name in (("bodies", "bodies"), ("grimm", "GrimmEx in play"),
                      ("attacked", "attacked this turn")):
        print(line(rows, key, name))
    fa = [r["first_attack"] for r in rows if r["first_attack"]]
    fg = [r["first_grimm"] for r in rows if r["first_grimm"]]
    turns = [r["own_turns"] for r in rows]
    print(f"    first attack own-turn   mean={sum(fa)/len(fa):.2f}"
          f"  never={sum(1 for r in rows if not r['first_attack'])}/{len(rows)}"
          if fa else "    never attacked")
    print(f"    first GrimmEx own-turn  mean={sum(fg)/len(fg):.2f}"
          f"  never={sum(1 for r in rows if not r['first_grimm'])}/{len(rows)}"
          if fg else "    GrimmEx never landed")
    print(f"    own turns in game       mean={sum(turns)/len(turns):.2f}")


for opp in ("Alakazam",):
    print("=" * 78)
    print(f"OPPONENT: {opp}")
    for second in (False, True):
        tag = "going second" if second else "going first"
        f = [r for r in ROWS if r["tag"] == "field" and r["opponent"] == opp
             and r["went_first"] is (not second)]
        o = [r for r in ROWS if r["tag"] in OURS and r["opponent"] == opp
             and r["went_first"] is (not second)]
        print(f"\n-- {tag} --")
        block(f, "FIELD")
        block(o, "OURS")
        # field, split by outcome, to see which lever the winners pull
        block([r for r in f if r["won"]], "FIELD wins")
        block([r for r in f if not r["won"]], "FIELD losses")

print("\n" + "=" * 78)
print("Mirror control, going second")
f = [r for r in ROWS if r["tag"] == "field" and r["opponent"] == "Marnie's Grimmsnarl ex"
     and r["went_first"] is False]
o = [r for r in ROWS if r["tag"] in OURS and r["opponent"] == "Marnie's Grimmsnarl ex"
     and r["went_first"] is False]
block(f, "FIELD")
block(o, "OURS")
