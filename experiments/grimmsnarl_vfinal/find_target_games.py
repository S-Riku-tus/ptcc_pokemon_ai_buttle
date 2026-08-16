"""List our stored games against the three losing families of the current meta."""

from __future__ import annotations

import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV = ROOT / "experiments" / "grimmsnarl_endgame_20260816" / "version_games.csv"
TARGET = {
    "other: Conkeldurr",
    "other: Hydrapple ex",
    "Mega Lopunny / Froslass",
    "Ogerpon",
}

rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
by_family = collections.defaultdict(list)
for row in rows:
    by_family[row["opponent_family"]].append(row)

for fam in sorted(TARGET):
    games = by_family.get(fam, [])
    if not games:
        print(f"\n=== {fam}: no stored games ===")
        continue
    wins = sum(int(g["won"]) for g in games)
    print(f"\n=== {fam}: {wins}-{len(games) - wins} ===")
    print(
        f"{'episode':>10} {'ver':>6} {'won':>3} {'seat':>5} {'opp_r':>7} "
        f"{'turns':>5} {'ourT':>4} {'ourPz':>5} {'oppPz':>5} "
        f"{'shadow':>6} {'atk':>4} {'grimEv':>6} {'bodies':>6} {'boardout':>8} "
        f"{'1stShad':>7} {'oppAtk1':>7} {'hash':>17}"
    )
    for g in sorted(games, key=lambda r: (r["opponent_deck_hash"], r["episode_id"])):
        print(
            f"{g['episode_id']:>10} {g['version']:>6} {g['won']:>3} "
            f"{g['went_first'][:5]:>5} {float(g['opponent_rating'] or 0):7.0f} "
            f"{g['turns']:>5} {g['our_turns']:>4} {g['our_prize_left']:>5} "
            f"{g['opp_prize_left']:>5} {g['shadow_attacks']:>6} {g['attacks']:>4} "
            f"{g['grim_evolutions']:>6} {g['our_bodies_left']:>6} "
            f"{g['board_out']:>8} {g['own_first_shadow_turn'] or '-':>7} "
            f"{g['opp_first_attack_turn'] or '-':>7} {g['opponent_deck_hash']:>17}"
        )
