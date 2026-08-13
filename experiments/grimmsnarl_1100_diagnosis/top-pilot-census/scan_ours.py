"""Scan our own ladder runs into the same row shape as scan.py, plus the
opponent's Kaggle rating at match time (agent_N_initial_score in episodes.csv).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ml.core.replay_io import deck_hash, extract_fast_header_from_bytes  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family, archetype  # noqa: E402

OUT = Path(__file__).resolve().parent
FIRST_PLAYER_RE = re.compile(rb'"firstPlayer"\s*:\s*([01])\b')
RUNS = ROOT / "data" / "runs" / "grimmsnarl"

VERSIONS = {
    "v21": ("20260813_grimmsnarl_ml_v21_sub55456713", "55456713"),
    "v20": ("20260812_grimmsnarl_ml_v20_sub55445769", "55445769"),
    "v19b": ("20260812_grimmsnarl_ml_v19_sub55445763", "55445763"),
    "v19a": ("20260811_grimmsnarl_ml_v19_sub55428196", "55428196"),
    "v18": ("20260811_grimmsnarl_ml_v18_sub55428191", "55428191"),
    "v17": ("20260811_grimmsnarl_ml_v17_sub55423572", "55423572"),
    "v16": ("20260811_grimmsnarl_ml_v16_sub55422280", "55422280"),
    "v15": ("20260810_grimmsnarl_ml_v15_sub55404196", "55404196"),
    "v15b": ("20260811_grimmsnarl_ml_v15_b_sub55409394", "55409394"),
    "v8": ("20260807_grimmsnarl_ml_v8_sub_55317804", "55317804"),
}


def main() -> int:
    rows = []
    for label, (dirname, sub) in VERSIONS.items():
        run = RUNS / dirname
        if not run.exists():
            print(f"missing {run}", file=sys.stderr)
            continue
        meta = {}
        f = run / "episodes.csv"
        if f.exists():
            with f.open(encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    meta[int(r["episode_id"])] = r
        n = 0
        for path in sorted(run.glob("episodes/*/replay/episode_*.json")):
            eid = int(path.stem.split("_")[1])
            m = meta.get(eid, {})
            if m.get("episode_type") not in (None, "EPISODE_TYPE_PUBLIC"):
                continue
            a0, a1 = m.get("agent_0_submission_id", ""), m.get("agent_1_submission_id", "")
            if a0 and a1 and a0 == a1:
                continue
            seat = 0 if a0 == sub else (1 if a1 == sub else None)
            raw = path.read_bytes()[:3_000_000]
            h = extract_fast_header_from_bytes(raw)
            decks = h["decks"]
            if len(decks) < 2 or not decks[0] or not decks[1]:
                continue
            if seat is None:
                cand = [i for i in (0, 1)
                        if h["team_names"][i] == "yoshitaka agent"]
                if len(cand) != 1:
                    continue
                seat = cand[0]
            fp = FIRST_PLAYER_RE.search(raw)
            first = int(fp.group(1)) if fp else -1
            rw = h["rewards"]
            if rw[seat] is None:
                continue
            opp_rating = m.get(f"agent_{1 - seat}_initial_score") or ""
            own_rating = m.get(f"agent_{seat}_initial_score") or ""
            rows.append({
                "version": label, "episode_id": eid, "seat": seat,
                "won": bool(rw[seat] > (rw[1 - seat] if rw[1 - seat] is not None else 0)),
                "went_first": (first == seat) if first >= 0 else None,
                "own_hash": deck_hash(decks[seat]),
                "opp_hash": deck_hash(decks[1 - seat]),
                "opp_family": family(decks[1 - seat]),
                "opp_arch": archetype(decks[1 - seat]),
                "opp_team": h["team_names"][1 - seat],
                "opp_sub": a1 if seat == 0 else a0,
                "opp_rating": float(opp_rating) if opp_rating else None,
                "own_rating": float(own_rating) if own_rating else None,
                "create_time": m.get("create_time", ""),
            })
            n += 1
        print(f"{label}: {n} games", file=sys.stderr)

    with (OUT / "our_games.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
