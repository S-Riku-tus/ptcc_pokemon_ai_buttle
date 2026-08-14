"""Rate-denominated levers, and the mirror at n=56.

Two corrections to the count-based read:

1. ``shadow_attacks``, ``attacks`` and ``adrena_brains`` are counts accumulated
   over a game, so "more is better" can be nothing but "my winning games had
   more turns to do it in".  Every lever here is re-expressed per own MAIN turn
   before it is believed, the denominator this repo already had to learn once.
2. The mirror is 56 of 194 games and the single largest block of losses, so it
   gets its own contrast: winners against losers on the same board.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

ELO = 400.0 / math.log(10.0)
GAMES = ROOT / "experiments" / "grimmsnarl_ml_v23" / "ladder_v22_v23_games.csv"
OUT = ROOT / "experiments" / "grimmsnarl_ml_v23" / "rates_and_mirror.json"

NUM = ("opponent_rating", "own_first_shadow_turn", "own_first_ready_turn",
       "shadow_attacks", "attacks", "grim_evolutions", "rare_candies",
       "adrena_brains", "stamps", "bosses", "froslass_evolves", "our_turns",
       "turns", "our_prize_left", "opp_prize_left", "our_bodies_left",
       "our_deck_left", "opp_deck_left", "first_shadow_turn",
       "first_ready_turn", "opp_first_attack_turn")


def load() -> list[dict]:
    rows = []
    for r in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        row = dict(r)
        row["won"] = r["won"] == "True"
        row["went_first"] = {"True": True, "False": False}.get(r["went_first"])
        for k in NUM:
            row[k] = None if r.get(k, "") in ("", None) else float(r[k])
        rows.append(row)
    return rows


def blk(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"games": 0}
    w = sum(1 for r in rows if r["won"])
    lo, hi = wilson(w, n)
    return {"games": n, "wins": w, "win_rate": round(w / n, 3),
            "wilson95": [round(lo, 3), round(hi, 3)]}


def fit(rows: list[dict], terms: dict[str, Callable[[dict], float | None]],
        controls: bool = True) -> dict:
    names, X, y = list(terms), [], []
    for r in rows:
        vec = []
        if controls:
            if r["opponent_rating"] is None or r["went_first"] is None:
                continue
            vec += [r["opponent_rating"] / 400.0, 1.0 if r["went_first"] else 0.0]
        vals = [fn(r) for fn in terms.values()]
        if any(v is None for v in vals):
            continue
        X.append(vec + vals)
        y.append(1 if r["won"] else 0)
    full = (["opp_rating/400", "went_first"] if controls else []) + names
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(y) < 12 or len(set(y.tolist())) < 2:
        return {"n": int(len(y)), "error": "insufficient"}
    m = LogisticRegression(penalty=None, max_iter=8000).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    Xd = np.hstack([X, np.ones((len(X), 1))])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(Xd.T @ np.diag(p * (1 - p)) @ Xd)))[:-1]
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], float("nan"))
    out = {"n": int(len(y)), "terms": {}}
    for name, b, s in zip(full, m.coef_[0], se):
        z = b / s if s and not math.isnan(s) else float("nan")
        out["terms"][name] = {
            "elo": round(float(b) * ELO, 1),
            "z": round(float(z), 2) if not math.isnan(z) else None,
            "p": round(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), 4)
            if not math.isnan(z) else None,
        }
    return out


def tercile(rows: list[dict], fn: Callable[[dict], float | None]) -> dict:
    vals = sorted(v for v in (fn(r) for r in rows) if v is not None)
    if len(vals) < 9:
        return {}
    lo_cut, hi_cut = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    b: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = fn(r)
        if v is None:
            continue
        b["low" if v < lo_cut else ("high" if v >= hi_cut else "mid")].append(r)
    return {"cuts": [round(lo_cut, 3), round(hi_cut, 3)],
            **{k: blk(v) for k, v in b.items()}}


def main() -> int:
    rows = load()
    v22 = [r for r in rows if r["version"].startswith("v22")]
    report: dict[str, Any] = {}

    # --- rate-denominated levers -------------------------------------------
    def per_turn(key: str) -> Callable[[dict], float | None]:
        def fn(r: dict) -> float | None:
            if r[key] is None or not r["our_turns"]:
                return None
            return r[key] / r["our_turns"]
        return fn

    rate_levers = {}
    for key in ("shadow_attacks", "attacks", "adrena_brains", "stamps",
                "bosses", "rare_candies", "froslass_evolves", "grim_evolutions"):
        rate_levers[f"{key}_per_own_turn"] = {
            "controlled": fit(v22, {key: per_turn(key)})["terms"].get(key),
            "tercile": tercile(v22, per_turn(key)),
            "count_controlled_for_length": fit(
                v22, {key: (lambda r, k=key: r[k]),
                      "our_turns": lambda r: r["our_turns"]})["terms"].get(key),
        }
    report["rate_levers"] = rate_levers

    # game length itself
    report["game_length"] = {
        "our_turns_tercile": tercile(v22, lambda r: r["our_turns"]),
        "mean_our_turns_win": round(sum(
            r["our_turns"] for r in v22 if r["won"]) / sum(1 for r in v22 if r["won"]), 2),
        "mean_our_turns_loss": round(sum(
            r["our_turns"] for r in v22 if not r["won"]) / sum(1 for r in v22 if not r["won"]), 2),
    }

    # --- the mirror ---------------------------------------------------------
    mirror = [r for r in v22 if r["opponent_family"] == "Grimmsnarl (mirror)"]
    others = [r for r in v22 if r["opponent_family"] != "Grimmsnarl (mirror)"]
    mw = [r for r in mirror if r["won"]]
    ml = [r for r in mirror if not r["won"]]

    def contrast(a: list[dict], b: list[dict], keys: list[str]) -> dict:
        out = {}
        for k in keys:
            va = [r[k] for r in a if r[k] is not None]
            vb = [r[k] for r in b if r[k] is not None]
            if not va or not vb:
                continue
            ma, mb = sum(va) / len(va), sum(vb) / len(vb)
            sa = (sum((x - ma) ** 2 for x in va) / max(1, len(va) - 1)) ** 0.5
            sb = (sum((x - mb) ** 2 for x in vb) / max(1, len(vb) - 1)) ** 0.5
            se = ((sa ** 2 / len(va)) + (sb ** 2 / len(vb))) ** 0.5
            t = (ma - mb) / se if se else float("nan")
            out[k] = {"win_mean": round(ma, 3), "loss_mean": round(mb, 3),
                      "diff": round(ma - mb, 3),
                      "t": round(t, 2) if not math.isnan(t) else None}
        return out

    keys = ["own_first_shadow_turn", "own_first_ready_turn", "shadow_attacks",
            "attacks", "adrena_brains", "stamps", "bosses", "rare_candies",
            "froslass_evolves", "grim_evolutions", "our_turns", "turns",
            "opp_first_attack_turn", "opponent_rating"]
    report["mirror"] = {
        "overall": blk(mirror),
        "others": blk(others),
        "turn_order": {"first": blk([r for r in mirror if r["went_first"]]),
                       "second": blk([r for r in mirror if r["went_first"] is False])},
        "win_vs_loss": contrast(mw, ml, keys),
        "controlled_in_mirror": fit(mirror, {
            "shadow_per_turn": per_turn("shadow_attacks"),
            "own_first_shadow_turn": lambda r: r["own_first_shadow_turn"],
        }),
        "prizes_taken_on_loss": {
            str(int(6 - (r["our_prize_left"] or 6))): 0 for r in ml},
        "opp_first_attack": {
            "we_attack_first": blk([r for r in mirror
                                    if r["first_shadow_turn"] is not None
                                    and r["opp_first_attack_turn"] is not None
                                    and r["first_shadow_turn"] < r["opp_first_attack_turn"]]),
            "they_attack_first": blk([r for r in mirror
                                      if r["first_shadow_turn"] is not None
                                      and r["opp_first_attack_turn"] is not None
                                      and r["first_shadow_turn"] > r["opp_first_attack_turn"]]),
        },
    }
    counter: dict[str, int] = defaultdict(int)
    for r in ml:
        counter[str(int(6 - (r["our_prize_left"] or 6)))] += 1
    report["mirror"]["prizes_taken_on_loss"] = dict(sorted(counter.items()))

    # same contrast pooled over every matchup, for reference
    report["all_games_win_vs_loss"] = contrast(
        [r for r in v22 if r["won"]], [r for r in v22 if not r["won"]], keys)

    # --- who actually beats us ---------------------------------------------
    by_opp: dict[str, list[dict]] = defaultdict(list)
    for r in v22:
        by_opp[r["opponent_submission"]].append(r)
    repeat = {k: blk(v) for k, v in by_opp.items() if len(v) >= 3}
    report["repeat_opponents"] = dict(sorted(
        repeat.items(), key=lambda i: i[1]["win_rate"])[:12])
    report["distinct_opponents"] = len(by_opp)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
