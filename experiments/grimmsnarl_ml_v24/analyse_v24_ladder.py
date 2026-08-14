"""Did v24 do anything, and if not, where is the rating actually lost?

Reads the per-episode table built by ``scripts/analyze_grimmsnarl_v20_ladder.py``
over v22's four byte-identical submissions (194 games) and v24's two (87), and
answers three questions in order:

1. **Is v24 distinguishable from v22 at all?**  A logistic fit of ``won`` on
   opponent rating, turn order and an ``is_v24`` indicator.  The v22 pool
   already calibrates the same-code noise floor, so the v24 term is read
   against that, not against zero.
2. **Where does the pooled policy lose?**  Matchup family and opponent-rating
   band, with Wilson bounds, because a 39-game run cannot separate a matchup
   from a pairing draw.
3. **What does a loss look like?**  Prizes conceded, game length, and whether
   we were still contesting the board when it ended.

Nothing here loads a model.  Every column is an observed replay fact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

ELO = 400.0 / math.log(10.0)


def fnum(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load(path: Path) -> list[dict]:
    rows = []
    for raw in csv.DictReader(path.open(encoding="utf-8-sig")):
        version = raw["version"]
        if not (version.startswith("v22") or version.startswith("v24")):
            continue
        row = dict(raw)
        row["won"] = raw["won"].strip().lower() in ("true", "1")
        row["went_first"] = (
            None if raw["went_first"] in ("", None)
            else raw["went_first"].strip().lower() in ("true", "1")
        )
        row["opponent_rating"] = fnum(raw, "opponent_rating")
        row["is_v24"] = 1.0 if version.startswith("v24") else 0.0
        rows.append(row)
    return rows


def block(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"games": 0}
    wins = sum(1 for r in rows if r["won"])
    low, high = wilson(wins, n)
    opp = [r["opponent_rating"] for r in rows if r["opponent_rating"] is not None]
    return {
        "games": n,
        "record": f"{wins}-{n - wins}",
        "win_rate": round(wins / n, 4),
        "wilson95": [round(low, 4), round(high, 4)],
        "opp_mean": round(sum(opp) / len(opp), 1) if opp else None,
    }


def fit(rows: list[dict], extra: dict[str, Callable[[dict], float | None]]) -> dict[str, Any]:
    usable = [
        r for r in rows
        if r["opponent_rating"] is not None and r["went_first"] is not None
    ]
    names = ["opp_rating/400", "went_first"] + list(extra)
    X, y = [], []
    for r in usable:
        vec = [r["opponent_rating"] / 400.0, 1.0 if r["went_first"] else 0.0]
        vec += [fn(r) for fn in extra.values()]
        if any(v is None for v in vec):
            continue
        X.append(vec)
        y.append(1 if r["won"] else 0)
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(set(y.tolist())) < 2 or len(y) < 12:
        return {"n": int(len(y)), "error": "insufficient variation"}
    model = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    beta = model.coef_[0]
    p = model.predict_proba(X)[:, 1]
    Xd = np.hstack([X, np.ones((len(X), 1))])
    W = np.diag(p * (1 - p))
    try:
        se = np.sqrt(np.diag(np.linalg.inv(Xd.T @ W @ Xd)))[:-1]
    except np.linalg.LinAlgError:
        se = np.full(len(beta), float("nan"))
    out: dict[str, Any] = {"n": int(len(y)), "wins": int(y.sum()), "terms": {}}
    for name, b, s in zip(names, beta, se):
        z = b / s if s and not math.isnan(s) else float("nan")
        out["terms"][name] = {
            "elo": round(float(b) * ELO, 1),
            "z": round(float(z), 2) if not math.isnan(z) else None,
            "p": round(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), 4)
            if not math.isnan(z) else None,
        }
    return out


def by_key(rows: list[dict], key: Callable[[dict], str]) -> dict[str, Any]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return {
        label: block(items)
        for label, items in sorted(
            buckets.items(), key=lambda kv: -len(kv[1])
        )
    }


def rating_band(row: dict) -> str:
    rating = row["opponent_rating"]
    if rating is None:
        return "unknown"
    for edge in (700, 800, 900, 1000, 1100):
        if rating < edge:
            return f"<{edge}"
    return ">=1100"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v24" / "ladder_v24_games.csv")
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v24" / "v24_verdict.json")
    args = parser.parse_args()

    rows = load(args.games)
    v22 = [r for r in rows if not r["is_v24"]]
    v24 = [r for r in rows if r["is_v24"]]

    payload: dict[str, Any] = {
        "per_submission": by_key(rows, lambda r: r["version"]),
        "pools": {"v22": block(v22), "v24": block(v24), "all": block(rows)},
        "is_v24_controlled": fit(rows, {"is_v24": lambda r: r["is_v24"]}),
        "same_code_noise_v22_only": {
            label: fit(v22, {label: lambda r, lab=label: 1.0 if r["version"] == lab else 0.0})
            .get("terms", {}).get(label)
            for label in sorted({r["version"] for r in v22})
        },
        "opponent_rating_band": {
            "v22": by_key(v22, rating_band),
            "v24": by_key(v24, rating_band),
            "pooled": by_key(rows, rating_band),
        },
        "matchup_family": {
            "pooled": by_key(rows, lambda r: r["opponent_family"] or "unknown"),
            "v24": by_key(v24, lambda r: r["opponent_family"] or "unknown"),
        },
        "mirror_exact_list": {
            "pooled": by_key(
                [r for r in rows if r["opponent_deck_hash"] == "9714ab5c3996f6cc"],
                lambda r: "mirror"),
            "v24": by_key(
                [r for r in v24 if r["opponent_deck_hash"] == "9714ab5c3996f6cc"],
                lambda r: "mirror"),
        },
        "turn_order": {
            "v22": by_key(v22, lambda r: "first" if r["went_first"] else "second"),
            "v24": by_key(v24, lambda r: "first" if r["went_first"] else "second"),
        },
        "froslass_evolution_split": {
            pool_name: by_key(
                pool,
                lambda r: "0" if (fnum(r, "froslass_evolves") or 0) == 0 else "1+")
            for pool_name, pool in (("v22", v22), ("v24", v24))
        },
        "loss_shape": {
            pool_name: {
                "prizes_we_took_on_loss": dict(sorted(Counter(
                    int(6 - (fnum(r, "our_prize_left") or 6))
                    for r in pool if not r["won"]).items())),
                "mean_turns_win": round(float(np.mean(
                    [fnum(r, "turns") or 0 for r in pool if r["won"]])), 2),
                "mean_turns_loss": round(float(np.mean(
                    [fnum(r, "turns") or 0 for r in pool if not r["won"]])), 2),
                "bodies_left_on_loss": dict(sorted(Counter(
                    int(fnum(r, "our_bodies_left") or 0)
                    for r in pool if not r["won"]).items())),
            }
            for pool_name, pool in (("v22", v22), ("v24", v24))
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
