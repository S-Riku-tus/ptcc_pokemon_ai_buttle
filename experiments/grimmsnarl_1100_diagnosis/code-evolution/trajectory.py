"""Version trajectory: does added code buy anything measurable?

Per submitted version: code size of the exact submitted snapshot, the reported
final rating, and the win rate / Elo residual computed from ``episodes.csv``
(win = our updated_score > our initial_score; a rating tick is monotone in the
result for this ladder, cross-checked against the replay-derived
``ladder_history_games.csv``).
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "data" / "runs" / "grimmsnarl"

# label -> (run dir, submission id, reported final rating)
LADDER = [
    ("v1", "20260803_grimmsnarl_ml_v1_sub55185513", 55185513, 871.0),
    ("v2", "20260803_grimmsnarl_ml_v2_sub55205556", 55205556, 967.4),
    ("v3a", "20260805_grimmsnarl_ml_v3_sub55216787", 55216787, 996.6),
    ("v3b", "20260805_grimmsnarl_ml_v3_sub55217233", 55217233, 907.6),
    ("v4", "20260805_grimmsnarl_ml_v4_sub55253296", 55253296, 1031.2),
    ("v4.5", "20260806_grimmsnarl_ml_v4_5_sub55275464", 55275464, 979.1),
    ("v5", "20260806_grimmsnarl_ml_v5_sub55275642", 55275642, 963.6),
    ("v6", "20260806_grimmsnarl_ml_v6_sub55290882", 55290882, 996.6),
    ("v7", "20260806_grimmsnarl_ml_v7_sub_55302846", 55302846, 943.3),
    ("v8", "20260807_grimmsnarl_ml_v8_sub_55317804", 55317804, 1035.8),
    ("v9", "20260807_grimmsnarl_ml_v9_sub_55325029", 55325029, 994.9),
    ("v11a", "20260808_grimmsnarl_ml_v11_a_sub55346539", 55346539, 874.2),
    ("v11b", "20260808_grimmsnarl_ml_v11_b_sub55346548", 55346548, 935.8),
    ("v11", "20260809_grimmsnarl_ml_v11_sub55353978", 55353978, 950.3),
    ("v12a", "20260809_grimmsnarl_ml_v12_a_sub55373676", 55373676, 914.8),
    ("v12b", "20260809_grimmsnarl_ml_v12_b_sub55374240", 55374240, 894.6),
    ("v13a", "20260810_grimmsnarl_ml_v13_a_sub55380882", 55380882, 942.5),
    ("v13b", "20260810_grimmsnarl_ml_v13_b_sub55380958", 55380958, 964.8),
    ("v14", "20260810_grimmsnarl_ml_v14_sub55395386", 55395386, 927.6),
    ("v15", "20260810_grimmsnarl_ml_v15_sub55404196", 55404196, 1007.7),
    ("v15b", "20260811_grimmsnarl_ml_v15_b_sub55409394", 55409394, 862.0),
    ("v16", "20260811_grimmsnarl_ml_v16_sub55422280", 55422280, 955.8),
    ("v17", "20260811_grimmsnarl_ml_v17_sub55423572", 55423572, 896.8),
    ("v18", "20260811_grimmsnarl_ml_v18_sub55428191", 55428191, 779.5),
    ("v19a", "20260811_grimmsnarl_ml_v19_sub55428196", 55428196, 978.3),
    ("v19b", "20260812_grimmsnarl_ml_v19_sub55445763", 55445763, 904.6),
    ("v20", "20260812_grimmsnarl_ml_v20_sub55445769", 55445769, 982.0),
    ("v21", "20260813_grimmsnarl_ml_v21_sub55456713", 55456713, 948.2),
]

GUARD_RE = re.compile(r"\bif\b|\breturn index\b")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def snapshot_stats(run_dir: Path) -> dict:
    snap = run_dir / "deck_snapshot"
    if not snap.exists():
        return {}
    files = sorted(snap.glob("*.py"))
    loc = sum(
        len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        for p in files
    )
    guards = 0
    returns_index = 0
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        guards += len(re.findall(r"^\s*if ", text, re.M))
        returns_index += len(re.findall(r"return index", text))
    model = snap / "ranker_model.json"
    trees = features = None
    if model.exists():
        data = json.loads(model.read_text(encoding="utf-8"))
        trees = len(data.get("trees") or [])
        features = len(data.get("feature_names") or [])
    return {
        "py_files": len(files),
        "loc": loc,
        "if_statements": guards,
        "return_index": returns_index,
        "trees": trees,
        "model_features": features,
        "model_bytes": model.stat().st_size if model.exists() else None,
    }


def ladder_stats(run_dir: Path, submission: int) -> dict:
    path = run_dir / "episodes.csv"
    if not path.exists():
        return {}
    wins = losses = ties = 0
    opp_ratings: list[float] = []
    expected = 0.0
    final_rating = None
    last_time = ""
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        if row.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        if row.get("state") != "COMPLETED":
            continue
        seat = None
        for s in (0, 1):
            if row[f"agent_{s}_submission_id"] == str(submission):
                seat = s
        if seat is None:
            continue
        other = 1 - seat
        if row[f"agent_{other}_submission_id"] == str(submission):
            continue  # self-play validation
        if f"agent_{seat}_initial_score" not in row:
            continue
        try:
            init = float(row[f"agent_{seat}_initial_score"])
            upd = float(row[f"agent_{seat}_updated_score"])
            opp = float(row[f"agent_{other}_initial_score"])
        except (TypeError, ValueError):
            continue
        if row["create_time"] > last_time:
            last_time = row["create_time"]
            final_rating = upd
        opp_ratings.append(opp)
        expected += 1.0 / (1.0 + 10 ** ((opp - init) / 400.0))
        if upd > init:
            wins += 1
        elif upd < init:
            losses += 1
        else:
            ties += 1
    n = wins + losses + ties
    lo, hi = wilson(wins, n)
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / n, 4) if n else None,
        "win_lo": round(lo, 4),
        "win_hi": round(hi, 4),
        "mean_opponent": round(sum(opp_ratings) / n, 1) if n else None,
        "elo_expected": round(expected / n, 4) if n else None,
        "elo_residual": round(wins / n - expected / n, 4) if n else None,
        "last_updated_score": final_rating,
    }


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else 0.0


def perm_p(xs: list[float], ys: list[float], trials: int = 20000) -> float:
    import random
    rng = random.Random(20260813)
    observed = abs(spearman(xs, ys))
    hits = 0
    pool = list(ys)
    for _ in range(trials):
        rng.shuffle(pool)
        if abs(spearman(xs, pool)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (trials + 1)


def main() -> None:
    rows = []
    for label, run, sub, rating in LADDER:
        run_dir = RUNS / run
        row = {"version": label, "submission": sub, "reported_rating": rating}
        row.update(snapshot_stats(run_dir))
        row.update(ladder_stats(run_dir, sub))
        rows.append(row)

    out = ROOT / "experiments" / "grimmsnarl_1100_diagnosis" / \
        "code-evolution" / "trajectory.json"
    header = [
        "version", "submission", "reported_rating", "games", "wins",
        "losses", "ties", "win_rate", "win_lo", "win_hi", "mean_opponent",
        "elo_expected", "elo_residual", "py_files", "loc", "if_statements",
        "return_index", "trees", "model_features", "model_bytes",
    ]
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in header))

    # correlations over versions that have both a code snapshot and games
    usable = [
        r for r in rows
        if r.get("loc") and r.get("games") and r.get("elo_residual") is not None
    ]
    stats = {}
    for xkey in ("loc", "py_files", "if_statements"):
        for ykey in ("reported_rating", "win_rate", "elo_residual"):
            xs = [float(r[xkey]) for r in usable]
            ys = [float(r[ykey]) for r in usable]
            rho = spearman(xs, ys)
            stats[f"{xkey}~{ykey}"] = {
                "n": len(xs), "rho": round(rho, 4),
                "p": round(perm_p(xs, ys), 4),
            }
    print()
    print(json.dumps(stats, indent=2))

    out.write_text(
        json.dumps({"rows": rows, "correlations": stats}, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
