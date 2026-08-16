"""Separate two explanations for the falling controlled win rate.

(a) rating deflation - a displayed 800 today is a stronger agent than a
    displayed 800 three days ago, because the pool keeps absorbing fresh
    600-start submissions.
(b) meta drift - the field genuinely outgrew our policy.

Under (a) the day coefficient disappears once the opponent is measured by the
score its team carries on ONE common date (the 08-16 board). Under (b) it
survives.
"""
import csv
import json
import math
import datetime as dt

import numpy as np

lb = json.load(open("experiments/grimmsnarl_endgame_20260816/leaderboard_full_20260816.json",
                    encoding="utf-8"))["publicLeaderboard"]
by_sub = {str(r["submissionId"]): float(r["displayScore"]) for r in lb
          if r.get("submissionId") is not None}

rows = list(csv.DictReader(open("experiments/grimmsnarl_endgame_20260816/version_games.csv",
                                encoding="utf-8-sig")))
base = dt.date(2026, 8, 13)


def fit(X, y, iters=300):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        b += np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b))
    W = np.clip(p * (1 - p), 1e-9, None)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def report(title, X, y, names):
    b, se = fit(X, y)
    beta = abs(b[1]) / 400.0
    print(f"--- {title} (n={len(y)}, wins={int(y.sum())}) ---")
    for nm, c, s in zip(names, b, se):
        z = c / s
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        elo = c / beta if beta > 1e-9 else float("nan")
        extra = f"{elo:9.1f}" if nm not in ("intercept", "opp/400") else " " * 9
        print(f"  {nm:14s} {c:8.4f} se={s:6.4f} z={z:6.2f} p={p:7.4f} "
              f"Elo={extra}")
    print()


names = ["intercept", "opp/400", "went_first", "day_index"]

for tag, keyfn in (("paired rating", lambda r: r["opponent_rating"]),
                   ("settled rating",
                    lambda r: by_sub.get(r["opponent_submission"]))):
    for subset, label in ((lambda r: True, "all versions"),
                          (lambda r: r["version"].startswith("v22"),
                           "v22 code only")):
        X, y = [], []
        for r in rows:
            if not subset(r):
                continue
            v = keyfn(r)
            if v in (None, ""):
                continue
            d = dt.date.fromisoformat(r["create_time"][:10])
            X.append([1.0, float(v) / 400.0,
                      1.0 if r["went_first"] == "first" else 0.0,
                      (d - base).days])
            y.append(int(r["won"]))
        if len(y) < 20:
            continue
        report(f"{tag} / {label}", np.array(X), np.array(y, float), names)
