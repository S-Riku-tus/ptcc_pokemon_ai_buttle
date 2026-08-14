"""Offer-side or take-side? The test that decides whether a metric is a lever.

The 194-game pool puts two numbers far apart in wins and losses: Adrena-Brain
activations per own turn (1.46 vs 0.96) and Froslass evolutions in the mirror
(0-Froslass 0.719, 1+ 0.292).  Neither is actionable until it is split:

* if we are *offered* Adrena-Brain and decline, the policy can be changed;
  if offers equal takes, the count is a scoreboard of board health and
  optimising it repeats the Powerful Hand mistake;
* if Froslass is evolved *early*, before the game is decided, it can be
  causing the mirror losses; if it is evolved late, it is what a losing board
  reaches for and the correlation is backwards.

Both are answered from the stored options, which record every legal action the
runtime was shown.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments" / "grimmsnarl_ml_v23" / "ladder_v22_v23_games.csv"
RUNS = ROOT / "data" / "runs" / "grimmsnarl"
OUT = ROOT / "experiments" / "grimmsnarl_ml_v23" / "offer_vs_take.json"


def own_turn(turn: int, went_first: bool | None) -> int:
    if went_first is None:
        return (turn + 1) // 2
    return (turn + 1) // 2 if went_first else turn // 2


def walk(replay: dict, seat: int, went_first: bool | None) -> dict[str, Any]:
    steps = replay.get("steps") or []
    adrena_offered = adrena_taken = 0
    adrena_declined_turns: list[int] = []
    fros_offered = fros_taken = 0
    fros_turns: list[int] = []
    # prizes remaining on both sides at the moment Froslass was evolved, so a
    # late "we are behind" evolution is distinguishable from an early one
    fros_context: list[dict[str, int | None]] = []
    grim_offered = grim_taken = 0

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        obs = record.get("observation") or {}
        select = obs.get("select") or {}
        options = list(select.get("option") or [])
        current = obs.get("current") or {}
        players = current.get("players") or []
        if len(players) < 2 or not options:
            continue
        turn = int(current.get("turn", -1))
        ot = own_turn(turn, went_first)
        action = (steps[index + 1][seat] or {}).get("action")
        picked = {int(v) for v in action
                  if isinstance(v, int) and 0 <= int(v) < len(options)} \
            if isinstance(action, list) else set()

        def prize(side: int) -> int | None:
            p = players[side].get("prize")
            return len(p) if isinstance(p, list) else (p if isinstance(p, int) else None)

        saw_adrena = saw_fros = saw_grim = False
        took_adrena = took_fros = took_grim = False
        for position, option in enumerate(options):
            try:
                kind = mf.action_type(current, option, select)
            except Exception:  # noqa: BLE001
                continue
            card = mf.candidate_card(current, option, select) or {}
            cid = int(card.get("id", -1))
            if kind == "ability" and cid == mf.MUNKIDORI_ID:
                saw_adrena = True
                took_adrena = took_adrena or position in picked
            elif kind not in {"attack", "ability", "end", "retreat"}:
                if cid == mf.FROSLASS_ID:
                    saw_fros = True
                    took_fros = took_fros or position in picked
                elif cid == mf.GRIMMSNARL_EX_ID and kind == "evolve":
                    saw_grim = True
                    took_grim = took_grim or position in picked
        if saw_adrena:
            adrena_offered += 1
            adrena_taken += int(took_adrena)
            if not took_adrena:
                adrena_declined_turns.append(ot)
        if saw_fros:
            fros_offered += 1
            fros_taken += int(took_fros)
            if took_fros:
                fros_turns.append(ot)
                fros_context.append({"own_turn": ot, "our_prize": prize(seat),
                                     "opp_prize": prize(1 - seat)})
        if saw_grim:
            grim_offered += 1
            grim_taken += int(took_grim)

    return {
        "adrena_offered": adrena_offered, "adrena_taken": adrena_taken,
        "adrena_declined_turns": adrena_declined_turns,
        "fros_offered": fros_offered, "fros_taken": fros_taken,
        "fros_turns": fros_turns, "fros_context": fros_context,
        "grim_offered": grim_offered, "grim_taken": grim_taken,
    }


def blk(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"games": 0}
    w = sum(1 for r in rows if r["won"])
    lo, hi = wilson(w, n)
    return {"games": n, "wins": w, "win_rate": round(w / n, 3),
            "wilson95": [round(lo, 3), round(hi, 3)]}


def welch(a: list, b: list) -> dict:
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if len(a) < 2 or len(b) < 2:
        return {}
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = (va / len(a) + vb / len(b)) ** 0.5
    t = (ma - mb) / se if se else float("nan")
    return {"win_mean": round(ma, 3), "loss_mean": round(mb, 3),
            "t": round(t, 2) if not math.isnan(t) else None}


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    from math import comb
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def p(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
    obs = p(a)
    total = 0.0
    for x in range(max(0, col1 - (n - row1)), min(row1, col1) + 1):
        v = p(x)
        if v <= obs + 1e-12:
            total += v
    return min(1.0, total)


def main() -> int:
    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (run_dir, int(row["detected_submission_agent_index"]))

    rows = []
    for meta in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not meta["version"].startswith("v22"):
            continue
        entry = index.get(meta["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = run_dir / "episodes" / meta["episode_id"] / "replay" / f"episode_{meta['episode_id']}.json"
        if not path.exists():
            continue
        went_first = {"True": True, "False": False}.get(meta["went_first"])
        data = walk(json.loads(path.read_text(encoding="utf-8")), seat, went_first)
        data.update({"episode_id": meta["episode_id"], "won": meta["won"] == "True",
                     "family": meta["opponent_family"], "went_first": went_first})
        rows.append(data)

    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    mirror = [r for r in rows if r["family"] == "Grimmsnarl (mirror)"]

    tot_off = sum(r["adrena_offered"] for r in rows)
    tot_take = sum(r["adrena_taken"] for r in rows)
    report: dict[str, Any] = {
        "games": len(rows),
        "adrena": {
            "offers": tot_off, "takes": tot_take,
            "uptake": round(tot_take / max(1, tot_off), 4),
            "declines": tot_off - tot_take,
            "uptake_win": round(sum(r["adrena_taken"] for r in wins)
                                / max(1, sum(r["adrena_offered"] for r in wins)), 4),
            "uptake_loss": round(sum(r["adrena_taken"] for r in losses)
                                 / max(1, sum(r["adrena_offered"] for r in losses)), 4),
            "offers_per_game": welch([r["adrena_offered"] for r in wins],
                                     [r["adrena_offered"] for r in losses]),
            "decline_turn_hist": dict(sorted(Counter(
                t for r in rows for t in r["adrena_declined_turns"]).items())),
        },
        "grimmsnarl_evolve": {
            "offers": sum(r["grim_offered"] for r in rows),
            "takes": sum(r["grim_taken"] for r in rows),
            "uptake": round(sum(r["grim_taken"] for r in rows)
                            / max(1, sum(r["grim_offered"] for r in rows)), 4),
        },
        "froslass": {
            "offers": sum(r["fros_offered"] for r in rows),
            "takes": sum(r["fros_taken"] for r in rows),
            "uptake_all": round(sum(r["fros_taken"] for r in rows)
                                / max(1, sum(r["fros_offered"] for r in rows)), 4),
            "uptake_mirror": round(sum(r["fros_taken"] for r in mirror)
                                   / max(1, sum(r["fros_offered"] for r in mirror)), 4),
            "mirror_offers_per_game": round(
                sum(r["fros_offered"] for r in mirror) / max(1, len(mirror)), 2),
            "evolve_own_turn_hist_mirror": dict(sorted(Counter(
                t for r in mirror for t in r["fros_turns"]).items())),
            "first_evolve_turn": welch(
                [min(r["fros_turns"]) for r in mirror if r["won"] and r["fros_turns"]],
                [min(r["fros_turns"]) for r in mirror if not r["won"] and r["fros_turns"]]),
            "prize_state_at_evolve_mirror": {
                "behind": sum(1 for r in mirror for c in r["fros_context"]
                              if c["our_prize"] is not None and c["opp_prize"] is not None
                              and c["our_prize"] > c["opp_prize"]),
                "even": sum(1 for r in mirror for c in r["fros_context"]
                            if c["our_prize"] is not None and c["opp_prize"] is not None
                            and c["our_prize"] == c["opp_prize"]),
                "ahead": sum(1 for r in mirror for c in r["fros_context"]
                             if c["our_prize"] is not None and c["opp_prize"] is not None
                             and c["our_prize"] < c["opp_prize"]),
            },
            "early_only_mirror": blk([r for r in mirror if r["fros_turns"]
                                      and min(r["fros_turns"]) <= 2]),
            "late_only_mirror": blk([r for r in mirror if r["fros_turns"]
                                     and min(r["fros_turns"]) >= 3]),
            "none_mirror": blk([r for r in mirror if not r["fros_turns"]]),
            "offered_but_declined_mirror": blk(
                [r for r in mirror if r["fros_offered"] > 0 and r["fros_taken"] == 0]),
            "never_offered_mirror": blk([r for r in mirror if r["fros_offered"] == 0]),
        },
    }
    m0 = [r for r in mirror if not r["fros_turns"]]
    m1 = [r for r in mirror if r["fros_turns"]]
    report["froslass"]["mirror_fisher_p"] = round(fisher(
        sum(1 for r in m0 if r["won"]), sum(1 for r in m0 if not r["won"]),
        sum(1 for r in m1 if r["won"]), sum(1 for r in m1 if not r["won"])), 5)

    report["per_game"] = [{k: r[k] for k in
                           ("episode_id", "won", "family", "adrena_offered",
                            "adrena_taken", "fros_offered", "fros_taken", "fros_turns")}
                          for r in rows]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_game"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
