"""Analyze saved ladder episodes under data/runs/<run>/episodes/.

For each episode: our seat, win/loss, end reason (deck-out / error / prizes),
opponent archetype (from their 60-card deck), and final counters.

Usage: python scripts/analyze_runs.py [run_dir ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from cg.api import all_card_data  # noqa: E402

CARD = {c.cardId: c for c in all_card_data()}


def archetype(deck: list[int]) -> str:
    """Label a deck by its most distinctive Pokémon (highest evolution/ex first)."""
    pokes = Counter(cid for cid in deck
                    if CARD.get(cid) and CARD[cid].cardType == 0)
    if not pokes:
        return "unknown"

    def key(item):
        cid, n = item
        c = CARD[cid]
        return (c.stage2, c.megaEx or c.ex, c.stage1, n, c.hp)

    best = max(pokes.items(), key=key)[0]
    return CARD[best].name


def analyze_episode(path: Path, my_sub: int, meta_row: dict):
    ep = json.loads(path.read_text(encoding="utf-8"))
    steps = ep["steps"]
    rewards = ep.get("rewards") or [None, None]
    my_seat = 0 if str(meta_row.get("agent_0_submission_id")) == str(my_sub) else 1
    opp_seat = 1 - my_seat

    decks = [None, None]
    if len(steps) > 1:
        for pi in (0, 1):
            act = steps[1][pi].get("action")
            if isinstance(act, list) and len(act) == 60:
                decks[pi] = act

    # walk backwards to the last observation that has a current state
    last_cur = None
    for st in reversed(steps):
        for pi in (0, 1):
            obs = st[pi].get("observation") or {}
            if obs.get("current"):
                last_cur = obs["current"]
                break
        if last_cur:
            break

    raw_statuses = ep.get("statuses") or [steps[-1][pi].get("status") for pi in (0, 1)]
    statuses = [s if isinstance(s, str) else (s or {}).get("status")
                for s in raw_statuses]

    res = {
        "episode": path.stem.replace("episode_", ""),
        "win": rewards[my_seat] == 1 if rewards[my_seat] is not None else None,
        "opp_archetype": archetype(decks[opp_seat]) if decks[opp_seat] else "unknown",
        "statuses": statuses,
        "turns": last_cur.get("turn") if last_cur else None,
        "reason": "",
    }
    if last_cur:
        me = last_cur["players"][my_seat]
        op = last_cur["players"][opp_seat]
        res["my_deck_left"] = me.get("deckCount")
        res["opp_deck_left"] = op.get("deckCount")
        res["my_prizes_left"] = len(me.get("prize") or [])
        res["opp_prizes_left"] = len(op.get("prize") or [])
        if not res["win"]:
            if me.get("deckCount") == 0:
                res["reason"] = "DECK-OUT(自分)"
            elif res["opp_prizes_left"] == 0:
                res["reason"] = "サイド取り切られ"
            elif statuses and any(s in ("ERROR", "TIMEOUT", "INVALID")
                                  for s in statuses[my_seat:my_seat + 1]):
                res["reason"] = f"自分の{statuses[my_seat]}"
            else:
                res["reason"] = "その他(たね切れ等)"
    return res


def main():
    run_dirs = [Path(a) for a in sys.argv[1:]] or \
               sorted((ROOT / "data" / "runs").glob("*_sub*"))
    for run in run_dirs:
        if not run.is_dir():
            continue
        meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
        my_sub = meta["submission_id"]
        rows = {}
        csv_path = run / "episodes.csv"
        if csv_path.exists():
            import csv as csvmod
            with open(csv_path, encoding="utf-8-sig") as f:
                for r in csvmod.DictReader(f):
                    rows[r["episode_id"]] = r

        print(f"\n===== {run.name} (deck={meta.get('deck_name')}) =====")
        results = []
        for ep_dir in sorted((run / "episodes").iterdir()):
            replay = next((ep_dir / "replay").glob("episode_*.json"), None)
            if replay is None:
                continue
            r = analyze_episode(replay, my_sub, rows.get(ep_dir.name, {}))
            results.append(r)
            mark = "WIN " if r["win"] else "LOSE"
            print(f"  {r['episode']}  {mark} vs {r['opp_archetype']:<28} "
                  f"turns={r['turns']} 残デッキ(自/相)={r.get('my_deck_left')}/"
                  f"{r.get('opp_deck_left')} 残サイド(自/相)={r.get('my_prizes_left')}/"
                  f"{r.get('opp_prizes_left')} {r['reason']}")

        wins = sum(1 for r in results if r["win"])
        print(f"  -- {wins}/{len(results)} wins")
        losses = [r for r in results if not r["win"]]
        if losses:
            print("  敗因内訳:", dict(Counter(r["reason"] for r in losses)))
            print("  負けた相手:", dict(Counter(r["opp_archetype"] for r in losses)))


if __name__ == "__main__":
    main()
