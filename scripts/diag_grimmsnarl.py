"""Diagnostic harness: why does Marnie's Grimmsnarl lose to alakazam_ml_v11?

Plays N games (alternating seats) and captures, per game, engine/tempo/prize
signals for the Grimmsnarl seat, then aggregates split by win vs loss so the
failure modes are visible instead of guessed.

Usage:
  python scripts/diag_grimmsnarl.py <grimm_agent> <opp_agent> --games 300
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_loader import load_dir_agent  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

MUNKIDORI, FROSLASS, GRIMMSNARL_EX, MORGREM, IMPIDIMP, SNORUNT = 112, 104, 648, 647, 646, 860
DARK = 7


def resolve(spec: str):
    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        cand = base / spec
        if cand.is_dir():
            return load_dir_agent(cand)
        if base.is_dir():
            for group in sorted(base.iterdir()):
                if (group / spec).is_dir():
                    return load_dir_agent(group / spec)
    raise FileNotFoundError(spec)


def board(player):
    return list(player.get("active") or []) + list(player.get("bench") or [])


def count_id(player, cid):
    return sum(1 for p in board(player) if p and p.get("id") == cid)


def dark_on(pokemon):
    return sum(1 for e in (pokemon.get("energies") or []) if e == DARK)


def grimm_ready(player):
    for p in board(player):
        if p and p.get("id") == GRIMMSNARL_EX and dark_on(p) >= 2:
            return True
    return False


def snap(diag):
    keys = ["shadow_bullets", "adrena_brains", "attackable_ends",
            "punk_up_searches", "retreats_to_attacker", "attack_reservation_active"]
    return {k: int(diag.get(k, 0) or 0) for k in keys}


def play(gagent, gdiag, oagent, gseat, max_steps=8000):
    decks = [None, None]
    decks[gseat] = gagent({"select": None})
    decks[1 - gseat] = oagent({"select": None})
    agents = [None, None]
    agents[gseat] = gagent
    agents[1 - gseat] = oagent

    before = snap(gdiag)
    rec = {
        "gseat": gseat, "winner": None, "turns": 0,
        "g_prizes_taken": 0, "o_prizes_taken": 0,
        "max_munki": 0, "froslass_online": False, "grimm_ready_turn": None,
        "first_shadow_turn": None, "max_board": 0, "max_dark_board": 0,
        "opp_maxhand": 0, "opp_hand_when_g_active_ko": [],
    }
    prev_shadow = before["shadow_bullets"]
    prev_g_active_alive = True

    obs, sd = battle_start(decks[0], decks[1])
    if obs is None:
        battle_finish()
        return None
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                rec["winner"] = ("g" if cur["result"] == gseat else
                                 "o" if cur["result"] == (1 - gseat) else "draw")
                rec["turns"] = cur["turn"]
                gp = cur["players"][gseat]
                op = cur["players"][1 - gseat]
                # In TCG you take from your OWN prize pile on a KO.
                rec["g_prizes_taken"] = 6 - len(gp.get("prize") or [])
                rec["o_prizes_taken"] = 6 - len(op.get("prize") or [])
                break
            seat = cur["yourIndex"]
            gp = cur["players"][gseat]
            op = cur["players"][1 - gseat]
            # engine snapshots from the grimmsnarl seat's perspective
            rec["max_munki"] = max(rec["max_munki"], count_id(gp, MUNKIDORI))
            rec["max_board"] = max(rec["max_board"], len(board(gp)))
            rec["max_dark_board"] = max(
                rec["max_dark_board"], sum(dark_on(p) for p in board(gp) if p))
            if count_id(gp, FROSLASS) > 0:
                rec["froslass_online"] = True
            if rec["grimm_ready_turn"] is None and grimm_ready(gp):
                rec["grimm_ready_turn"] = cur["turn"]
            rec["opp_maxhand"] = max(rec["opp_maxhand"], op.get("handCount", 0) or 0)
            # detect our active grimmsnarl getting KO'd: was a grimmsnarl active, now gone
            g_active = gp.get("active") or []
            g_active_is_grimm = bool(g_active and g_active[0] and g_active[0].get("id") == GRIMMSNARL_EX)
            if prev_g_active_alive and not g_active_is_grimm and cur["turn"] > 2:
                rec["opp_hand_when_g_active_ko"].append(op.get("handCount", 0) or 0)
            prev_g_active_alive = g_active_is_grimm

            try:
                action = agents[seat](obs)
            except Exception:
                rec["winner"] = "o" if seat == gseat else "g"
                break
            try:
                obs = battle_select(list(action))
            except Exception:
                rec["winner"] = "o" if seat == gseat else "g"
                break
            if seat == gseat:
                now_shadow = int(gdiag.get("shadow_bullets", 0) or 0)
                if now_shadow > prev_shadow and rec["first_shadow_turn"] is None:
                    rec["first_shadow_turn"] = cur["turn"]
                prev_shadow = now_shadow
        else:
            rec["winner"] = "draw"
    finally:
        battle_finish()

    after = snap(gdiag)
    for k in before:
        rec[k] = after[k] - before[k]
    return rec


def agg(records, key, filt=None):
    vals = [r[key] for r in records if (filt is None or filt(r)) and r[key] is not None]
    if not vals:
        return "n/a"
    if isinstance(vals[0], bool):
        return f"{sum(vals)/len(vals):.1%}"
    return f"{statistics.mean(vals):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grimm")
    ap.add_argument("opp")
    ap.add_argument("--games", type=int, default=300)
    args = ap.parse_args()

    gagent, gdiag, _ = resolve(args.grimm)
    oagent, _, _ = resolve(args.opp)

    records = []
    for g in range(args.games):
        rec = play(gagent, gdiag, oagent, gseat=(g % 2))
        if rec:
            records.append(rec)

    wins = [r for r in records if r["winner"] == "g"]
    losses = [r for r in records if r["winner"] == "o"]
    print(f"== {args.grimm} vs {args.opp}: {len(records)} games ==")
    print(f"grimm wins: {len(wins)}  losses: {len(losses)}  "
          f"win rate: {len(wins)/max(1,len(wins)+len(losses)):.1%}")
    print(f"grimm going first win rate: "
          f"{sum(1 for r in records if r['gseat']==0 and r['winner']=='g')}/"
          f"{sum(1 for r in records if r['gseat']==0)}   "
          f"going second: "
          f"{sum(1 for r in records if r['gseat']==1 and r['winner']=='g')}/"
          f"{sum(1 for r in records if r['gseat']==1)}")

    metrics = ["turns", "g_prizes_taken", "o_prizes_taken", "shadow_bullets",
               "adrena_brains", "attackable_ends", "punk_up_searches",
               "retreats_to_attacker", "max_munki", "froslass_online",
               "grimm_ready_turn", "first_shadow_turn", "max_board",
               "max_dark_board", "opp_maxhand"]
    print(f"\n{'metric':24s} {'WIN':>10s} {'LOSS':>10s}")
    for m in metrics:
        print(f"{m:24s} {agg(wins,m):>10s} {agg(losses,m):>10s}")

    # prize distribution in losses: how far did grimm get before losing?
    from collections import Counter
    print("\ng_prizes_taken distribution in LOSSES:",
          dict(sorted(Counter(r["g_prizes_taken"] for r in losses).items())))
    print("g_prizes_taken distribution in WINS:  ",
          dict(sorted(Counter(r["g_prizes_taken"] for r in wins).items())))
    # opponent hand size at the moment our active grimmsnarl dies
    ko_hands = [h for r in records for h in r["opp_hand_when_g_active_ko"]]
    if ko_hands:
        print(f"\nopp hand size when our active Grimmsnarl KO'd: "
              f"mean={statistics.mean(ko_hands):.1f} max={max(ko_hands)} n={len(ko_hands)}")


def r_first(r):
    return 0


if __name__ == "__main__":
    main()
