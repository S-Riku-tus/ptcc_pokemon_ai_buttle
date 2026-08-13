"""Addressable-alternative probes: legal lines the replay shows we declined.

Every hit is emitted with episode id, shared turn, our own-turn ordinal, the
legal option index we did not take, and what we took instead, so each one can
be opened in the replay by hand.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v20"))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

from extract import RUNS, OUR_DECK_HASH, decks, late_current  # noqa: E402

DRAW_ENGINE_IDS = {
    mf.POFFIN_ID, mf.POKE_PAD_ID, mf.PETREL_ID, mf.LILLIE_ID,
    mf.POKEGEAR_ID, mf.DAWN_ID,
}


def own_turn(turn: int, went_first: bool | None) -> int:
    if went_first is None:
        return (turn + 1) // 2
    return (turn + 1) // 2 if went_first else turn // 2


def shield_ids(player: dict[str, Any]) -> set[int]:
    out = {int(c.get("id", -1)) for c in mf._cards(player, "active")}
    out |= {int(c.get("id", -1)) for c in mf._cards(player, "bench")}
    return out


def prizes_from_shadow(current: dict[str, Any], seat: int) -> int:
    """Prizes one Shadow Bullet takes right now (Active 180 + Bench 30)."""
    players = current.get("players") or []
    if len(players) < 2:
        return 0
    them = players[1 - seat]
    stadium = mf._stadium_id(current)
    shields = shield_ids(them)
    total = 0
    for card in mf._cards(them, "active"):
        total += mf.active_prizes(card, stadium)
    for card in mf._cards(them, "bench"):
        total += mf.snipe_prizes(card, stadium, shields)
    return total


def opp_prizes_left(current: dict[str, Any], seat: int) -> int:
    players = current.get("players") or []
    if len(players) < 2:
        return 6
    prize = players[1 - seat].get("prize")
    if isinstance(prize, list):
        return len(prize)
    if isinstance(prize, int):
        return prize
    return 6


def scan(replay: dict[str, Any], seat: int, episode_id: int, label: str):
    steps = replay.get("steps") or []
    dk = decks(steps)
    if dk[seat] is None or deck_hash(dk[seat]) != OUR_DECK_HASH:
        return []
    rewards = replay.get("rewards") or [None, None]
    ours, theirs = rewards[seat], rewards[1 - seat]
    if ours is None:
        return []
    won = bool(ours > (theirs if theirs is not None else 0))
    cur_late = late_current(steps)
    went_first = None
    if cur_late is not None:
        first = int(cur_late.get("firstPlayer", -1))
        went_first = (first == seat) if first >= 0 else None

    rows = []
    for index, step in enumerate(steps[:-1]):
        rec = step[seat] or {}
        if rec.get("status") != "ACTIVE":
            continue
        obs = rec.get("observation") or {}
        sel = obs.get("select") or {}
        cur = obs.get("current") or {}
        options = list(sel.get("option") or [])
        players = cur.get("players") or []
        if len(players) < 2 or not options:
            continue
        action = (steps[index + 1][seat] or {}).get("action")
        picked = [
            int(v) for v in action
            if isinstance(v, int) and 0 <= int(v) < len(options)
        ] if isinstance(action, list) else []
        ctx = int(sel.get("context", -1))
        turn = int(cur.get("turn", -1))
        us, them = players[seat], players[1 - seat]

        kinds = []
        for opt in options:
            try:
                kinds.append(mf.action_type(cur, opt, sel))
            except Exception:  # noqa: BLE001
                kinds.append("?")

        rows.append({
            "version": label,
            "episode_id": episode_id,
            "won": won,
            "step": index,
            "turn": turn,
            "own_turn": own_turn(turn, went_first),
            "went_first": went_first,
            "context": ctx,
            "max_count": int(sel.get("maxCount", 0)),
            "min_count": int(sel.get("minCount", 0)),
            "n_options": len(options),
            "kinds": json.dumps(kinds),
            "picked": json.dumps(picked),
            "picked_kinds": json.dumps([kinds[p] for p in picked]),
            "our_deck": us.get("deckCount"),
            "our_bench": len(mf._cards(us, "bench")),
            "our_bodies": len(mf._cards(us, "active")) + len(mf._cards(us, "bench")),
            "opp_prizes_left": opp_prizes_left(cur, seat),
            "shadow_prizes_now": prizes_from_shadow(cur, seat),
            "attack_option_idx": json.dumps(
                [i for i, k in enumerate(kinds) if k == "attack"]
            ),
            "attack_ids": json.dumps(
                [mf._int(options[i].get("attackId"))
                 for i, k in enumerate(kinds) if k == "attack"]
            ),
            "bench_play_idx": json.dumps([
                i for i, k in enumerate(kinds)
                if k not in {"attack", "end", "retreat", "ability", "evolve"}
                and int((mf.candidate_card(cur, options[i], sel) or {}).get("id", -1))
                in mf.BASIC_POKEMON_IDS
            ]),
            "grim_evolve_idx": json.dumps([
                i for i, k in enumerate(kinds)
                if k == "evolve"
                and int((mf.candidate_card(cur, options[i], sel) or {}).get("id", -1))
                == mf.GRIMMSNARL_EX_ID
            ]),
            "candy_idx": json.dumps([
                i for i, k in enumerate(kinds)
                if int((mf.candidate_card(cur, options[i], sel) or {}).get("id", -1))
                == mf.RARE_CANDY_ID
            ]),
            "draw_engine_idx": json.dumps([
                i for i, k in enumerate(kinds)
                if int((mf.candidate_card(cur, options[i], sel) or {}).get("id", -1))
                in DRAW_ENGINE_IDS
            ]),
            "end_idx": json.dumps([i for i, k in enumerate(kinds) if k == "end"]),
            "ready_grim": any(
                int(c.get("id", -1)) == mf.GRIMMSNARL_EX_ID
                and mf._dark_energy_count(c) >= mf.SHADOW_BULLET_COST
                for c in mf._cards(us, "active") + mf._cards(us, "bench")
            ),
            "active_ready_grim": any(
                int(c.get("id", -1)) == mf.GRIMMSNARL_EX_ID
                and mf._dark_energy_count(c) >= mf.SHADOW_BULLET_COST
                for c in mf._cards(us, "active")
            ),
            "opponent_family": family(dk[1 - seat]),
        })
    return rows


def main() -> int:
    out: list[dict[str, Any]] = []
    for label, path in RUNS:
        run_dir = ROOT / path
        manifest = {
            row["episode_id"]: row
            for row in csv.DictReader(
                (run_dir / "manifest.csv").open(encoding="utf-8-sig")
            )
        }
        for episode_id, entry in manifest.items():
            seat_text = entry.get("detected_submission_agent_index", "")
            if seat_text not in {"0", "1"}:
                continue
            p = run_dir / "episodes" / episode_id / "replay" / f"episode_{episode_id}.json"
            if not p.exists():
                continue
            replay = json.loads(p.read_text(encoding="utf-8"))
            out.extend(scan(replay, int(seat_text), int(episode_id), label))
        print(f"{label}: {len(out)} decisions cumulative")
    fields = sorted({k for r in out for k in r})
    dest = HERE / "decisions.csv"
    with dest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(f"{len(out)} decisions -> {dest}")
    print(Counter(r["context"] for r in out).most_common(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
