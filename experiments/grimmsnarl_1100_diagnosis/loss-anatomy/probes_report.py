"""Query decisions.csv for declined legal lines, restricted to lost games.

A turn can hold many MAIN decisions and it only ends on an attack or on END,
so "we did not attack" is a property of the *turn*, never of a single
decision - playing an energy first and attacking second is not a declined
attack.  Every probe below therefore aggregates our MAIN decisions by
(episode, shared turn) and asks whether the turn closed without the line.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

MAIN = 0
SHADOW_BULLET_ID = 937


def load(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    for row in rows:
        row["won"] = row["won"] == "True"
        for key in ("turn", "own_turn", "context", "n_options", "opp_prizes_left",
                    "shadow_prizes_now", "our_bench", "our_bodies", "max_count",
                    "min_count", "step"):
            row[key] = int(row[key]) if row[key] not in ("", "None") else None
        row["our_deck"] = (
            int(row["our_deck"]) if row["our_deck"] not in ("", "None") else None
        )
        for key in ("kinds", "picked", "picked_kinds", "attack_option_idx",
                    "attack_ids", "bench_play_idx", "grim_evolve_idx",
                    "candy_idx", "draw_engine_idx", "end_idx"):
            row[key] = json.loads(row[key])
        row["ready_grim"] = row["ready_grim"] == "True"
        row["active_ready_grim"] = row["active_ready_grim"] == "True"
    return rows


def turns_of(main_rows):
    buckets = defaultdict(list)
    for row in main_rows:
        buckets[(row["episode_id"], row["turn"])].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda r: r["step"])
    return buckets


def attacked(items) -> bool:
    return any(
        p in r["attack_option_idx"] for r in items for p in r["picked"]
    )


def summarise(hits, lost, total_losses=95):
    """``hits`` is a list of (episode, turn, evidence-dict)."""
    by_ep = defaultdict(list)
    for episode, turn, evidence in hits:
        by_ep[episode].append((turn, evidence))
    lost_eps = {e: v for e, v in by_ep.items() if e in lost}
    examples = []
    for episode, items in sorted(lost_eps.items())[:10]:
        turn, evidence = items[0]
        examples.append({"episode": episode, "turn": turn,
                         "turns_in_episode": len(items), **evidence})
    return {
        "hit_turns": len(hits),
        "episodes_any": len(by_ep),
        "episodes_lost": len(lost_eps),
        "share_of_losses": round(len(lost_eps) / total_losses, 4),
        "wilson95_of_losses": wilson(len(lost_eps), total_losses),
        "lost_episode_ids": sorted(lost_eps),
        "examples": examples,
    }


def main() -> int:
    rows = load(HERE / "decisions.csv")
    episodes = {r["episode_id"]: r["won"] for r in rows}
    lost = {e for e, w in episodes.items() if not w}
    main_rows = [r for r in rows if r["context"] == MAIN]
    buckets = turns_of(main_rows)
    print(f"decisions={len(rows)} MAIN={len(main_rows)} "
          f"our_turns={len(buckets)} episodes={len(episodes)} lost={len(lost)}")

    out = {}

    # ---- P1: a turn ended with a GAME-WINNING Shadow Bullet legal ---------
    hits = []
    for (episode, turn), items in buckets.items():
        if attacked(items):
            continue
        for r in items:
            legal = [
                i for i, a in zip(r["attack_option_idx"], r["attack_ids"])
                if a == SHADOW_BULLET_ID
            ]
            if legal and r["shadow_prizes_now"] >= r["opp_prizes_left"]:
                hits.append((episode, turn, {
                    "own_turn": r["own_turn"], "step": r["step"],
                    "legal_attack_option_index": legal,
                    "we_picked": r["picked"], "picked_kinds": r["picked_kinds"],
                    "opp_prizes_left": r["opp_prizes_left"],
                    "shadow_prizes_now": r["shadow_prizes_now"],
                    "family": r["opponent_family"], "version": r["version"],
                }))
                break
    out["P1_turn_ended_with_game_winning_shadow_bullet_legal"] = summarise(
        hits, lost
    )

    # ---- P2: a turn ended with a PRIZE-TAKING Shadow Bullet legal ----------
    hits = []
    for (episode, turn), items in buckets.items():
        if attacked(items):
            continue
        for r in items:
            legal = [
                i for i, a in zip(r["attack_option_idx"], r["attack_ids"])
                if a == SHADOW_BULLET_ID
            ]
            if legal and r["shadow_prizes_now"] >= 1:
                hits.append((episode, turn, {
                    "own_turn": r["own_turn"], "step": r["step"],
                    "legal_attack_option_index": legal,
                    "we_picked": r["picked"], "picked_kinds": r["picked_kinds"],
                    "opp_prizes_left": r["opp_prizes_left"],
                    "shadow_prizes_now": r["shadow_prizes_now"],
                    "family": r["opponent_family"], "version": r["version"],
                }))
                break
    out["P2_turn_ended_with_prize_taking_shadow_bullet_legal"] = summarise(
        hits, lost
    )

    # ---- P3: a turn ended with ANY attack legal and a ready Grimmsnarl -----
    hits = []
    for (episode, turn), items in buckets.items():
        if attacked(items):
            continue
        for r in items:
            if r["attack_option_idx"] and r["active_ready_grim"]:
                hits.append((episode, turn, {
                    "own_turn": r["own_turn"], "step": r["step"],
                    "legal_attack_option_index": r["attack_option_idx"],
                    "we_picked": r["picked"], "picked_kinds": r["picked_kinds"],
                    "shadow_prizes_now": r["shadow_prizes_now"],
                    "family": r["opponent_family"], "version": r["version"],
                }))
                break
    out["P3_turn_ended_with_ready_grimmsnarl_and_legal_attack"] = summarise(
        hits, lost
    )

    # ---- P4: a turn ended at <=2 bodies with a Basic playable -------------
    hits = []
    for (episode, turn), items in buckets.items():
        played = any(
            p in r["bench_play_idx"] for r in items for p in r["picked"]
        )
        if played:
            continue
        for r in items:
            if r["bench_play_idx"] and (r["our_bodies"] or 9) <= 2:
                hits.append((episode, turn, {
                    "own_turn": r["own_turn"], "step": r["step"],
                    "legal_bench_option_index": r["bench_play_idx"],
                    "we_picked": r["picked"], "picked_kinds": r["picked_kinds"],
                    "our_bodies": r["our_bodies"],
                    "family": r["opponent_family"], "version": r["version"],
                }))
                break
    out["P4_turn_ended_at_<=2_bodies_with_a_basic_playable"] = summarise(
        hits, lost
    )

    # ---- P5: optional draw/search taken with a thin deck ------------------
    for limit in (10, 6, 4):
        hits = []
        for (episode, turn), items in buckets.items():
            for r in items:
                if (r["our_deck"] is not None and r["our_deck"] <= limit
                        and r["draw_engine_idx"] and r["end_idx"]
                        and any(p in r["draw_engine_idx"] for p in r["picked"])):
                    hits.append((episode, turn, {
                        "own_turn": r["own_turn"], "step": r["step"],
                        "deck_left": r["our_deck"],
                        "declined_END_option_index": r["end_idx"],
                        "we_picked": r["picked"],
                        "picked_kinds": r["picked_kinds"],
                        "family": r["opponent_family"],
                        "version": r["version"],
                    }))
                    break
        out[f"P5_draw_engine_played_with_deck<={limit}"] = summarise(hits, lost)

    # ---- P6: a turn ended with a Grimmsnarl evolution / Rare Candy legal ---
    hits = []
    for (episode, turn), items in buckets.items():
        took = any(
            p in (r["grim_evolve_idx"] + r["candy_idx"])
            for r in items for p in r["picked"]
        )
        if took:
            continue
        for r in items:
            if r["grim_evolve_idx"] or r["candy_idx"]:
                hits.append((episode, turn, {
                    "own_turn": r["own_turn"], "step": r["step"],
                    "legal_option_index": r["grim_evolve_idx"] + r["candy_idx"],
                    "we_picked": r["picked"], "picked_kinds": r["picked_kinds"],
                    "ready_grim_already": r["ready_grim"],
                    "family": r["opponent_family"], "version": r["version"],
                }))
                break
    out["P6_turn_ended_with_grim_evolution_or_candy_legal"] = summarise(
        hits, lost
    )

    (HERE / "probes.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, payload in out.items():
        print(f"\n=== {name}: turns={payload['hit_turns']} "
              f"eps_any={payload['episodes_any']} "
              f"eps_lost={payload['episodes_lost']} "
              f"({payload['share_of_losses']:.3f} of 95, "
              f"CI {payload['wilson95_of_losses']})")
        for ex in payload["examples"][:4]:
            print("   ", json.dumps(ex, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
