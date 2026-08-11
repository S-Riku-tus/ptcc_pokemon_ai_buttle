"""What happens to a v15 Shadow Bullet after it lands.

v15 closed the attack-access gap: first Shadow Bullet moved from turn 3.60 to
2.84 over 110 rated games and 82.7% of games have one by our own turn 3.  The
ladder did not move with it, and the reason the v15 autopsy proposes is that
the next bottleneck is *conversion*: 73% of the losses had started attacking.

This measures conversion directly, from our own 110 ladder replays, on the four
questions the v16 plan rests on:

1. **Prize per Shadow Bullet**, split by result and by matchup.  If losses
   attack as often as wins, the deficit is in what the attack buys.
2. **Orphan Bench-30**: Shadow Bullet's 30 to a Benched body is only worth
   anything if that body is later knocked out.  v15's own metadata lists this
   as unmeasured.
3. **No-progress runs**: our own turns where we attacked and neither prize
   count, nor their Active's HP, nor their board changed.  This is the state
   the wall circuit breaker would have to detect, and it says how often it
   would fire.
4. **Attacker construction**: the turn our first Grimmsnarl ex reaches the
   board at all, split by opening body - the state ``attack_access`` cannot
   help with, because there is no attacker to give access to.

    python scripts/analyze_grimmsnarl_v16_prize_conversion.py \
        --run data/runs/grimmsnarl/20260810_grimmsnarl_ml_v15_sub55404196 \
        --submission 55404196 \
        --run data/runs/grimmsnarl/20260811_grimmsnarl_ml_v15_b_sub55409394 \
        --submission 55409394
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}

MAIN_CONTEXT = 0
CTX_DAMAGE = 15
OPTION_ATTACK = 13
SHADOW_BULLET_ID = 937
GRIMMSNARL_EX_ID = 648
IMPIDIMP_ID = 646
SNORUNT_ID = 860
MUNKIDORI_ID = 112
FROSLASS_ID = 104
OPENING_NAMES = {
    IMPIDIMP_ID: "impidimp",
    SNORUNT_ID: "snorunt",
    MUNKIDORI_ID: "munkidori",
    FROSLASS_ID: "froslass",
}


def deck_label(deck: list[int] | None, top: int = 2) -> str:
    if not deck:
        return "unknown"
    pokemon = Counter(
        card_id for card_id in deck
        if CARDS.get(card_id, {}).get("cardType") == 0
        and (CARDS[card_id].get("stage1") or CARDS[card_id].get("stage2")
             or CARDS[card_id].get("ex") or CARDS[card_id].get("megaEx"))
    )
    names = [
        CARDS[cid].get("name", str(cid))
        for cid, _ in sorted(
            pokemon.items(),
            key=lambda kv: (-kv[1], -int(CARDS[kv[0]].get("hp", 0))),
        )[:top]
    ]
    return " + ".join(names) if names else "unknown"


def matchup_of(label: str) -> str:
    low = label.lower()
    if "grimmsnarl" in low:
        return "mirror"
    if "alakazam" in low or "kadabra" in low:
        return "alakazam"
    if "crustle" in low or "sylveon" in low or "ogerpon" in low:
        return "wall"
    if "dipplin" in low or "hydrapple" in low:
        return "festival"
    return "other"


def own_turn(current: dict[str, Any], seat: int, first_player: int) -> int:
    """Our own turn ordinal. ``current.turn`` is shared between both seats."""
    turn = int(current.get("turn", 0) or 0)
    if first_player < 0:
        return turn
    return (turn + 1) // 2 if first_player == seat else turn // 2


def cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    value = player.get(area)
    if not isinstance(value, list):
        return []
    return [c for c in value if isinstance(c, dict)]


def board_signature(opponent: dict[str, Any]) -> tuple:
    active = (cards(opponent, "active") or [{}])[0]
    return (
        len(opponent.get("prize") or []),
        int(active.get("id", -1)),
        float(active.get("hp", 0) or 0),
        tuple(sorted(
            (int(c.get("id", -1)), float(c.get("hp", 0) or 0))
            for c in cards(opponent, "bench")
        )),
    )


def analyse_episode(
    replay: dict[str, Any], seat: int
) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    if len(steps) < 3:
        return None

    decks: list[list[int] | None] = [None, None]
    for side in (0, 1):
        action = (steps[1][side] or {}).get("action")
        if isinstance(action, list) and len(action) == 60:
            decks[side] = [int(v) for v in action]

    first_player = -1
    for step in reversed(steps):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if not isinstance(current, dict):
            continue
        if int(current.get("firstPlayer", -1)) >= 0:
            first_player = int(current.get("firstPlayer", -1))
            break

    # ----- walk our own decisions ------------------------------------------
    turn_state: dict[int, dict[str, Any]] = {}
    shadow_turns: list[int] = []
    other_attack_turns: list[int] = []
    first_grim_turn: int | None = None
    first_grim_ready_turn: int | None = None
    opening_body: int | None = None
    last_deck_count: int | None = None
    final_our_prizes: int | None = None
    bench30: list[dict[str, Any]] = []
    serial_last_seen: dict[int, int] = {}
    serial_hp: dict[int, float] = {}
    evolved_into: dict[int, int] = {}
    pending_bench30: dict[str, Any] | None = None

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        entry = step[seat] or {}
        observation = entry.get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(current, dict) or not current.get("players"):
            continue
        players = current["players"]
        if len(players) < 2:
            continue
        me, opponent = players[seat], players[1 - seat]
        turn = own_turn(current, seat, first_player)
        last_deck_count = int(me.get("deckCount", 0) or 0)
        final_our_prizes = len(me.get("prize") or [])

        in_play = cards(me, "active") + cards(me, "bench")
        if opening_body is None and in_play:
            active = (cards(me, "active") or [{}])[0]
            if int(active.get("id", -1)) > 0:
                opening_body = int(active.get("id", -1))
        for card in in_play:
            if int(card.get("id", -1)) == GRIMMSNARL_EX_ID:
                if first_grim_turn is None:
                    first_grim_turn = turn
                if (
                    first_grim_ready_turn is None
                    and len(
                        card.get("energyCards")
                        or card.get("energies") or []
                    ) >= 2
                ):
                    first_grim_ready_turn = turn

        # remember every opposing body we have seen, to detect a later KO.
        # Evolving replaces the serial while carrying the damage, so the
        # pre-evolution chain has to be followed or every evolved target reads
        # as a knock-out.
        for card in cards(opponent, "active") + cards(opponent, "bench"):
            serial = card.get("serial")
            if serial is None:
                continue
            serial_last_seen[int(serial)] = index
            serial_hp[int(serial)] = float(card.get("hp", 0) or 0)
            for pre in (card.get("preEvolution") or []):
                pre_serial = (pre or {}).get("serial")
                if pre_serial is not None:
                    evolved_into[int(pre_serial)] = int(serial)

        if turn not in turn_state:
            turn_state[turn] = {
                "turn": turn,
                "our_prizes_start": len(me.get("prize") or []),
                "signature": board_signature(opponent),
                "attacked": False,
                "shadow": False,
            }

        if not isinstance(select, dict):
            continue
        action = (steps[index + 1][seat] or {}).get("action")
        if not (isinstance(action, list) and len(action) == 1
                and isinstance(action[0], int)):
            continue
        chosen = int(action[0])
        options = select.get("option") or []
        if not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        context = int(select.get("context", -1))

        if context == MAIN_CONTEXT and option.get("type") == OPTION_ATTACK:
            turn_state[turn]["attacked"] = True
            if int(option.get("attackId", -1)) == SHADOW_BULLET_ID:
                turn_state[turn]["shadow"] = True
                shadow_turns.append(turn)
                pending_bench30 = {"turn": turn, "step": index}
            else:
                other_attack_turns.append(turn)
        elif context == CTX_DAMAGE and pending_bench30 is not None:
            # The Bench-30 target select carries its own post-180 board, and
            # the option indexes straight into the opposing Bench.
            bench = cards(opponent, "bench")
            slot = option.get("index")
            offered = [
                bench[int(o.get("index"))]
                for o in options
                if isinstance(o.get("index"), int)
                and 0 <= int(o["index"]) < len(bench)
            ]
            if isinstance(slot, int) and 0 <= slot < len(bench):
                target = bench[slot]
                hp = float(target.get("hp", 0) or 0)
                bench30.append({
                    "turn": pending_bench30["turn"],
                    "step": index,
                    "serial": int(target.get("serial", -1)),
                    "id": int(target.get("id", -1)),
                    "hp_before": hp,
                    "lethal_now": 0 < hp <= 30.0,
                    "lethal_next": 30.0 < hp <= 60.0,
                    "offered_lethal": any(
                        0 < float(c.get("hp", 0) or 0) <= 30.0 for c in offered
                    ),
                    "offered_count": len(offered),
                })
            pending_bench30 = None

    # ----- fill in the per-turn deltas -------------------------------------
    ordered = [turn_state[t] for t in sorted(turn_state)]
    for position, state in enumerate(ordered):
        nxt = position + 1
        following = ordered[nxt] if nxt < len(ordered) else None
        end_prizes = (
            following["our_prizes_start"] if following
            else (
                final_our_prizes if final_our_prizes is not None
                else state["our_prizes_start"]
            )
        )
        state["prizes_taken"] = max(0, state["our_prizes_start"] - end_prizes)
        state["next_signature"] = following["signature"] if following else None
        state["stalled"] = bool(
            state["attacked"]
            and state["prizes_taken"] == 0
            and following is not None
            and following["signature"] == state["signature"]
        )

    stall_run = 0
    longest_stall = 0
    for state in ordered:
        stall_run = stall_run + 1 if state["stalled"] else 0
        longest_stall = max(longest_stall, stall_run)

    # ----- did the Bench-30 ever become a prize? ---------------------------
    final_step = max(serial_last_seen.values(), default=-1)
    for entry in bench30:
        serial = entry["serial"]
        chain = []
        while serial is not None and serial not in chain:
            chain.append(serial)
            serial = evolved_into.get(serial)
        seen_until = max(
            (serial_last_seen.get(s, -1) for s in chain), default=-1
        )
        # The body (or whatever it evolved into) stopped appearing before the
        # last board we saw: it was knocked out. Still there at the end means
        # the 30 never became anything.
        entry["died_later"] = bool(0 <= seen_until < final_step)

    rewards = replay.get("rewards") or [None, None]
    won = None
    if rewards[seat] is not None:
        other = rewards[1 - seat]
        won = bool(rewards[seat] > (other if other is not None else 0))

    label = deck_label(decks[1 - seat])
    shadow_count = len(shadow_turns)
    return {
        "won": won,
        "opponent_deck": label,
        "matchup": matchup_of(label),
        "went_first": (first_player == seat) if first_player >= 0 else None,
        "own_turns": len(ordered),
        "shadow_bullets": shadow_count,
        "other_attacks": len(other_attack_turns),
        "first_shadow_turn": min(shadow_turns) if shadow_turns else None,
        "first_grim_turn": first_grim_turn,
        "first_grim_ready_turn": first_grim_ready_turn,
        "opening_body": OPENING_NAMES.get(opening_body or -1, "other"),
        "prizes_taken": sum(s["prizes_taken"] for s in ordered),
        "prizes_on_shadow_turns": sum(
            s["prizes_taken"] for s in ordered if s["shadow"]
        ),
        "shadow_turns_without_prize": sum(
            1 for s in ordered if s["shadow"] and s["prizes_taken"] == 0
        ),
        "stalled_turns": sum(1 for s in ordered if s["stalled"]),
        "longest_stall_run": longest_stall,
        "bench30_shots": len(bench30),
        "bench30_lethal_now": sum(1 for e in bench30 if e["lethal_now"]),
        "bench30_lethal_next": sum(1 for e in bench30 if e["lethal_next"]),
        "bench30_missed_lethal": sum(
            1 for e in bench30
            if e["offered_lethal"] and not e["lethal_now"]
        ),
        "bench30_choices": sum(1 for e in bench30 if e["offered_count"] > 1),
        "bench30_orphaned": sum(
            1 for e in bench30 if not e["lethal_now"] and not e["died_later"]
        ),
        "final_deck_count": last_deck_count,
        "final_our_prizes": final_our_prizes,
        "deckout_risk": bool(
            last_deck_count is not None and last_deck_count <= 3
        ),
    }


def load_run(run_dir: Path, submission: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(
        (run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if raw["state"] != "COMPLETED":
            continue
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC":
            continue
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if a0 == a1:
            continue
        seat = 0 if a0 == submission else 1
        episode_id = int(raw["episode_id"])
        path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        row = analyse_episode(
            json.loads(path.read_text(encoding="utf-8")), seat
        )
        if row is None:
            continue

        def score(key: str) -> float | None:
            text = (raw.get(key) or "").strip()
            try:
                return float(text) if text else None
            except ValueError:
                return None

        before = score(f"agent_{seat}_initial_score")
        after = score(f"agent_{seat}_updated_score")
        row.update({
            "episode_id": episode_id,
            "run": run_dir.name,
            "our_score_before": before,
            "our_score_after": after,
            "rating_delta": (
                round(after - before, 2)
                if before is not None and after is not None else None
            ),
            "opponent_score": score(f"agent_{1 - seat}_initial_score"),
        })
        rows.append(row)
    return rows


def mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 3) if clean else None


def summarise(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if r["won"] is False]

    def side(subset: list[dict[str, Any]]) -> dict[str, Any]:
        shots = sum(r["bench30_shots"] for r in subset)
        return {
            "games": len(subset),
            "first_shadow_turn": mean(
                [r["first_shadow_turn"] for r in subset]
            ),
            "first_grim_turn": mean([r["first_grim_turn"] for r in subset]),
            "shadow_bullets_per_game": mean(
                [r["shadow_bullets"] for r in subset]
            ),
            "prizes_per_shadow_bullet": (
                round(
                    sum(r["prizes_on_shadow_turns"] for r in subset)
                    / max(1, sum(r["shadow_bullets"] for r in subset)), 3
                )
            ),
            "shadow_turns_without_prize_share": (
                round(
                    sum(r["shadow_turns_without_prize"] for r in subset)
                    / max(1, sum(r["shadow_bullets"] for r in subset)), 3
                )
            ),
            "stalled_turns_per_game": mean(
                [r["stalled_turns"] for r in subset]
            ),
            "games_with_stall_run_2plus": sum(
                1 for r in subset if r["longest_stall_run"] >= 2
            ),
            "bench30_orphan_share": (
                round(
                    sum(r["bench30_orphaned"] for r in subset) / max(1, shots),
                    3,
                )
            ),
            "bench30_shots": shots,
            "bench30_lethal_now_share": (
                round(
                    sum(r["bench30_lethal_now"] for r in subset)
                    / max(1, shots), 3
                )
            ),
            "bench30_lethal_next_share": (
                round(
                    sum(r["bench30_lethal_next"] for r in subset)
                    / max(1, shots), 3
                )
            ),
            "bench30_missed_lethal": sum(
                r["bench30_missed_lethal"] for r in subset
            ),
            "bench30_real_choices": sum(
                r["bench30_choices"] for r in subset
            ),
            "mean_final_deck_count": mean(
                [r["final_deck_count"] for r in subset]
            ),
        }

    return {
        "name": name,
        "games": len(rows),
        "record": f"{len(wins)}-{len(losses)}",
        "win_rate": round(len(wins) / len(rows), 4) if rows else None,
        "mean_rating_delta": mean([r["rating_delta"] for r in rows]),
        "mean_opponent_rating": mean([r["opponent_score"] for r in rows]),
        "wins": side(wins),
        "losses": side(losses),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--submission", action="append", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run_dir, submission in zip(args.run, args.submission):
        rows.extend(load_run(run_dir, submission))

    by_matchup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_matchup[row["matchup"]].append(row)
    by_opening: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_opening[row["opening_body"]].append(row)

    out = {
        "overall": summarise(rows, "all"),
        "by_matchup": {
            key: summarise(value, key)
            for key, value in sorted(
                by_matchup.items(), key=lambda kv: -len(kv[1])
            )
        },
        "by_opening_body": {
            key: summarise(value, key)
            for key, value in sorted(
                by_opening.items(), key=lambda kv: -len(kv[1])
            )
        },
        "worst_stalls": sorted(
            (
                {
                    "episode_id": r["episode_id"],
                    "matchup": r["matchup"],
                    "opponent_deck": r["opponent_deck"],
                    "won": r["won"],
                    "shadow_bullets": r["shadow_bullets"],
                    "stalled_turns": r["stalled_turns"],
                    "longest_stall_run": r["longest_stall_run"],
                    "prizes_taken": r["prizes_taken"],
                    "final_deck_count": r["final_deck_count"],
                    "rating_delta": r["rating_delta"],
                }
                for r in rows
            ),
            key=lambda r: (-r["longest_stall_run"], -r["shadow_bullets"]),
        )[:12],
        "slowest_construction": sorted(
            (
                {
                    "episode_id": r["episode_id"],
                    "matchup": r["matchup"],
                    "opening_body": r["opening_body"],
                    "won": r["won"],
                    "first_grim_turn": r["first_grim_turn"],
                    "first_shadow_turn": r["first_shadow_turn"],
                    "rating_delta": r["rating_delta"],
                }
                for r in rows
            ),
            key=lambda r: -(r["first_grim_turn"] or 99),
        )[:12],
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
        (args.report.with_suffix(".rows.json")).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
