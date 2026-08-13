"""Per-episode fact extraction for the loss anatomy of v15/v19a/v19b/v20/v21.

Everything here is read off the stored replay.  Two sources are used:

* ``observation.current`` for board snapshots (prizes, deck counts, bodies);
* ``observation.logs`` for the event stream.  Logs are *deltas* handed to a
  seat when it becomes ACTIVE, so concatenating one seat's ACTIVE-step logs
  (plus the trailing DONE record) reproduces the whole public event stream
  exactly once.  Taking both seats double-counts.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v20"))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"

RUNS = (
    ("v15", "data/runs/grimmsnarl/20260810_grimmsnarl_ml_v15_sub55404196"),
    ("v19a", "data/runs/grimmsnarl/20260811_grimmsnarl_ml_v19_sub55428196"),
    ("v19b", "data/runs/grimmsnarl/20260812_grimmsnarl_ml_v19_sub55445763"),
    ("v20", "data/runs/grimmsnarl/20260812_grimmsnarl_ml_v20_sub55445769"),
    ("v21", "data/runs/grimmsnarl/20260813_grimmsnarl_ml_v21_sub55456713"),
)

L_HAS_BASIC = 1
L_TURN_START = 2
L_TURN_END = 3
L_DRAW = 4
L_MOVE = 6
L_ATTACK = 15
L_HP = 16

A_DECK, A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE = 1, 2, 3, 4, 5, 6

CARDS = {
    int(c["cardId"]): c
    for c in json.loads((ROOT / "vendor" / "cg" / "cards.json").read_text("utf-8"))
}


def is_pokemon(card_id: int) -> bool:
    return CARDS.get(card_id, {}).get("cardType") == 0


def max_hp(card_id: int) -> int:
    return int(CARDS.get(card_id, {}).get("hp", 0) or 0)


def decks(steps) -> list[list[int] | None]:
    out: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                out[seat] = [int(v) for v in action]
    return out


def late_current(steps) -> dict[str, Any] | None:
    best, best_key = None, (-1, -1)
    for index, step in enumerate(steps):
        for actor in (0, 1):
            if actor >= len(step):
                continue
            cur = ((step[actor] or {}).get("observation") or {}).get("current")
            if not (isinstance(cur, dict) and cur.get("players")):
                continue
            key = (int(cur.get("turn", -1)), index)
            if key > best_key:
                best, best_key = cur, key
    return best


_REVERSE = {4: 5, 5: 4, 6: 7, 7: 6}


def _norm_type(entry: dict[str, Any]) -> int:
    etype = entry.get("type")
    return min(etype, _REVERSE.get(etype, etype))


def log_stream(steps, seat: int) -> list[dict[str, Any]]:
    """Full public event stream, each event exactly once.

    ``observation.logs`` is the delta since *that seat's* previous action, so a
    single seat's stream misses everything after its last action (the DONE
    record just repeats the previous delta).  Merge both seats by keeping a
    per-seat pointer into the global stream and appending only the suffix the
    seat brings that is new.  Hidden-information mirroring (DRAW 4 <-> DRAW
    REVERSE 5, MOVE 6 <-> MOVE REVERSE 7) is 1:1 so the offsets line up; this
    was verified to produce zero overlap mismatches over all 255 episodes.
    Where both seats carry an event, the *seat* argument's version wins so our
    own card ids stay visible.
    """
    stream: list[dict[str, Any]] = []
    pointer = {0: 0, 1: 0}
    for step in steps:
        for actor in (0, 1):
            rec = step[actor] or {}
            if rec.get("status") != "ACTIVE":
                continue
            delta = (rec.get("observation") or {}).get("logs") or []
            offset = len(stream) - pointer[actor]
            if offset < 0 or offset > len(delta):
                stream.extend(delta)
                pointer[actor] = len(stream)
                continue
            if actor == seat:
                for k in range(offset):
                    if _norm_type(stream[pointer[actor] + k]) == _norm_type(delta[k]):
                        stream[pointer[actor] + k] = delta[k]
            stream.extend(delta[offset:])
            pointer[actor] = len(stream)
    return stream


def cards_of(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return mf._cards(player, area)


def bodies(player: dict[str, Any]) -> int:
    return len(cards_of(player, "active")) + len(cards_of(player, "bench"))


def prize_left(player: dict[str, Any]) -> int | None:
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    if isinstance(prize, int):
        return prize
    return None


def walk(replay: dict[str, Any], seat: int) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    dk = decks(steps)
    if dk[seat] is None or deck_hash(dk[seat]) != OUR_DECK_HASH:
        return None

    rewards = replay.get("rewards") or [None, None]
    ours, theirs = rewards[seat], rewards[1 - seat]
    if ours is None:
        return None
    won = bool(ours > (theirs if theirs is not None else 0))

    cur_late = late_current(steps)
    went_first = None
    if cur_late is not None:
        first = int(cur_late.get("firstPlayer", -1))
        went_first = (first == seat) if first >= 0 else None
    players_late = (cur_late or {}).get("players") or [{}, {}]
    us_late = players_late[seat] if len(players_late) > 1 else {}
    them_late = players_late[1 - seat] if len(players_late) > 1 else {}

    # ---- our-seat observation walk: opening hand, contexts, actions ---------
    opening_hand: list[int] = []
    is_first_choice: int | None = None
    is_first_offered = False
    our_active_steps = 0
    our_empty_actions = 0
    min_overage = None
    contexts: Counter[int] = Counter()
    first_ready_turn = None
    max_turn = 0
    our_main_turns: set[int] = set()

    for index, step in enumerate(steps[:-1]):
        for actor in (0, 1):
            rec = step[actor] or {}
            if rec.get("status") != "ACTIVE":
                continue
            obs = rec.get("observation") or {}
            sel = obs.get("select") or {}
            cur = obs.get("current") or {}
            pls = cur.get("players") or []
            options = list(sel.get("option") or [])
            if len(pls) < 2:
                continue
            turn = int(cur.get("turn", -1))
            max_turn = max(max_turn, turn)
            if actor != seat:
                continue
            ctx = int(sel.get("context", -1))
            contexts[ctx] += 1
            our_active_steps += 1
            overage = obs.get("remainingOverageTime")
            if isinstance(overage, (int, float)):
                min_overage = overage if min_overage is None else min(min_overage, overage)
            action = (steps[index + 1][actor] or {}).get("action")
            if options and (not isinstance(action, list) or not action):
                if int(sel.get("minCount", 0)) > 0:
                    our_empty_actions += 1
            if ctx == 41 and options:
                is_first_offered = True
                if isinstance(action, list) and action:
                    is_first_choice = int(action[0])
            if ctx == 1 and not opening_hand:
                opening_hand = [int(c.get("id", -1)) for c in (pls[seat].get("hand") or [])]
            if ctx == mf.MAIN_CONTEXT:
                our_main_turns.add(turn)
            if first_ready_turn is None:
                for card in cards_of(pls[seat], "active") + cards_of(pls[seat], "bench"):
                    if (
                        int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
                        and mf._dark_energy_count(card) >= mf.SHADOW_BULLET_COST
                    ):
                        first_ready_turn = turn
                        break

    # ---- public log stream --------------------------------------------------
    logs = log_stream(steps, seat)
    mulligans = Counter()
    attacks: list[dict[str, Any]] = []
    cur_attack: dict[str, Any] | None = None
    turn = 0
    turn_owner = None
    ko_ours: list[dict[str, Any]] = []
    ko_theirs: list[dict[str, Any]] = []
    damage_state: dict[int, int] = {}   # serial -> accumulated damage
    serial_card: dict[int, int] = {}
    our_draw_total = 0
    first_shadow_turn = None
    first_attack_turn = None
    opp_first_attack_turn = None

    def close_attack() -> None:
        nonlocal cur_attack
        if cur_attack is not None:
            attacks.append(cur_attack)
            cur_attack = None

    for entry in logs:
        etype = entry.get("type")
        pi = entry.get("playerIndex")
        if etype == L_HAS_BASIC and entry.get("hasBasicPokemon") is False:
            mulligans[pi] += 1
        elif etype == L_TURN_START:
            close_attack()
            turn += 1
            turn_owner = pi
        elif etype == L_DRAW and pi == seat:
            our_draw_total += 1
        elif etype == L_ATTACK:
            close_attack()
            cur_attack = {
                "turn": turn,
                "by": pi,
                "attack_id": entry.get("attackId"),
                "card_id": entry.get("cardId"),
                "damage": 0,
                "kos": 0,
                "one_shot_kos": 0,
                "targets": [],
            }
            if pi == seat:
                if first_attack_turn is None:
                    first_attack_turn = turn
                if int(entry.get("attackId", -1)) == mf.SHADOW_BULLET_ID and first_shadow_turn is None:
                    first_shadow_turn = turn
            elif opp_first_attack_turn is None:
                opp_first_attack_turn = turn
        elif etype == L_HP:
            serial = entry.get("serial")
            card_id = int(entry.get("cardId", -1))
            value = int(entry.get("value", 0))
            serial_card[serial] = card_id
            if value < 0:
                prev = damage_state.get(serial, 0)
                damage_state[serial] = prev + (-value)
                if cur_attack is not None and pi != cur_attack["by"]:
                    cur_attack["damage"] += -value
                    cur_attack["targets"].append((card_id, -value, prev))
            else:
                damage_state[serial] = max(0, damage_state.get(serial, 0) - value)
        elif etype == L_MOVE:
            frm, to = entry.get("fromArea"), entry.get("toArea")
            card_id = int(entry.get("cardId", -1))
            serial = entry.get("serial")
            if frm in (A_ACTIVE, A_BENCH) and to == A_DISCARD and is_pokemon(card_id):
                hp = max_hp(card_id)
                dealt = damage_state.get(serial, 0)
                if hp and dealt >= hp:  # a real KO, not a discard effect
                    rec = {"turn": turn, "card_id": card_id, "owner": pi,
                           "one_shot": False}
                    if cur_attack is not None:
                        cur_attack["kos"] += 1
                        # was this body undamaged before the attack landed?
                        before = 0
                        for cid, dmg, prev in cur_attack["targets"]:
                            if cid == card_id:
                                before = prev
                                break
                        if before == 0:
                            cur_attack["one_shot_kos"] += 1
                            rec["one_shot"] = True
                        rec["by_attack"] = True
                    else:
                        rec["by_attack"] = False
                    (ko_ours if pi == seat else ko_theirs).append(rec)
                damage_state.pop(serial, None)
    close_attack()

    our_attacks = [a for a in attacks if a["by"] == seat]
    opp_attacks = [a for a in attacks if a["by"] != seat]

    hand_ids = Counter(opening_hand)
    return {
        "won": won,
        "seat": seat,
        "went_first": went_first,
        "is_first_offered": is_first_offered,
        "is_first_choice": is_first_choice,
        "opponent_family": family(dk[1 - seat]),
        "opponent_hash": deck_hash(dk[1 - seat]) if dk[1 - seat] else "",
        "turns": max_turn,
        "our_turns": len(our_main_turns),
        "log_turns": turn,
        "our_prize_left": prize_left(us_late),
        "opp_prize_left": prize_left(them_late),
        "our_deck_left": us_late.get("deckCount"),
        "opp_deck_left": them_late.get("deckCount"),
        "our_bodies_left": bodies(us_late) if us_late else None,
        "opp_bodies_left": bodies(them_late) if them_late else None,
        "our_hand_left": us_late.get("handCount"),
        "opening_hand": "|".join(str(c) for c in opening_hand),
        "opening_impidimp": hand_ids.get(mf.IMPIDIMP_ID, 0),
        "opening_rare_candy": hand_ids.get(mf.RARE_CANDY_ID, 0),
        "opening_grim_ex": hand_ids.get(mf.GRIMMSNARL_EX_ID, 0),
        "opening_morgrem": hand_ids.get(mf.MORGREM_ID, 0),
        "opening_dark_energy": hand_ids.get(mf.DARK_ENERGY_ID, 0),
        "opening_basics": sum(
            n for cid, n in hand_ids.items() if cid in mf.BASIC_POKEMON_IDS
        ),
        "our_mulligans": mulligans.get(seat, 0),
        "opp_mulligans": mulligans.get(1 - seat, 0),
        "our_attacks": len(our_attacks),
        "opp_attacks": len(opp_attacks),
        "our_damage": sum(a["damage"] for a in our_attacks),
        "opp_damage": sum(a["damage"] for a in opp_attacks),
        "our_zero_damage_attacks": sum(1 for a in our_attacks if a["damage"] == 0),
        "our_kos": len(ko_theirs),
        "opp_kos": len(ko_ours),
        "opp_one_shot_kos": sum(a["one_shot_kos"] for a in opp_attacks),
        "our_one_shot_kos": sum(a["one_shot_kos"] for a in our_attacks),
        "our_grim_ko": sum(
            1 for k in ko_ours if k["card_id"] == mf.GRIMMSNARL_EX_ID
        ),
        "our_grim_one_shot": sum(
            1 for k in ko_ours
            if k["card_id"] == mf.GRIMMSNARL_EX_ID and k["one_shot"]
        ),
        "our_small_ko": sum(
            1 for k in ko_ours if k["card_id"] != mf.GRIMMSNARL_EX_ID
        ),
        "opp_grim_ko": sum(
            1 for k in ko_theirs if k["card_id"] == mf.GRIMMSNARL_EX_ID
        ),
        "opp_max_attack_damage": max(
            [a["damage"] for a in opp_attacks], default=0
        ),
        "our_max_attack_damage": max(
            [a["damage"] for a in our_attacks], default=0
        ),
        "our_ko_ledger": json.dumps(
            [{"t": k["turn"], "c": k["card_id"], "os": k["one_shot"]}
             for k in ko_ours]
        ),
        "first_ready_turn": first_ready_turn,
        "first_shadow_turn": first_shadow_turn,
        "first_attack_turn": first_attack_turn,
        "opp_first_attack_turn": opp_first_attack_turn,
        "our_draws": our_draw_total,
        "our_active_steps": our_active_steps,
        "our_empty_actions": our_empty_actions,
        "min_overage": min_overage,
        "contexts": json.dumps(dict(sorted(contexts.items()))),
        "attack_ledger": json.dumps(
            [
                {
                    "t": a["turn"], "by": ("us" if a["by"] == seat else "opp"),
                    "atk": a["attack_id"], "card": a["card_id"],
                    "dmg": a["damage"], "ko": a["kos"], "os": a["one_shot_kos"],
                }
                for a in attacks
            ]
        ),
    }


def load(label: str, run_dir: Path) -> list[dict[str, Any]]:
    manifest = {
        row["episode_id"]: row
        for row in csv.DictReader((run_dir / "manifest.csv").open(encoding="utf-8-sig"))
    }
    episodes = {
        row["episode_id"]: row
        for row in csv.DictReader((run_dir / "episodes.csv").open(encoding="utf-8-sig"))
    }
    rows = []
    for episode_id, entry in manifest.items():
        seat_text = entry.get("detected_submission_agent_index", "")
        if seat_text not in {"0", "1"}:
            continue
        seat = int(seat_text)
        path = run_dir / "episodes" / episode_id / "replay" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        row = walk(replay, seat)
        if row is None:
            continue
        meta = episodes.get(episode_id, {})
        opp_rating = meta.get(f"agent_{1 - seat}_initial_score", "")
        our_rating = meta.get(f"agent_{seat}_initial_score", "")
        row.update({
            "version": label,
            "episode_id": int(episode_id),
            "create_time": meta.get("create_time", ""),
            "opponent_submission": meta.get(f"agent_{1 - seat}_submission_id", ""),
            "opponent_rating": float(opp_rating) if opp_rating else None,
            "our_rating": float(our_rating) if our_rating else None,
            "statuses": "|".join(replay.get("statuses") or []),
        })
        rows.append(row)
    rows.sort(key=lambda r: r["episode_id"])
    return rows


def main() -> int:
    out: list[dict[str, Any]] = []
    for label, path in RUNS:
        rows = load(label, ROOT / path)
        print(f"{label}: {len(rows)} games")
        out.extend(rows)
    fields = sorted({k for r in out for k in r})
    dest = Path(__file__).resolve().parent / "episodes.csv"
    with dest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(f"total {len(out)} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
