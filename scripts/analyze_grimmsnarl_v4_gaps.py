"""Is each v3 ladder gap a decision error or a resource shortfall?

The v3 ladder analysis lists four per-game deficits against the top-five
pilots:
energy 3.17 vs 3.77, Adrena-Brain 4.77 vs 6.07, Boss 0.46 vs 0.65, Grimmsnarl
evolves 1.19 vs 1.32. A per-game count cannot tell those two causes apart, and
the fix is opposite in each case:

* offered and declined -> a preference the ranker or a planner rule owns;
* never offered -> a resource/tempo problem upstream, and no amount of
  target-selection work touches it.

So every MAIN decision is classified by what was *on offer*, and the take rate
is reported per source. The same pass measures the Shadow Bullet wall question
at the strictness the fix needs: v2.1 left the wall as a feature because "the
teachers still attack 17% of the time", but that figure is for `dead_swing`
(walled Active, no Bench-30 *kill*). The proposed hard guard is the strictly
narrower `worthless` (walled Active, Bench-30 cannot damage *anything*),
which buys literally zero. Those two rates have to be separated before a veto
ships, because a veto on a shape the teachers use is how a shell starts costing
more than it saves.

Everything is aggregated **per own turn**, not per MAIN decision. MAIN is
re-asked
after every intermediate action, and attacking ends the turn, so a per-decision
denominator is dominated by how many actions a turn happens to contain: a turn
with five MAIN decisions that ends in an attack reads as a 20% attack rate. The
first pass of this script measured exactly that and made the teachers look like
they attack into a wall 18.9% of the time. Per turn is the only denominator
that
answers "did the player do this when they could".

Splits: seat (went first / second), mirror vs not, and per teacher team.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

AGENT_DIR = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v3"

MAIN = 0
OPTION_ATTACK = 13
OPTION_END = 14
OPTION_EVOLVE = 9
OPTION_ABILITY = 10
OPTION_ENERGY = 8
OPTION_PLAY_CARD = 7
OPTION_RETREAT = 12

DARK_ENERGY_ID = 7
IMPIDIMP_ID = 646
MORGREM_ID = 647
GRIMMSNARL_EX_ID = 648
FROSLASS_ID = 104
SNORUNT_ID = 860
MUNKIDORI_ID = 112
RARE_CANDY_ID = 1079
UNFAIR_STAMP_ID = 1080
BOSS_ID = 1182
SHADOW_BULLET_ID = 937
SHADOW_BULLET_BENCH = 30.0
SHADOW_BULLET_COST = 2


def _load_tables() -> Any:
    """v3's own feature module, so wall/weakness logic cannot drift from it."""
    sys.path.insert(0, str(AGENT_DIR))
    import ml_features  # noqa: PLC0415

    return ml_features


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [c for c in (player.get(area) or []) if isinstance(c, dict)]


def _offer_kinds(
    mf: Any,
    current: dict[str, Any],
    select: dict[str, Any],
    options: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Option slots grouped by the action class they belong to."""
    kinds: dict[str, list[int]] = defaultdict(list)
    for slot, option in enumerate(options):
        option_type = int(option.get("type", -1))
        card = mf.candidate_card(current, option, select) or {}
        card_id = int(card.get("id", -1))
        if option_type == OPTION_ATTACK:
            kinds["attack"].append(slot)
        elif option_type == OPTION_END:
            kinds["end"].append(slot)
        elif option_type == OPTION_RETREAT:
            kinds["retreat"].append(slot)
        elif option_type == OPTION_EVOLVE:
            kinds["evolve"].append(slot)
            if card_id == GRIMMSNARL_EX_ID:
                kinds["evolve_grimmsnarl"].append(slot)
            elif card_id == FROSLASS_ID:
                kinds["evolve_froslass"].append(slot)
            elif card_id == MORGREM_ID:
                kinds["evolve_morgrem"].append(slot)
        elif option_type == OPTION_ABILITY:
            kinds["ability"].append(slot)
            if card_id == MUNKIDORI_ID:
                kinds["ability_munkidori"].append(slot)
        elif option_type in (OPTION_ENERGY, OPTION_PLAY_CARD):
            if card_id == DARK_ENERGY_ID:
                kinds["energy"].append(slot)
            elif card_id == BOSS_ID:
                kinds["boss"].append(slot)
            elif card_id == UNFAIR_STAMP_ID:
                kinds["stamp"].append(slot)
            elif card_id == RARE_CANDY_ID:
                kinds["candy"].append(slot)
            elif card_id in (IMPIDIMP_ID, SNORUNT_ID, MUNKIDORI_ID):
                kinds["bench"].append(slot)
    return kinds


def _swing_class(
    mf: Any,
    current: dict[str, Any],
    opponent: dict[str, Any],
) -> dict[str, Any]:
    """What one Shadow Bullet does to this board, right now."""
    stadium_id = mf._stadium_id(current)
    shield_ids = {int(c.get("id", -1)) for c in mf._in_play(opponent)}
    opp_active = (_cards(opponent, "active") or [{}])[0]
    opp_bench = _cards(opponent, "bench")
    effective = mf.shadow_damage_to(opp_active, stadium_id)
    active_hp = float(opp_active.get("hp", 0))
    walled = int(opp_active.get("id", -1)) >= 0 and effective <= 0.0
    hittable = [
        c for c in opp_bench
        if mf.bench_snipe_lands(c, stadium_id, shield_ids)
    ]
    kills = [
        c for c in hittable
        if 0 < float(c.get("hp", 0)) <= SHADOW_BULLET_BENCH
    ]
    return {
        "walled": walled,
        # No damage to the Active and no prize off the Bench: 30 chip at most.
        "dead_swing": walled and not kills,
        # No damage anywhere on their board. Strictly nothing.
        "worthless": walled and not hittable,
        "kills_active": (not walled) and 0 < active_hp <= effective,
        "prizes": (
            (mf.prize_value(int(opp_active.get("id", -1)))
             if (not walled) and 0 < active_hp <= effective else 0)
            + max((mf.prize_value(int(c.get("id", -1))) for c in kills),
                  default=0)
        ),
    }


def _turn_tally(
    mf: Any,
    decisions: list[dict[str, Any]],
) -> Counter:
    """One own turn, reduced to "could we, and did we".

    ``decisions`` is every MAIN decision of one own turn in order, each holding
    the observation and the action actually taken.
    """
    out: Counter = Counter()
    out["turns"] = 1
    out["main_decisions"] = len(decisions)

    offered: set[str] = set()
    taken: Counter = Counter()
    attack_swings: list[dict[str, Any]] = []
    taken_swing: dict[str, Any] | None = None
    best_swing_prizes = 0
    energy_targets: list[int] = []
    enabling_energy = False
    enabling_taken = False
    unlock_available = False
    first = decisions[0]

    for record in decisions:
        current = record["current"]
        select = record["select"]
        options = record["options"]
        chosen = record["chosen"]
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", 0))
        if len(players) < 2:
            continue
        me, opponent = players[your], players[1 - your]
        kinds = _offer_kinds(mf, current, select, options)
        for kind, slots in kinds.items():
            offered.add(kind)
            if chosen in slots:
                taken[kind] += 1
        if kinds["attack"]:
            swing = _swing_class(mf, current, opponent)
            attack_swings.append(swing)
            best_swing_prizes = max(best_swing_prizes, int(swing["prizes"]))
            if chosen in kinds["attack"]:
                taken_swing = swing
            # The Kangaskhan/Crustle shape: the Active cannot be damaged, but
            # Boss is in hand and some benched body can be. Gusting converts a
            # 0-damage swing into a 180 - which is a route, not a veto.
            if swing["walled"] and kinds["boss"]:
                stadium_id = mf._stadium_id(current)
                if any(
                    mf.shadow_damage_to(c, stadium_id) > 0
                    for c in _cards(opponent, "bench")
                ):
                    unlock_available = True
        for slot in kinds["energy"]:
            target = mf.candidate_target(current, options[slot]) or {}
            target_id = int(target.get("id", -1))
            dark = mf._dark_energy_count(target)
            # Enabling: turns Adrena-Brain on for the rest of the game, or
            # completes Shadow Bullet's cost. Anything else is a stockpile.
            enables = (
                (target_id == MUNKIDORI_ID and dark == 0)
                or (target_id == GRIMMSNARL_EX_ID
                    and dark == SHADOW_BULLET_COST - 1)
            )
            if enables:
                enabling_energy = True
                if slot == chosen:
                    enabling_taken = True
            if slot == chosen:
                energy_targets.append(target_id)

    for kind in offered:
        out[f"turn_offer_{kind}"] += 1
    for kind, count in taken.items():
        out[f"turn_take_{kind}"] += 1
        out[f"turn_count_{kind}"] += count

    # ----- the wall question, per turn -----------------------------------
    if attack_swings:
        out["turn_attack_offered"] += 1
        out["turn_attacked"] += int(taken_swing is not None)
        # Classified on the swing that was actually taken; if the turn ended
        # without attacking, on the last state where one was legal, because
        # that is the swing that was declined.
        swing = taken_swing or attack_swings[-1]
        for key in ("walled", "dead_swing", "worthless"):
            if swing[key]:
                out[f"turn_{key}_offered"] += 1
                out[f"turn_{key}_attacked"] += int(taken_swing is not None)
        if swing["kills_active"]:
            out["turn_lethal_offered"] += 1
            out["turn_lethal_attacked"] += int(taken_swing is not None)
        # Did the turn ever have a swing worth prizes that it declined?
        if best_swing_prizes > 0:
            out["turn_prize_swing_offered"] += 1
            out["turn_prize_swing_attacked"] += int(taken_swing is not None)
        if unlock_available:
            out["turn_wall_unlock_offered"] += 1
            out["turn_wall_unlock_bossed"] += int(taken["boss"] > 0)
            # Boss played *and* the swing still taken: the wall converted into
            # damage in the same turn, which is what the elite pilots do.
            out["turn_wall_unlock_converted"] += int(
                taken["boss"] > 0 and taken_swing is not None
            )

    # ----- energy routing, per turn --------------------------------------
    # One dark energy per turn, and Punk Up attaches only to "your Marnie's
    # Pokemon" - so a hand attachment is the *only* way a Munkidori ever gets
    # {D} on it, and the only way Adrena-Brain ever turns on.
    if "energy" in offered:
        out["turn_energy_offered"] += 1
        out["turn_energy_taken"] += int(taken["energy"] > 0)
        if enabling_energy:
            # The narrow shape a dominance rule could own: this attachment
            # converts into a repeatable Ability or an attack that is otherwise
            # unavailable, and ending the turn destroys the chance.
            out["turn_enabling_energy_offered"] += 1
            out["turn_enabling_energy_taken"] += int(taken["energy"] > 0)
            out["turn_enabling_energy_taken_enabling"] += int(
                enabling_taken
            )
    for card_id in energy_targets:
        if card_id == GRIMMSNARL_EX_ID:
            out["turn_energy_to_grimmsnarl"] += 1
        elif card_id == MUNKIDORI_ID:
            out["turn_energy_to_munkidori"] += 1
        elif card_id in (IMPIDIMP_ID, MORGREM_ID):
            out["turn_energy_to_prevo"] += 1
        else:
            out["turn_energy_to_other"] += 1

    # ----- board state at the top of the turn ----------------------------
    current = first["current"]
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    if len(players) >= 2:
        me = players[your]
        in_play = mf._in_play(me)
        active = (_cards(me, "active") or [{}])[0]
        munk = [c for c in in_play if int(c.get("id", -1)) == MUNKIDORI_ID]
        fuelled = [c for c in munk if mf._dark_energy_count(c) > 0]
        grimm = [
            c for c in in_play
            if int(c.get("id", -1)) == GRIMMSNARL_EX_ID
        ]
        ready = [
            c for c in grimm
            if mf._dark_energy_count(c) >= SHADOW_BULLET_COST
        ]
        out["turn_munkidori_in_play"] += int(bool(munk))
        out["turn_munkidori_fuelled"] += int(bool(fuelled))
        out["turn_munkidori_all_dry"] += int(bool(munk) and not fuelled)
        out["turn_munkidori_count"] += len(munk)
        out["turn_munkidori_fuelled_count"] += len(fuelled)
        out["turn_grimmsnarl_count"] += len(grimm)
        out["turn_grimmsnarl_ready_count"] += len(ready)
        out["turn_grimmsnarl_ge2"] += int(len(grimm) >= 2)
        out["turn_active_ready_grimm"] += int(
            int(active.get("id", -1)) == GRIMMSNARL_EX_ID
            and mf._dark_energy_count(active) >= SHADOW_BULLET_COST
        )
        out["turn_hand_energy"] += sum(
            int(int(c.get("id", -1)) == DARK_ENERGY_ID)
            for c in _cards(me, "hand")
        )
    return out


def _scan(payload: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    replay_root, rows = payload
    mf = _load_tables()
    per_source: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        path = Path(replay_root) / row["replay_name"] if replay_root else Path(
            row["replay_path"]
        )
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seat = int(row["seat_index"])
        steps = replay.get("steps") or []
        if len(steps) < 2:
            continue
        final = steps[-1]
        own = final[seat].get("reward") if seat < len(final) else None
        other = (
            final[1 - seat].get("reward")
            if 1 - seat < len(final) else None
        )
        won = int(own is not None and other is not None and own > other)

        went_first: int | None = None
        mirror = 0
        by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            record = step[seat] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            # firstPlayer is -1 until the coin flip resolves, and the seat
            # that acts during setup sees the sentinel on its first ACTIVE
            # step. Accepting it (it is not None) pinned went_first=0 on 50
            # of 96 v3_a games and 3603 of 3655 teacher games, which is what
            # made every earlier first/second split read as ~99% "second".
            first_player = mf._int(current.get("firstPlayer", -1))
            if went_first is None and first_player >= 0:
                went_first = int(
                    first_player == int(current.get("yourIndex", 0))
                )
            players = current.get("players") or [{}, {}]
            your = int(current.get("yourIndex", 0))
            if len(players) >= 2 and any(
                int(c.get("id", -1))
                in (GRIMMSNARL_EX_ID, IMPIDIMP_ID, MORGREM_ID)
                for c in mf._in_play(players[1 - your])
            ):
                mirror = 1
            if int(select.get("context", -1)) != MAIN:
                continue
            options = list(select.get("option") or [])
            action = (steps[index + 1][seat] or {}).get("action")
            if not (isinstance(action, list) and len(action) == 1
                    and isinstance(action[0], int)
                    and 0 <= action[0] < len(options)):
                continue
            by_turn[int(current.get("turn", -1))].append({
                "current": current,
                "select": select,
                "options": options,
                "chosen": action[0],
            })

        tally: Counter = Counter()
        for turn in sorted(by_turn):
            try:
                tally.update(_turn_tally(mf, by_turn[turn]))
            except Exception as error:  # noqa: BLE001 - keep scanning
                # Named, not swallowed: the first draft of this script ate a
                # TypeError on every turn that attached an energy, which zeroed
                # the attachment rate and cut the turn count by more than half
                # while still printing a plausible-looking table.
                tally["classify_errors"] += 1
                tally[f"error_{type(error).__name__}"] += 1
        tally["episodes"] = 1
        tally["wins"] = won

        source = str(row["source"])
        # Never fold "we could not tell" into "second" - that is exactly how
        # the sentinel bug above stayed invisible.
        seat_key = (
            "seatunknown" if went_first is None
            else "first" if went_first else "second"
        )
        keys = [source, f"{source}|{seat_key}"]
        if mirror:
            keys += [f"{source}|mirror", f"{source}|mirror|{seat_key}"]
        bucket = str(row.get("opp_bucket") or "")
        if bucket:
            keys += [f"{source}|{bucket}", f"{source}|{bucket}|{seat_key}"]
            if mirror:
                keys.append(f"{source}|{bucket}|mirror")
        for key in keys:
            per_source[key].update(tally)
    return {k: dict(v) for k, v in per_source.items()}


def _rates(counts: dict[str, int]) -> dict[str, Any]:
    episodes = max(1, int(counts.get("episodes", 0)))
    turns = max(1, int(counts.get("turns", 0)))

    def rate(taken: str, offered: str) -> float | None:
        total = int(counts.get(offered, 0))
        return round(int(counts.get(taken, 0)) / total, 4) if total else None

    def per_game(key: str) -> float:
        return round(int(counts.get(key, 0)) / episodes, 3)

    def per_turn(key: str) -> float:
        return round(int(counts.get(key, 0)) / turns, 4)

    return {
        "episodes": int(counts.get("episodes", 0)),
        "wins": int(counts.get("wins", 0)),
        "win_rate": round(int(counts.get("wins", 0)) / episodes, 4),
        "own_turns_per_game": per_game("turns"),
        "main_decisions_per_turn": per_turn("main_decisions"),

        # ---- the wall question, per turn, three strictness levels
        "wall_turns": int(counts.get("turn_walled_offered", 0)),
        "wall_attack_rate": rate("turn_walled_attacked",
                                 "turn_walled_offered"),
        "dead_swing_turns": int(counts.get("turn_dead_swing_offered", 0)),
        "dead_swing_attack_rate": rate("turn_dead_swing_attacked",
                                       "turn_dead_swing_offered"),
        "worthless_turns": int(counts.get("turn_worthless_offered", 0)),
        "worthless_attack_rate": rate("turn_worthless_attacked",
                                      "turn_worthless_offered"),
        "worthless_turns_per_game": per_game("turn_worthless_attacked"),
        "lethal_attack_rate": rate("turn_lethal_attacked",
                                   "turn_lethal_offered"),
        # Walled Active, Boss in hand, a damageable body on their Bench.
        "wall_unlock_turns": int(counts.get("turn_wall_unlock_offered", 0)),
        "wall_unlock_boss_rate": rate("turn_wall_unlock_bossed",
                                      "turn_wall_unlock_offered"),
        "wall_unlock_convert_rate": rate("turn_wall_unlock_converted",
                                         "turn_wall_unlock_offered"),
        "prize_swing_attack_rate": rate("turn_prize_swing_attacked",
                                        "turn_prize_swing_offered"),
        "attack_rate_when_offered": rate("turn_attacked",
                                         "turn_attack_offered"),
        "attacks_per_game": per_game("turn_attacked"),

        # ---- energy: one attachment per turn, so this is a real rate
        "energy_offer_share_of_turns": per_turn("turn_energy_offered"),
        "energy_take_rate": rate("turn_energy_taken", "turn_energy_offered"),
        "energy_per_game": per_game("turn_energy_taken"),
        "energy_to_grimmsnarl_per_game": per_game("turn_energy_to_grimmsnarl"),
        "energy_to_munkidori_per_game": per_game("turn_energy_to_munkidori"),
        "energy_to_prevo_per_game": per_game("turn_energy_to_prevo"),
        "hand_energy_per_turn": per_turn("turn_hand_energy"),
        # The narrow shape: an attachment that switches Adrena-Brain on or
        # completes Shadow Bullet's cost was legal, and the turn ended.
        "enabling_energy_turns": int(
            counts.get("turn_enabling_energy_offered", 0)
        ),
        "enabling_energy_take_rate": rate("turn_enabling_energy_taken",
                                          "turn_enabling_energy_offered"),
        "enabling_energy_hit_rate": rate(
            "turn_enabling_energy_taken_enabling",
            "turn_enabling_energy_offered",
        ),
        "enabling_energy_wasted_per_game": round(
            (int(counts.get("turn_enabling_energy_offered", 0))
             - int(counts.get("turn_enabling_energy_taken_enabling", 0)))
            / episodes, 3
        ),

        # ---- Adrena-Brain: availability against use
        "munkidori_in_play_share": per_turn("turn_munkidori_in_play"),
        "munkidori_fuelled_share": per_turn("turn_munkidori_fuelled"),
        "munkidori_all_dry_share": per_turn("turn_munkidori_all_dry"),
        "munkidori_count_per_turn": per_turn("turn_munkidori_count"),
        "munkidori_fuelled_count_per_turn": per_turn(
            "turn_munkidori_fuelled_count"
        ),
        "munkidori_offer_share_of_turns": per_turn(
            "turn_offer_ability_munkidori"
        ),
        "munkidori_take_rate": rate("turn_take_ability_munkidori",
                                    "turn_offer_ability_munkidori"),
        "munkidori_uses_per_game": per_game("turn_count_ability_munkidori"),
        "munkidori_uses_per_turn_when_fuelled": (
            round(int(counts.get("turn_count_ability_munkidori", 0))
                  / int(counts.get("turn_munkidori_fuelled", 1)), 3)
            if counts.get("turn_munkidori_fuelled") else None
        ),

        # ---- attacker build-up
        "grimmsnarl_count_per_turn": per_turn("turn_grimmsnarl_count"),
        "grimmsnarl_ready_per_turn": per_turn("turn_grimmsnarl_ready_count"),
        "two_grimmsnarl_share": per_turn("turn_grimmsnarl_ge2"),
        "active_ready_grimm_share": per_turn("turn_active_ready_grimm"),

        # ---- other decisions, per turn
        "boss_offer_share_of_turns": per_turn("turn_offer_boss"),
        "boss_take_rate": rate("turn_take_boss", "turn_offer_boss"),
        "boss_per_game": per_game("turn_take_boss"),
        "stamp_take_rate": rate("turn_take_stamp", "turn_offer_stamp"),
        "candy_take_rate": rate("turn_take_candy", "turn_offer_candy"),
        "bench_take_rate": rate("turn_take_bench", "turn_offer_bench"),
        "grimmsnarl_evolve_take_rate": rate("turn_take_evolve_grimmsnarl",
                                            "turn_offer_evolve_grimmsnarl"),
        "grimmsnarl_evolve_offer_turns": int(
            counts.get("turn_offer_evolve_grimmsnarl", 0)
        ),
        "froslass_evolve_take_rate": rate("turn_take_evolve_froslass",
                                          "turn_offer_evolve_froslass"),
        "froslass_evolve_offer_turns": int(
            counts.get("turn_offer_evolve_froslass", 0)
        ),
        "retreat_take_rate": rate("turn_take_retreat", "turn_offer_retreat"),
        "classify_errors": int(counts.get("classify_errors", 0)),
    }



def _chunks(
    rows: list[dict[str, Any]], workers: int
) -> list[list[dict[str, Any]]]:
    if workers <= 1:
        return [rows]
    size = max(1, (len(rows) + workers - 1) // workers)
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def _teacher_rows(
    data_root: Path, deck_hash: str, elite_teams: set[int]
) -> tuple[str, list[dict[str, Any]]]:
    index_path = data_root / "indexes" / "episodes.csv"
    if not index_path.exists():
        print(f"no teacher index at {index_path}", file=sys.stderr)
        return "", []
    index = pd.read_csv(index_path)
    index = index[index["download_status"] == "success"]
    if deck_hash:
        index = index[index["deck_hash"] == deck_hash]
    index = index.drop_duplicates(subset=["episode_id", "seat_index"])
    rows = []
    for record in index.to_dict("records"):
        team = int(record["team_id"])
        sources = [f"team_{team}", "teachers_all"]
        if team in elite_teams:
            sources.append("teachers_elite")
        for source in sources:
            rows.append({
                "source": source,
                "seat_index": int(record["seat_index"]),
                "replay_name": f"episode_{int(record['episode_id'])}.json",
            })
    return str(data_root / "replays"), rows


RATING_BUCKETS = ((0, 800), (800, 900), (900, 1000), (1000, 9999))


def _opponent_bucket(record: dict[str, Any], seat: int) -> str:
    """Tag for the opponent's rating when the match was paired.

    Ladder win rate tracks the opponent pool, not agent quality, so every
    ladder slice needs this alongside the seat. Runs fetched before
    2026-08-04 have no score columns and get no tag rather than a wrong one.
    """
    raw = record.get(f"agent_{1 - seat}_initial_score")
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return ""
    if score != score:  # NaN, which is what pandas gives a blank cell
        return ""
    for low, high in RATING_BUCKETS:
        if low <= score < high:
            return f"opp{low}" if high < 9999 else "opp1000up"
    return ""


def _ladder_rows(
    run_dir: Path, submission_id: int, label: str
) -> list[dict[str, Any]]:
    manifest = run_dir / "episodes.csv"
    if not manifest.exists():
        print(f"no ladder manifest at {manifest}", file=sys.stderr)
        return []
    frame = pd.read_csv(manifest, encoding="utf-8-sig")
    rows = []
    for record in frame.to_dict("records"):
        episode_id = int(record["episode_id"])
        seat = 0 if int(
            record.get("agent_0_submission_id", -1)
        ) == submission_id else 1
        path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        rows.append({
            "source": label,
            "seat_index": seat,
            "replay_name": "",
            "replay_path": str(path),
            "opp_bucket": _opponent_bucket(record, seat),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/kaggle_grimmsnarl_top50")
    parser.add_argument("--deck-hash", default="9714ab5c3996f6cc")
    parser.add_argument(
        "--elite-teams", default="16371703,16494330",
        help="team ids to pool as 'teachers_elite'",
    )
    parser.add_argument(
        "--ladder", action="append", default=[],
        metavar="LABEL:SUBMISSION_ID:RUN_DIR",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    elite = {int(x) for x in args.elite_teams.split(",") if x.strip()}
    teacher_root, teacher_rows = _teacher_rows(
        ROOT / args.data_root, args.deck_hash, elite
    )

    ladder_rows: list[dict[str, Any]] = []
    for spec in args.ladder:
        label, submission, run_dir = spec.split(":", 2)
        ladder_rows.extend(
            _ladder_rows(ROOT / run_dir, int(submission), label)
        )

    results: dict[str, Counter] = defaultdict(Counter)
    payloads = [
        (teacher_root, chunk) for chunk in _chunks(teacher_rows, args.workers)
    ]
    payloads += [("", chunk) for chunk in _chunks(ladder_rows, args.workers)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for out in pool.map(_scan, payloads):
            for source, counts in out.items():
                results[source].update(counts)

    report = {
        source: _rates(dict(counts))
        for source, counts in sorted(results.items())
    }
    raw = {source: dict(counts) for source, counts in sorted(results.items())}

    # Ladder labels print every slice, teachers only the pooled headline:
    # a ladder run is the thing being diagnosed, so its seat and
    # opponent-rating cuts are the point, and there are only a handful.
    ladder_labels = sorted({r["source"] for r in ladder_rows})
    ladder_slices = [
        source
        for source in report
        if any(
            source == label or source.startswith(f"{label}|")
            for label in ladder_labels
        )
    ]
    headline = [
        "teachers_all", "teachers_elite", "teachers_all|mirror",
        "teachers_all|second", "teachers_all|mirror|second",
    ] + ladder_slices
    for source in headline:
        stats = report.get(source)
        if not stats:
            continue
        print(
            f"{source:30s} n={stats['episodes']:5d} "
            f"win={stats['win_rate']:.3f} "
            f"turns={stats['own_turns_per_game']:5.2f} | "
            f"worthless {stats['worthless_turns']:4d}"
            f"@{stats['worthless_attack_rate']} "
            f"dead {stats['dead_swing_turns']:4d}"
            f"@{stats['dead_swing_attack_rate']} | "
            f"nrg {stats['energy_per_game']:.2f}@{stats['energy_take_rate']} "
            f"(G{stats['energy_to_grimmsnarl_per_game']:.2f}"
            f"/M{stats['energy_to_munkidori_per_game']:.2f}) | "
            f"munk fuel={stats['munkidori_fuelled_share']:.3f} "
            f"use={stats['munkidori_uses_per_game']:.2f}"
            f"@{stats['munkidori_take_rate']} | "
            f"boss {stats['boss_per_game']:.2f}@{stats['boss_take_rate']} | "
            f"Gready={stats['grimmsnarl_ready_per_turn']:.2f}"
        )

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"rates": report, "raw": raw}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
