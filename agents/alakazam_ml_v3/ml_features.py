from __future__ import annotations

from collections import Counter
from typing import Any

KEY_CARD_IDS = [5, 13, 19, 66, 140, 305, 343, 741, 742, 743, 1079, 1081, 1086, 1097, 1129, 1152, 1182, 1197, 1225, 1231, 1266]
ENERGY_IDS = {3, 5, 13, 17, 19}
BASIC_IDS = {66, 140, 305, 343, 741}
SPECIFIC_ACTIONS = {1182: "boss", 1197: "xerosic", 1081: "hammer"}
ROUTE_IDS = {741, 742, 743, 1079}


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [x for x in (player.get(area) or []) if isinstance(x, dict)]


def _in_play(player: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(player, "active") + _cards(player, "bench")


def _energy_cards(card: dict[str, Any]) -> list[Any]:
    return list(card.get("energyCards") or card.get("energies") or [])


def _energy_count(card: dict[str, Any]) -> int:
    return len(_energy_cards(card))


def _special_energy_count(card: dict[str, Any]) -> int:
    count = 0
    for energy in _energy_cards(card):
        if isinstance(energy, dict):
            card_id = int(energy.get("id", energy.get("cardId", -1)))
            if card_id not in {3, 5, 17}:
                count += 1
    return count


def _card_at_area(player: dict[str, Any], area: int | None, index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    if area == 4:
        seq = _cards(player, "active")
    elif area == 5:
        seq = _cards(player, "bench")
    else:
        return None
    return seq[index] if 0 <= index < len(seq) else None


def candidate_card(current: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    your = int(current.get("yourIndex", 0))
    player = current.get("players", [{}, {}])[your]
    option_type = int(option.get("type", -1))
    if option_type in (7, 8, 9):
        index = option.get("index")
        hand = _cards(player, "hand")
        return hand[index] if isinstance(index, int) and 0 <= index < len(hand) else None
    if option_type == 10:
        return _card_at_area(player, option.get("area"), option.get("index"))
    return None


def candidate_target(current: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    players = current.get("players", [{}, {}])
    your = int(current.get("yourIndex", 0))
    target_player = int(option.get("playerIndex", your))
    if target_player not in (0, 1):
        target_player = your
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return _card_at_area(players[target_player], area, index)


def action_type(current: dict[str, Any], option: dict[str, Any]) -> str:
    option_type = int(option.get("type", -1))
    if option_type == 13:
        return "attack"
    if option_type == 12:
        return "retreat"
    if option_type == 14:
        return "end"
    if option_type == 9:
        return "evolve"
    if option_type == 10:
        return "ability"
    if option_type in (7, 8):
        card = candidate_card(current, option) or {}
        card_id = int(card.get("id", -1))
        if card_id in SPECIFIC_ACTIONS:
            return SPECIFIC_ACTIONS[card_id]
        if option_type == 8 or card_id in ENERGY_IDS:
            return "energy"
        if card_id in BASIC_IDS:
            return "bench"
        return "trainer"
    return "other"


def _active_features(prefix: str, player: dict[str, Any], out: dict[str, float | int]) -> None:
    active = (_cards(player, "active") or [{}])[0]
    out[f"{prefix}_active_id"] = int(active.get("id", -1))
    out[f"{prefix}_active_hp"] = float(active.get("hp", 0))
    out[f"{prefix}_active_max_hp"] = float(active.get("maxHp", 0))
    out[f"{prefix}_active_damage"] = max(0.0, float(active.get("maxHp", 0)) - float(active.get("hp", 0)))
    out[f"{prefix}_active_energy"] = _energy_count(active)
    out[f"{prefix}_active_special_energy"] = _special_energy_count(active)
    out[f"{prefix}_active_tool_count"] = len(active.get("tools") or [])
    out[f"{prefix}_active_appear_this_turn"] = int(bool(active.get("appearThisTurn")))


def _bench_aggregates(prefix: str, player: dict[str, Any], out: dict[str, float | int]) -> None:
    bench = _cards(player, "bench")
    hps = [float(card.get("hp", 0)) for card in bench]
    max_hps = [float(card.get("maxHp", 0)) for card in bench]
    energies = [_energy_count(card) for card in bench]
    out[f"{prefix}_bench_open"] = max(0, int(player.get("benchMax", 5)) - len(bench))
    out[f"{prefix}_bench_min_hp"] = min(hps) if hps else 0.0
    out[f"{prefix}_bench_max_hp"] = max(hps) if hps else 0.0
    out[f"{prefix}_bench_max_energy"] = max(energies) if energies else 0
    out[f"{prefix}_bench_total_damage"] = sum(max(0.0, max_hp - hp) for hp, max_hp in zip(hps, max_hps))
    out[f"{prefix}_bench_damaged_count"] = sum(int(hp < max_hp) for hp, max_hp in zip(hps, max_hps))
    out[f"{prefix}_bench_low_hp_count"] = sum(int(hp <= 80) for hp in hps)


def state_features(current: dict[str, Any]) -> dict[str, float | int]:
    players = current.get("players", [{}, {}])
    your = int(current.get("yourIndex", 0))
    me, opp = players[your], players[1 - your]
    me_in_play, opp_in_play = _in_play(me), _in_play(opp)
    self_hand_count = int(me.get("handCount", len(me.get("hand") or [])))
    self_deck_count = int(me.get("deckCount", 0))
    turn = int(current.get("turn", 0))
    out: dict[str, float | int] = {
        "turn": turn,
        "turn_action_count": int(current.get("turnActionCount", 0)),
        "first_player_is_self": int(current.get("firstPlayer", -1) == your),
        "energy_attached": int(bool(current.get("energyAttached"))),
        "retreated": int(bool(current.get("retreated"))),
        "stadium_played": int(bool(current.get("stadiumPlayed"))),
        "supporter_played": int(bool(current.get("supporterPlayed"))),
        "self_hand_count": self_hand_count,
        "self_deck_count": self_deck_count,
        "self_prize_count": len(me.get("prize") or []),
        "self_bench_count": len(me.get("bench") or []),
        "opp_hand_count": int(opp.get("handCount", 0)),
        "opp_deck_count": int(opp.get("deckCount", 0)),
        "opp_prize_count": len(opp.get("prize") or []),
        "opp_bench_count": len(opp.get("bench") or []),
        "self_total_energy": sum(_energy_count(c) for c in me_in_play),
        "opp_total_energy": sum(_energy_count(c) for c in opp_in_play),
        "self_total_hp": sum(float(c.get("hp", 0)) for c in me_in_play),
        "opp_total_hp": sum(float(c.get("hp", 0)) for c in opp_in_play),
        "self_status_count": sum(int(bool(me.get(k))) for k in ("asleep", "burned", "confused", "paralyzed", "poisoned")),
        "opp_status_count": sum(int(bool(opp.get(k))) for k in ("asleep", "burned", "confused", "paralyzed", "poisoned")),
        "early_game": int(turn <= 4),
        "mid_game": int(5 <= turn <= 10),
        "late_game": int(turn >= 11),
        "deck_low_10": int(self_deck_count <= 10),
        "deck_low_5": int(self_deck_count <= 5),
        "current_powerful_hand_damage": 20 * self_hand_count,
    }
    _active_features("self", me, out)
    _active_features("opp", opp, out)
    _bench_aggregates("self", me, out)
    _bench_aggregates("opp", opp, out)

    hand_counts = Counter(int(c.get("id", -1)) for c in _cards(me, "hand"))
    field_counts = Counter(int(c.get("id", -1)) for c in me_in_play)
    discard_counts = Counter(int(c.get("id", -1)) for c in _cards(me, "discard"))
    opp_field_counts = Counter(int(c.get("id", -1)) for c in opp_in_play)
    for card_id in KEY_CARD_IDS:
        out[f"hand_{card_id}"] = hand_counts[card_id]
        out[f"field_{card_id}"] = field_counts[card_id]
        out[f"discard_{card_id}"] = discard_counts[card_id]
        out[f"opp_field_{card_id}"] = opp_field_counts[card_id]

    ready_alakazam = sum(int(int(card.get("id", -1)) == 743 and _energy_count(card) >= 1) for card in me_in_play)
    field_route_bodies = field_counts[741] + field_counts[742] + field_counts[743]
    out.update({
        "ready_alakazam_count": ready_alakazam,
        "has_ready_alakazam": int(ready_alakazam > 0),
        "field_route_body_count": field_route_bodies,
        "has_abra_anywhere": int(field_counts[741] + hand_counts[741] > 0),
        "has_bridge_anywhere": int(field_counts[742] + hand_counts[742] + hand_counts[1079] > 0),
        "has_alakazam_anywhere": int(field_counts[743] + hand_counts[743] > 0),
        "has_psychic_energy_in_hand": int(sum(hand_counts[x] for x in ENERGY_IDS) > 0),
        "complete_route_components": int(field_counts[741] + field_counts[742] + field_counts[743] > 0)
        + int(hand_counts[742] + hand_counts[1079] + field_counts[742] > 0)
        + int(hand_counts[743] + field_counts[743] > 0)
        + int(sum(hand_counts[x] for x in ENERGY_IDS) > 0),
        "needs_first_abra": int(field_counts[741] + field_counts[742] + field_counts[743] == 0),
        "needs_stage2": int(field_counts[743] == 0),
        "needs_attacker_energy": int(ready_alakazam == 0 and field_counts[743] > 0),
        "dudunsparce_engine_count": field_counts[66],
        "dunsparce_engine_count": field_counts[305],
    })
    stadium = current.get("stadium") or []
    out["stadium_id"] = int(stadium[0].get("id", -1)) if stadium and isinstance(stadium[0], dict) else -1
    return out


def option_features(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
    base_state: dict[str, float | int] | None = None,
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = dict(base_state) if base_state is not None else state_features(current)
    card = candidate_card(current, option) or {}
    target = candidate_target(current, option) or {}
    action = action_type(current, option)
    option_type = int(option.get("type", -1))
    card_id = int(card.get("id", -1))
    target_id = int(target.get("id", -1))
    target_energy = _energy_count(target)
    hand_cost = int(option_type in (7, 8, 9))
    post_hand_count = max(0, int(out["self_hand_count"]) - hand_cost)
    current_damage = int(out["current_powerful_hand_damage"])
    post_damage = 20 * post_hand_count
    opp_hp = float(out["opp_active_hp"])
    self_active_id = int(out["self_active_id"])
    is_powerful_hand_attack = int(option_type == 13 and self_active_id == 743)
    current_ko = int(self_active_id == 743 and current_damage >= opp_hp > 0)
    post_ko = int(self_active_id == 743 and post_damage >= opp_hp > 0)
    target_area = int(option.get("inPlayArea", option.get("area", -1)))

    out.update({
        "select_type": int(select.get("type", -1)),
        "select_context": int(select.get("context", -1)),
        "select_min_count": int(select.get("minCount", 0)),
        "select_max_count": int(select.get("maxCount", 0)),
        "option_type": option_type,
        "candidate_card_id": card_id,
        "candidate_attack_id": int(option.get("attackId", -1)),
        "candidate_area": int(option.get("area", -1)),
        "candidate_inplay_area": int(option.get("inPlayArea", -1)),
        "candidate_target_id": target_id,
        "candidate_target_hp": float(target.get("hp", 0)),
        "candidate_target_max_hp": float(target.get("maxHp", 0)),
        "candidate_target_energy": target_energy,
        "candidate_target_special_energy": _special_energy_count(target),
        "candidate_target_appear_this_turn": int(bool(target.get("appearThisTurn"))),
        "candidate_target_is_active": int(target_area == 4),
        "candidate_target_is_bench": int(target_area == 5),
        "candidate_hand_cost": hand_cost,
        "post_action_hand_count": post_hand_count,
        "post_action_powerful_hand_damage": post_damage,
        "current_ko_estimate": current_ko,
        "post_action_ko_estimate": post_ko,
        "breaks_current_ko_estimate": int(current_ko and not post_ko),
        "preserves_current_ko_estimate": int(not current_ko or post_ko),
        "powerful_hand_attack_option": is_powerful_hand_attack,
        "attack_lethal_estimate": int(is_powerful_hand_attack and current_damage >= opp_hp > 0),
        "attack_overkill_estimate": max(0.0, current_damage - opp_hp) if is_powerful_hand_attack else 0.0,
        "candidate_is_route_card": int(card_id in ROUTE_IDS),
        "candidate_is_abra": int(card_id == 741),
        "candidate_is_kadabra": int(card_id == 742),
        "candidate_is_alakazam": int(card_id == 743),
        "candidate_is_rare_candy": int(card_id == 1079),
        "candidate_is_psychic_energy": int(card_id in ENERGY_IDS),
        "target_is_abra": int(target_id == 741),
        "target_is_kadabra": int(target_id == 742),
        "target_is_alakazam": int(target_id == 743),
        "target_is_dunsparce": int(target_id == 305),
        "target_is_dudunsparce": int(target_id == 66),
        "candidate_fills_missing_abra": int(card_id == 741 and int(out["needs_first_abra"]) == 1),
        "candidate_fills_stage2": int(card_id == 743 and int(out["needs_stage2"]) == 1),
        "candidate_fills_energy": int(action == "energy" and int(out["needs_attacker_energy"]) == 1),
        "action_type": action,
    })
    for name in ("attack", "retreat", "end", "evolve", "ability", "energy", "boss", "xerosic", "hammer", "bench", "trainer", "other"):
        out[f"is_{name}"] = int(action == name)

    # Explicit action/state interactions make context learnable without requiring
    # deep trees to rediscover every conjunction from scratch.
    out.update({
        "hammer_opp_active_energy_value": int(action == "hammer") * int(out["opp_active_energy"]),
        "hammer_opp_active_special_energy_value": int(action == "hammer") * int(out["opp_active_special_energy"]),
        "xerosic_opp_hand_value": int(action == "xerosic") * int(out["opp_hand_count"]),
        "boss_opp_bench_value": int(action == "boss") * int(out["opp_bench_count"]),
        "boss_opp_bench_low_hp_value": int(action == "boss") * int(out["opp_bench_low_hp_count"]),
        "retreat_active_damage_value": int(action == "retreat") * float(out["self_active_damage"]),
        "retreat_status_value": int(action == "retreat") * int(out["self_status_count"]),
        "energy_target_existing_energy_value": int(action == "energy") * target_energy,
        "energy_target_is_alakazam_value": int(action == "energy" and target_id == 743),
        "energy_target_is_fezandipiti_value": int(action == "energy" and target_id == 140),
        "evolve_target_is_abra_value": int(action == "evolve" and target_id == 741),
        "evolve_target_is_kadabra_value": int(action == "evolve" and target_id == 742),
        "bench_early_game_value": int(action == "bench") * int(out["early_game"]),
        "ability_low_deck_risk": int(action == "ability") * int(out["deck_low_5"]),
        "trainer_low_deck_risk": int(action == "trainer") * int(out["deck_low_5"]),
        "end_has_attack_ready_penalty": int(action == "end") * int(out["has_ready_alakazam"]),
    })
    return out


LEAKAGE_DENYLIST = {
    "reward", "target_reward", "target_win", "target_loss", "result", "winner",
    "initial_deck_json", "deck_hash", "deck_type", "majkel_distance", "visualize",
}


def assert_no_leakage(feature_columns: list[str]) -> None:
    leaked = sorted(set(feature_columns) & LEAKAGE_DENYLIST)
    if leaked:
        raise ValueError(f"Leakage columns present in policy features: {leaked}")
