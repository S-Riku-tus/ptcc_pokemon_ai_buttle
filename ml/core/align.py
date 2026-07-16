from __future__ import annotations

from typing import Any

from .features import candidate_card, candidate_target


def _logs(agent_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in ((agent_state.get("observation") or {}).get("logs") or []) if isinstance(x, dict)]


def _serial(card: dict[str, Any] | None) -> int | None:
    return int(card["serial"]) if card and "serial" in card else None


def _active_serial(current: dict[str, Any] | None, seat: int) -> int | None:
    if not current:
        return None
    players = current.get("players") or []
    if seat >= len(players):
        return None
    active = players[seat].get("active") or []
    return int(active[0].get("serial")) if active else None


def _transition_validation(
    current: dict[str, Any],
    options: list[dict[str, Any]],
    selected: int,
    next_agent_state: dict[str, Any],
) -> str:
    """Return a non-policy, alignment-only validation label.

    Kaggle replay rows store the response to observation t on the same seat in
    step t+1.  The returned integer is therefore authoritative.  Public logs in
    t+1 are used only to audit the decoded index; an inactive seat may retain
    stale logs, so missing validation is not treated as a contradiction.
    """
    option = options[selected]
    option_type = int(option.get("type", -1))
    logs = _logs(next_agent_state)
    your = int(current.get("yourIndex", 0))

    if option_type == 13:
        attack_id = int(option.get("attackId", -1))
        if any(int(x.get("type", -1)) == 15 and int(x.get("playerIndex", your)) == your and int(x.get("attackId", -2)) == attack_id for x in logs):
            return "attack_log"
    elif option_type in (7, 8):
        serial = _serial(candidate_card(current, option))
        expected = 11 if option_type == 8 else 10
        if serial is not None and any(int(x.get("type", -1)) == expected and int(x.get("playerIndex", your)) == your and int(x.get("serial", -2)) == serial for x in logs):
            return "card_log"
    elif option_type == 9:
        serial = _serial(candidate_card(current, option))
        target = _serial(candidate_target(current, option))
        if serial is not None and any(
            int(x.get("type", -1)) == 12
            and int(x.get("playerIndex", your)) == your
            and int(x.get("serial", -2)) == serial
            and (target is None or int(x.get("serialTarget", -3)) == target)
            for x in logs
        ):
            return "evolution_log"
    elif option_type == 10:
        source = _serial(candidate_card(current, option))
        if source is not None and any(int(x.get("playerIndex", your)) == your and int(x.get("serial", -2)) == source for x in logs):
            return "ability_log"
    elif option_type == 12:
        old = _active_serial(current, your)
        new = _active_serial((next_agent_state.get("observation") or {}).get("current"), your)
        if old is not None and new is not None and old != new:
            return "retreat_transition"
    elif option_type == 14:
        if any(int(x.get("type", -1)) == 3 and int(x.get("playerIndex", -1)) == your for x in logs):
            return "end_log"
    return "unvalidated"


def infer_main_selected_option(
    current: dict[str, Any],
    select: dict[str, Any],
    options: list[dict[str, Any]],
    next_agent_state: dict[str, Any],
) -> tuple[int | None, str, float]:
    """Decode the expert's legal main action from a Kaggle replay.

    In CABT an agent returns indices into ``select.option``.  Kaggle replay
    serialization places the action produced for observation t on the same
    seat in replay step t+1.  Reading the next seat's ``action`` is therefore
    exact and avoids fragile inference from event logs.
    """
    action = next_agent_state.get("action")
    if isinstance(action, list) and len(action) == 1:
        selected = action[0]
        if isinstance(selected, int) and 0 <= selected < len(options):
            validation = _transition_validation(current, options, selected, next_agent_state)
            method = "replay_option_index" if validation == "unvalidated" else f"replay_option_index+{validation}"
            return selected, method, 1.0
    return None, "missing_or_invalid_replay_action", 0.0
