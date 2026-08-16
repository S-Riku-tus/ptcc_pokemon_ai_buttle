"""Within-turn line search over the official cg search API.

Why this and not the v7/v11/v27 line.  Those three all searched *forward past
our own turn* and needed a belief over the opponent's hidden cards plus a
learned value head to score the leaf; every one of them ended up gated to zero
overrides because neither ingredient is trustworthy.

This searches strictly to the end of our own turn and stops.  Inside our own
turn the only hidden information that matters is the order of *our* deck, and
we know our own 60-card list exactly, so the determinization is honest.  The
leaf is scored on public quantities only - prizes taken, damage dealt, whether
the attacker is still powered - so there is nothing to overfit.

The claim it can support is narrow and checkable: a greedy action-by-action
ranker maximises each decision in isolation and can therefore miss a *line*
whose first action looks worse.  Whether such lines exist at a useful rate is
measured by ``probe_turn_search.py``, not assumed.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable

MAIN = 0
OPT_ATTACK = 13
OPT_END = 14

# Card ids from our own list; a filler for the opponent's hidden zones that
# cannot act during our turn.
FILLER_CARD = 7  # Basic {D} Energy: no copy limit, no effect while idle.


class SearchUnavailable(RuntimeError):
    """The position cannot be searched honestly; the caller must not guess."""


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item) for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _card_ids(cards: Iterable[Any] | None) -> list[int]:
    out: list[int] = []
    for card in cards or ():
        if isinstance(card, dict):
            ident = card.get("id")
            if isinstance(ident, int):
                out.append(ident)
    return out


def _pokemon_ids(pokemon: dict[str, Any] | None) -> list[int]:
    """Every card id that a Pokemon in play accounts for."""
    if not isinstance(pokemon, dict):
        return []
    out = [int(pokemon["id"])] if isinstance(pokemon.get("id"), int) else []
    out += _card_ids(pokemon.get("energyCards"))
    out += _card_ids(pokemon.get("tools"))
    out += _card_ids(pokemon.get("preEvolution"))
    return out


def own_public_ids(current: dict[str, Any], seat: int) -> list[int]:
    """Every card of ours whose identity is already known from the state."""
    players = current.get("players") or []
    me = players[seat] if seat < len(players) else {}
    out: list[int] = []
    for card in (me.get("active") or []) + (me.get("bench") or []):
        out += _pokemon_ids(card)
    out += _card_ids(me.get("discard"))
    out += _card_ids(me.get("hand"))
    out += [c["id"] for c in (me.get("prize") or [])
            if isinstance(c, dict) and isinstance(c.get("id"), int)]
    for card in current.get("stadium") or []:
        if isinstance(card, dict) and int(card.get("playerIndex", -1)) == seat:
            if isinstance(card.get("id"), int):
                out.append(int(card["id"]))
    return out


def determinize(
    observation: dict[str, Any], deck_list: list[int], rng_shuffle: Callable | None
) -> list[list[int]]:
    """Hidden-zone arguments for ``search_begin``.

    Our own side is exact: the 60-card list minus everything already visible.
    The opponent's hidden zones are filled with inert Basic Energy, because the
    search never runs past the end of our own turn and the opponent therefore
    never draws, plays, or attacks inside it.
    """
    current = observation.get("current") or {}
    players = current.get("players") or []
    if len(players) < 2:
        raise SearchUnavailable("no player states")
    seat = int(current.get("yourIndex", 0))
    me, them = players[seat], players[1 - seat]

    if any(card is None for card in (them.get("active") or [])):
        raise SearchUnavailable("opponent Active is face down")

    remaining: dict[int, int] = {}
    for card_id in deck_list:
        remaining[card_id] = remaining.get(card_id, 0) + 1
    for card_id in own_public_ids(current, seat):
        if remaining.get(card_id, 0) <= 0:
            raise SearchUnavailable(f"public card {card_id} not in our list")
        remaining[card_id] -= 1

    pool: list[int] = []
    for card_id, count in sorted(remaining.items()):
        pool.extend([card_id] * count)
    if rng_shuffle is not None:
        rng_shuffle(pool)

    prize_slots = [c for c in (me.get("prize") or [])]
    unknown_prizes = sum(1 for c in prize_slots if not isinstance(c, dict))
    deck_count = int(me.get("deckCount", 0) or 0)
    if len(pool) != deck_count + unknown_prizes:
        raise SearchUnavailable(
            f"own hidden count mismatch: pool={len(pool)} "
            f"deck={deck_count} prizes={unknown_prizes}"
        )

    own_prize: list[int] = []
    cursor = 0
    for card in prize_slots:
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            own_prize.append(int(card["id"]))
        else:
            own_prize.append(pool[cursor])
            cursor += 1
    own_deck = pool[cursor:]

    opponent_deck = [FILLER_CARD] * int(them.get("deckCount", 0) or 0)
    opponent_prize = [FILLER_CARD] * len(them.get("prize") or [])
    opponent_hand = [FILLER_CARD] * int(them.get("handCount", 0) or 0)
    return [own_deck, own_prize, opponent_deck, opponent_prize, opponent_hand, []]


class SearchApi:
    """Thin wrapper over ``cg.api``'s search entry points."""

    def __init__(self) -> None:
        from cg import api

        self.api = api

    @staticmethod
    def _state(result: Any) -> dict[str, Any]:
        error = getattr(result, "error", None)
        if error is not None and int(error) != 0:
            raise SearchUnavailable(f"search error {int(error)}")
        state = getattr(result, "state", result)
        plain = _plain(state)
        if not isinstance(plain, dict) or not plain:
            raise SearchUnavailable("search returned no state")
        return plain

    def begin(self, observation: dict[str, Any], hidden: list[list[int]]) -> dict:
        converted = self.api.to_observation_class(observation)
        return self._state(self.api.search_begin(converted, *hidden, False))

    def step(self, search_id: int, selection: list[int]) -> dict:
        return self._state(self.api.search_step(int(search_id), list(selection)))

    def end(self) -> None:
        self.api.search_end()

    def release(self, search_id: int) -> None:
        self.api.search_release(int(search_id))


def leaf_value(
    current: dict[str, Any], seat: int, base_prizes: int, base_damage: int
) -> float:
    """Public score for the board at the end of our own turn.

    Deliberately dominated by prizes: the whole point of the layer is to find
    lines that convert, and anything softer would let a hand-tuned weight
    outvote a real prize.
    """
    result = int(current.get("result", -1))
    if result == seat:
        return 1e9
    if result == 1 - seat:
        return -1e9
    players = current.get("players") or []
    me, them = players[seat], players[1 - seat]
    prizes_taken = base_prizes - len(me.get("prize") or [])
    damage = 0
    for card in (them.get("active") or []) + (them.get("bench") or []):
        if isinstance(card, dict):
            damage += max(0, int(card.get("maxHp", 0)) - int(card.get("hp", 0)))
    bodies = len(them.get("active") or []) + len(them.get("bench") or [])
    our_bodies = len(me.get("active") or []) + len(me.get("bench") or [])
    return (
        100_000.0 * prizes_taken
        + 10.0 * (damage - base_damage)
        - 2_000.0 * bodies
        + 300.0 * our_bodies
    )


def total_damage(current: dict[str, Any], seat: int) -> int:
    players = current.get("players") or []
    them = players[1 - seat]
    return sum(
        max(0, int(card.get("maxHp", 0)) - int(card.get("hp", 0)))
        for card in (them.get("active") or []) + (them.get("bench") or [])
        if isinstance(card, dict)
    )


class TurnSearch:
    """Depth-first enumeration of the remainder of our own turn."""

    def __init__(
        self,
        deck_list: list[int],
        multi_pick: Callable[[dict, dict], list[int]] | None = None,
        max_nodes: int = 800,
        max_depth: int = 40,
        max_seconds: float = 20.0,
        branch_cap: int = 6,
        beam_width: int = 16,
    ) -> None:
        self.deck_list = list(deck_list)
        self.multi_pick = multi_pick
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_seconds = max_seconds
        self.branch_cap = branch_cap
        self.beam_width = beam_width
        self.api = SearchApi()
        self.stats: dict[str, Any] = {}

    # ---- candidate generation -------------------------------------------
    def _candidates(
        self, observation: dict[str, Any], seat: int, order: Callable | None
    ) -> list[list[int]]:
        select = observation.get("select") or {}
        options = select.get("option") or []
        minimum = int(select.get("minCount") or 0)
        maximum = int(select.get("maxCount") or 0)
        current = observation.get("current") or {}
        if int(current.get("yourIndex", seat)) != seat:
            # A forced selection on the opponent's side inside our own turn.
            return [list(range(min(minimum, len(options))))] if options else [[]]
        if maximum <= 1:
            indices = list(range(len(options)))
            if order is not None:
                indices = order(observation, indices)
            # Attacks end the turn and are the only way to take a prize, so
            # they are never allowed to fall off the end of a truncated prior
            # ordering.  Same for END: a line that stops early can be right.
            forced = [
                i for i, option in enumerate(options)
                if int(option.get("type", -1)) in (OPT_ATTACK, OPT_END)
            ]
            kept = list(dict.fromkeys(
                indices[: self.branch_cap] + forced
            ))
            if minimum == 0:
                return [[]] + [[i] for i in kept]
            return [[i] for i in kept]
        if self.multi_pick is not None:
            try:
                chosen = self.multi_pick(observation, select)
                if isinstance(chosen, list):
                    return [chosen]
            except Exception:  # noqa: BLE001
                pass
        return [list(range(min(maximum, len(options))))]

    # ---- baseline walk ----------------------------------------------------
    def walk(
        self,
        root: dict[str, Any],
        seat: int,
        start_turn: int,
        policy: Callable[[dict[str, Any]], list[int]],
        base_prizes: int,
        base_damage: int,
    ) -> dict[str, Any]:
        """Follow one policy from the root to the end of our own turn.

        Descends the same search tree the enumeration uses, so the comparison
        between the two is paired on one determinization rather than on two
        independent shuffles.
        """
        state = root
        path: list[list[int]] = []
        for depth in range(self.max_depth):
            observation = state.get("observation") or {}
            current = observation.get("current") or {}
            select = observation.get("select")
            if (
                select is None
                or int(current.get("result", -1)) >= 0
                or int(current.get("turn", start_turn)) != start_turn
            ):
                break
            if int(current.get("yourIndex", seat)) != seat:
                selection = list(range(int(select.get("minCount") or 0)))
            else:
                selection = policy(observation)
            state = self.api.step(state["searchId"], selection)
            path.append(list(selection))
        observation = state.get("observation") or {}
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        return {
            "path": path,
            "value": leaf_value(current, seat, base_prizes, base_damage),
            "prizes": base_prizes - len(players[seat].get("prize") or []),
            "damage": total_damage(current, seat) - base_damage,
            "result": int(current.get("result", -1)),
            "depth": len(path),
        }

    def prepare(
        self, observation: dict[str, Any], rng_shuffle: Callable | None = None
    ) -> tuple[dict[str, Any], int, int, int, int]:
        current = observation.get("current") or {}
        seat = int(current.get("yourIndex", 0))
        start_turn = int(current.get("turn", -1))
        players = current.get("players") or []
        base_prizes = len(players[seat].get("prize") or [])
        base_damage = total_damage(current, seat)
        hidden = determinize(observation, self.deck_list, rng_shuffle)
        root = self.api.begin(observation, hidden)
        return root, seat, start_turn, base_prizes, base_damage

    # ---- driver -----------------------------------------------------------
    def search(
        self,
        observation: dict[str, Any],
        order: Callable | None = None,
        rng_shuffle: Callable | None = None,
        prepared: tuple[dict[str, Any], int, int, int, int] | None = None,
        close: bool = True,
    ) -> list[dict[str, Any]]:
        """Return one record per distinct complete line of our remaining turn."""
        if prepared is None:
            prepared = self.prepare(observation, rng_shuffle)
        root, seat, start_turn, base_prizes, base_damage = prepared
        deadline = time.monotonic() + self.max_seconds
        lines: list[dict[str, Any]] = []
        nodes = 0
        truncated = False

        def describe(
            state: dict[str, Any], path: list[list[int]], depth: int,
            complete: bool = True,
        ) -> dict:
            node_current = (state.get("observation") or {}).get("current") or {}
            players = node_current.get("players") or [{}, {}]
            return {
                "path": [list(step) for step in path],
                "value": leaf_value(node_current, seat, base_prizes, base_damage),
                "prizes": base_prizes - len(players[seat].get("prize") or []),
                "damage": total_damage(node_current, seat) - base_damage,
                "result": int(node_current.get("result", -1)),
                "depth": depth,
                "complete": complete,
            }

        def finished(state: dict[str, Any]) -> bool:
            node_observation = state.get("observation") or {}
            node_current = node_observation.get("current") or {}
            return (
                node_observation.get("select") is None
                or int(node_current.get("result", -1)) >= 0
                or int(node_current.get("turn", start_turn)) != start_turn
            )

        # Breadth-limited beam.  Depth-first with a node cap explores one
        # arbitrary corner of the tree and never reaches the attack that a
        # different opening would have unlocked; a beam keeps the whole
        # frontier and is what makes the enumeration comparable to the greedy
        # walk it is judging.
        frontier = [(root, [], 0)]
        pruned = False
        while frontier and nodes < self.max_nodes:
            if time.monotonic() > deadline:
                truncated = True
                break
            out_of_budget = False
            children: list[tuple[dict, list, int]] = []
            for state, path, depth in frontier:
                if depth >= self.max_depth:
                    lines.append(describe(state, path, depth, complete=False))
                    continue
                observation_here = state.get("observation") or {}
                for selection in self._candidates(observation_here, seat, order):
                    if nodes >= self.max_nodes or time.monotonic() > deadline:
                        truncated = True
                        out_of_budget = True
                        break
                    nodes += 1
                    try:
                        child = self.api.step(state["searchId"], selection)
                    except Exception:  # noqa: BLE001
                        continue
                    child_path = path + [selection]
                    if finished(child):
                        lines.append(describe(child, child_path, depth + 1))
                    else:
                        children.append((child, child_path, depth + 1))
                if out_of_budget:
                    break
            children.sort(
                key=lambda item: describe(item[0], item[1], item[2])["value"],
                reverse=True,
            )
            if len(children) > self.beam_width:
                pruned = True
            frontier = children[: self.beam_width]

        if frontier:
            truncated = True
            for state, path, depth in frontier:
                lines.append(describe(state, path, depth, complete=False))

        if close:
            try:
                self.api.end()
            except Exception:  # noqa: BLE001
                pass
        self.stats = {
            "nodes": nodes, "lines": len(lines), "truncated": truncated,
            "pruned": pruned, "seat": seat, "turn": start_turn,
        }
        return lines
