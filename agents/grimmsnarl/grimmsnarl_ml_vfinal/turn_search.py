"""Within-turn line search with prize authority.

Three previous versions shipped a search layer and none of them ever changed a
played action: v7 overrode once in 1,706 decisions, v11 and v27 zero times.
The reason each time was the same - they searched *past* the end of our own
turn, so the leaf had to be scored by a value model over a believed opponent
hand, and the gates needed to make that trustworthy were tight enough to make
it inert.

This layer searches strictly to the end of our own turn and stops there.
Inside our own turn the only hidden information that matters is the order of
our own deck, and our 60-card list is known exactly, so the determinization is
honest rather than believed.  The leaf is then scored on one thing the value
model was never needed for: how many prizes the line takes.

Authority is correspondingly narrow.  The layer may only overrule the v22
ranker when some line starting from a different action takes strictly more
prizes this turn - or wins outright - than any line starting from the ranker's
own action, and when that holds on every determinization sampled.  Anything
else, any exception, any budget pressure, and the ranker's answer stands.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Callable

MAIN_CONTEXT = 0
OPT_ATTACK = 13
OPT_END = 14

# Basic {D} Energy.  Fills the opponent's hidden zones: they never draw, play
# or attack inside our own turn, so the contents cannot change the line, and a
# basic Energy has no copy limit to violate.
FILLER_CARD = 7

# OFF, on measurement.  Paired 320-game local mirror arenas against v22, with
# the arena calibrated on a null control (v22 against a byte-identical copy of
# itself scored 0.4917 [0.429, 0.554] over 240 games with a 59/120-59/120 seat
# split, so the instrument is unbiased):
#
#   override the opening only         0.478  [0.424, 0.533]
#   commit the whole line             0.344  [0.294, 0.397]
#
# The layer finds real extra prizes - 4.7% of our turns have a line taking a
# prize the ranker's action cannot, and committing collects 36 where the ranker
# takes 10 - but converting them costs more than they are worth, and the more
# faithfully the prize-maximal line is executed the worse the agent plays.
# Prizes-taken-this-turn is not a sufficient objective for a turn.  The module
# ships disabled so the result is reproducible, not so it can be switched on.
ENABLED = False

DEFAULT_MAX_NODES = 6000
DEFAULT_BEAM = 32
DEFAULT_BRANCH = 30
DEFAULT_DEPTH = 44
DEFAULT_DETERMINIZATIONS = 2
DEFAULT_PER_DECISION_SECONDS = 3.5
MIN_TURN = 3

# The Kaggle bank is 600 s per episode and actTimeout is 0.  v22 itself spends
# 5-20 s of it, so almost all of it is free; keep a wide reserve anyway.
OVERAGE_RESERVE_SECONDS = 150.0
MAX_GAME_SEARCH_SECONDS = 300.0


class SearchUnavailable(RuntimeError):
    """The position cannot be searched honestly."""


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


def _card_ids(cards: Any) -> list[int]:
    out: list[int] = []
    for card in cards or ():
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            out.append(int(card["id"]))
    return out


def _pokemon_ids(pokemon: Any) -> list[int]:
    if not isinstance(pokemon, dict):
        return []
    out = [int(pokemon["id"])] if isinstance(pokemon.get("id"), int) else []
    out += _card_ids(pokemon.get("energyCards"))
    out += _card_ids(pokemon.get("tools"))
    out += _card_ids(pokemon.get("preEvolution"))
    return out


def own_public_ids(current: dict[str, Any], seat: int) -> list[int]:
    players = current.get("players") or []
    me = players[seat] if seat < len(players) else {}
    out: list[int] = []
    for card in (me.get("active") or []) + (me.get("bench") or []):
        out += _pokemon_ids(card)
    out += _card_ids(me.get("discard"))
    out += _card_ids(me.get("hand"))
    out += _card_ids(me.get("prize"))
    for card in current.get("stadium") or []:
        if (
            isinstance(card, dict)
            and int(card.get("playerIndex", -1)) == seat
            and isinstance(card.get("id"), int)
        ):
            out.append(int(card["id"]))
    return out


def determinize(
    observation: dict[str, Any], deck_list: list[int], rng: random.Random
) -> list[list[int]]:
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
            raise SearchUnavailable("public card outside our list")
        remaining[card_id] -= 1

    pool: list[int] = []
    for card_id, count in sorted(remaining.items()):
        pool.extend([card_id] * count)
    rng.shuffle(pool)

    prize_slots = list(me.get("prize") or [])
    unknown_prizes = sum(1 for card in prize_slots if not isinstance(card, dict))
    deck_count = int(me.get("deckCount", 0) or 0)
    if len(pool) != deck_count + unknown_prizes:
        raise SearchUnavailable("own hidden count mismatch")

    own_prize: list[int] = []
    cursor = 0
    for card in prize_slots:
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            own_prize.append(int(card["id"]))
        else:
            own_prize.append(pool[cursor])
            cursor += 1
    own_deck = pool[cursor:]
    return [
        own_deck,
        own_prize,
        [FILLER_CARD] * int(them.get("deckCount", 0) or 0),
        [FILLER_CARD] * len(them.get("prize") or []),
        [FILLER_CARD] * int(them.get("handCount", 0) or 0),
        [],
    ]


AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12
_AREA_KEY = {
    AREA_HAND: "hand", AREA_DISCARD: "discard",
    AREA_ACTIVE: "active", AREA_BENCH: "bench", AREA_PRIZE: "prize",
}


# Cards already in play keep the serial they had when the search was opened,
# so a Pokemon can be named exactly.  Cards that come out of the deck get a
# serial from the determinization and can only be named by card id - which is
# the right granularity there anyway, since two copies are interchangeable.
_SERIAL_AREAS = {AREA_ACTIVE, AREA_BENCH, AREA_DISCARD, AREA_STADIUM}


def _card_at(observation: dict[str, Any], area, index, player) -> tuple | None:
    """What an ``(area, index, playerIndex)`` option refers to, by identity.

    The engine does **not** populate ``Option.cardId`` - measured over a whole
    stored episode, 1,993 options and not one of them carried it - so an option
    is named purely by position.  A plan replayed against a differently
    shuffled deck therefore has to resolve the position back to a card here, or
    it will play whatever happens to sit at that index instead.
    """
    if area is None or index is None:
        return None
    area, index = int(area), int(index)
    if area == AREA_PRIZE:
        # Prize cards are face down.  Which one we take is not a decision we
        # have information about, every prize option is interchangeable, and
        # the identity the determinization invented for it will never match
        # the real one - so identity must not enter the signature at all.
        return None
    current = observation.get("current") or {}
    if area == AREA_DECK:
        zone = (observation.get("select") or {}).get("deck") or []
    elif area == AREA_LOOKING:
        zone = current.get("looking") or []
    elif area == AREA_STADIUM:
        zone = current.get("stadium") or []
    else:
        key = _AREA_KEY.get(area)
        if key is None:
            return None
        players = current.get("players") or []
        seat = int(player if player is not None else current.get("yourIndex", 0))
        if not 0 <= seat < len(players):
            return None
        zone = players[seat].get(key) or []
    card = zone[index] if 0 <= index < len(zone) else None
    if not isinstance(card, dict):
        return None
    if area in _SERIAL_AREAS and isinstance(card.get("serial"), int):
        return ("serial", int(card["serial"]))
    if isinstance(card.get("id"), int):
        return ("card", int(card["id"]))
    return None


def option_signature(
    option: dict[str, Any], observation: dict[str, Any] | None = None
) -> tuple:
    """Identity of an option, by card rather than by position where possible.

    Without ``observation`` this falls back to the positional identity, which
    is correct for comparing options *within one* observation (the override
    decision) but not for replaying a plan against a reshuffled board.
    """
    resolved = option.get("index")
    target = option.get("inPlayIndex")
    if observation is not None:
        resolved = _card_at(
            observation, option.get("area"), option.get("index"),
            option.get("playerIndex"),
        )
        # The in-play coordinates name a Pokemon by Bench slot, and a Bench
        # slot is not stable once something is knocked out.
        target = _card_at(
            observation, option.get("inPlayArea"), option.get("inPlayIndex"),
            option.get("playerIndex"),
        )
    return (
        int(option.get("type", -1)),
        option.get("area"),
        resolved,
        option.get("inPlayArea"),
        target,
        option.get("attackId"),
        option.get("number"),
        option.get("playerIndex"),
        option.get("energyIndex"),
        option.get("toolIndex"),
        option.get("specialConditionType"),
    )


def resolve_selection(
    observation: dict[str, Any], wanted: tuple
) -> list[int] | None:
    """Indices matching each signature in ``wanted``, or None if any is absent."""
    options = (observation.get("select") or {}).get("option") or []
    used: set[int] = set()
    out: list[int] = []
    for signature in wanted:
        match = None
        for index, option in enumerate(options):
            if index in used:
                continue
            if option_signature(option, observation) == signature:
                match = index
                break
        if match is None:
            return None
        used.add(match)
        out.append(match)
    return out


SHADOW_BULLET_DAMAGE = 180
DARKNESS = 7
_WEAKNESS: dict[int, int] = {}


def load_weakness_table() -> dict[int, int]:
    """Card id -> Weakness energy type, from the engine's own card table.

    Needed to know whether Shadow Bullet's 180 is really 360 against a given
    Active.  ``cards.json`` is not in the submission bundle, but the native
    library ships the same table and ``all_card_data`` exposes it.
    """
    global _WEAKNESS
    if _WEAKNESS:
        return _WEAKNESS
    try:
        from cg import api

        table = {}
        for card in api.all_card_data():
            card_id = getattr(card, "cardId", None)
            weakness = getattr(card, "weakness", None)
            if isinstance(card_id, int) and isinstance(weakness, int):
                table[card_id] = weakness
        _WEAKNESS = table
    except Exception:  # noqa: BLE001
        _WEAKNESS = {}
    return _WEAKNESS


def in_shadow_range(card: Any) -> bool:
    """Would one more Shadow Bullet knock this Pokemon out where it stands?"""
    if not isinstance(card, dict):
        return False
    hit = SHADOW_BULLET_DAMAGE
    if load_weakness_table().get(int(card.get("id", -1))) == DARKNESS:
        hit *= 2
    return int(card.get("hp", 10 ** 6)) <= hit


def total_damage(current: dict[str, Any], seat: int) -> int:
    players = current.get("players") or [{}, {}]
    them = players[1 - seat]
    return sum(
        max(0, int(card.get("maxHp", 0)) - int(card.get("hp", 0)))
        for card in (them.get("active") or []) + (them.get("bench") or [])
        if isinstance(card, dict)
    )


class SearchApi:
    def __init__(self) -> None:
        from cg import api

        self.api = api

    @staticmethod
    def _state(result: Any) -> dict[str, Any]:
        error = getattr(result, "error", None)
        if error is not None and int(error) != 0:
            raise SearchUnavailable(f"search error {int(error)}")
        plain = _plain(getattr(result, "state", result))
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


class Budget:
    """Track the Kaggle overage bank so the layer can never time the game out."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.spent = 0.0
        self.searches = 0
        self.overrides = 0
        self.considered = 0
        self.skipped_budget = 0
        self.errors = 0
        # Why a search that ran did not produce an override.  v7/v11/v27 all
        # reported "zero overrides" without ever separating "no better line
        # exists" from "the gate refused a better line", so both are counted.
        self.no_lines = 0
        self.no_baseline = 0
        self.no_improvement = 0
        self.world_disagreement = 0
        self.same_index = 0
        self.improved_first_world = 0
        self.plans = 0
        self.plan_steps = 0
        self.plans_abandoned = 0
        self.abandon_at: dict[str, int] = {}
        self.no_committable = 0
        self.last_remaining: float | None = None

    def note(self, observation: dict[str, Any]) -> None:
        raw = observation.get("remainingOverageTime")
        if not isinstance(raw, (int, float)):
            return
        remaining = float(raw)
        if self.last_remaining is not None and remaining > self.last_remaining + 30.0:
            self.reset()
        self.last_remaining = remaining

    def available(self) -> float:
        own = MAX_GAME_SEARCH_SECONDS - self.spent
        bank = (
            float("inf") if self.last_remaining is None
            else self.last_remaining - OVERAGE_RESERVE_SECONDS
        )
        return max(0.0, min(own, bank))

    def charge(self, seconds: float) -> None:
        self.spent += max(0.0, float(seconds))
        self.searches += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "search_seconds": round(self.spent, 2),
            "searches": self.searches,
            "considered": self.considered,
            "overrides": self.overrides,
            "skipped_budget": self.skipped_budget,
            "errors": self.errors,
            "no_lines": self.no_lines,
            "no_baseline": self.no_baseline,
            "no_improvement": self.no_improvement,
            "world_disagreement": self.world_disagreement,
            "same_index": self.same_index,
            "improved_first_world": self.improved_first_world,
            "plans": self.plans,
            "plan_steps": self.plan_steps,
            "plans_abandoned": self.plans_abandoned,
            "abandon_at": dict(self.abandon_at),
            "no_committable": self.no_committable,
            "overage_last": (
                round(self.last_remaining, 1)
                if self.last_remaining is not None else -1.0
            ),
        }


class TurnSearch:
    def __init__(
        self,
        deck_list: list[int],
        multi_pick: Callable[[dict, dict], list[int]] | None = None,
        max_nodes: int = DEFAULT_MAX_NODES,
        beam_width: int = DEFAULT_BEAM,
        branch_cap: int = DEFAULT_BRANCH,
        max_depth: int = DEFAULT_DEPTH,
        determinizations: int = DEFAULT_DETERMINIZATIONS,
        per_decision_seconds: float = DEFAULT_PER_DECISION_SECONDS,
        authority: int = 1,
        commit_plan: bool = True,
        state_guard: tuple[Callable[[], Any], Callable[[Any], None]] | None = None,
    ) -> None:
        self.deck_list = list(deck_list)
        self.multi_pick = multi_pick
        self.max_nodes = max_nodes
        self.beam_width = beam_width
        self.branch_cap = branch_cap
        self.max_depth = max_depth
        self.determinizations = max(1, determinizations)
        self.per_decision_seconds = per_decision_seconds
        self.authority = authority
        self.commit_plan = commit_plan
        self.state_guard = state_guard
        self.plan: dict[str, Any] | None = None
        self.api = SearchApi()
        self.budget = Budget()
        self.last: dict[str, Any] = {}

    # ---- candidates ------------------------------------------------------
    def _candidates(self, observation: dict[str, Any], seat: int) -> list[list[int]]:
        select = observation.get("select") or {}
        options = select.get("option") or []
        minimum = int(select.get("minCount") or 0)
        maximum = int(select.get("maxCount") or 0)
        current = observation.get("current") or {}
        if int(current.get("yourIndex", seat)) != seat:
            return [list(range(min(minimum, len(options))))] if options else [[]]
        if maximum <= 1:
            forced = [
                index for index, option in enumerate(options)
                if int(option.get("type", -1)) in (OPT_ATTACK, OPT_END)
            ]
            kept = list(dict.fromkeys(list(range(len(options)))[: self.branch_cap]
                                      + forced))
            if minimum == 0:
                return [[]] + [[index] for index in kept]
            return [[index] for index in kept]
        if self.multi_pick is not None:
            try:
                chosen = self.multi_pick(observation, select)
                if isinstance(chosen, list):
                    return [chosen]
            except Exception:  # noqa: BLE001
                pass
        return [list(range(min(maximum, len(options))))]

    # ---- one determinization --------------------------------------------
    def _lines(
        self, observation: dict[str, Any], seat: int, rng: random.Random,
        deadline: float,
    ) -> list[dict[str, Any]]:
        current = observation.get("current") or {}
        start_turn = int(current.get("turn", -1))
        players = current.get("players") or []
        base_prizes = len(players[seat].get("prize") or [])
        base_damage = total_damage(current, seat)
        base_threat = int(in_shadow_range(
            ((players[1 - seat].get("active") or [None]) or [None])[0]
        ))

        hidden = determinize(observation, self.deck_list, rng)
        root = self.api.begin(observation, hidden)
        lines: list[dict[str, Any]] = []
        nodes = 0

        def threat_of(node_current) -> int:
            them = (node_current.get("players") or [{}, {}])[1 - seat]
            active = (them.get("active") or [None])[0]
            return int(in_shadow_range(active))

        def describe(state, path, signatures, complete, uses_deck=False):
            node_current = (state.get("observation") or {}).get("current") or {}
            after = (node_current.get("players") or [{}, {}])[seat]
            return {
                "first": list(path[0]) if path else [],
                "first_signature": signatures[0][2] if signatures else (),
                # Only our own selections: the opponent also gets asked things
                # inside our turn, and the live agent is never shown those, so
                # a plan that included them would desynchronise immediately.
                "plan": [(ctx, sig) for ours, ctx, sig in signatures if ours],
                # A line that reaches into the deck was planned against one
                # sampled deck order.  Live, the card it wanted may be in the
                # prizes instead, the step will not resolve, and the agent is
                # left half-way through someone else's turn - which is the
                # exact state ``probe_commitment.py`` measured as worse than
                # not overriding at all.  Only deck-free lines are committable.
                "uses_deck": uses_deck,
                "prizes": base_prizes - len(after.get("prize") or []),
                "damage": total_damage(node_current, seat) - base_damage,
                "threat": threat_of(node_current) - base_threat,
                "result": int(node_current.get("result", -1)),
                "complete": complete,
            }

        def finished(state) -> bool:
            node_observation = state.get("observation") or {}
            node_current = node_observation.get("current") or {}
            return (
                node_observation.get("select") is None
                or int(node_current.get("result", -1)) >= 0
                or int(node_current.get("turn", start_turn)) != start_turn
            )

        def rank(item) -> float:
            state = item[0]
            node_current = (state.get("observation") or {}).get("current") or {}
            after = (node_current.get("players") or [{}, {}])[seat]
            taken = base_prizes - len(after.get("prize") or [])
            return 100_000.0 * taken + (
                total_damage(node_current, seat) - base_damage
            )

        frontier = [(root, [], [], 0, False)]
        try:
            while frontier and nodes < self.max_nodes:
                if time.monotonic() > deadline:
                    break
                children = []
                stop = False
                for state, path, signatures, depth, uses_deck in frontier:
                    if depth >= self.max_depth:
                        lines.append(
                            describe(state, path, signatures, False, uses_deck)
                        )
                        continue
                    node_observation = state.get("observation") or {}
                    options = (node_observation.get("select") or {}).get("option") or []
                    node_current = node_observation.get("current") or {}
                    ours = int(node_current.get("yourIndex", seat)) == seat
                    node_select = node_observation.get("select") or {}
                    deck_offer = bool(node_select.get("deck"))
                    for selection in self._candidates(node_observation, seat):
                        if nodes >= self.max_nodes or time.monotonic() > deadline:
                            stop = True
                            break
                        nodes += 1
                        try:
                            child = self.api.step(state["searchId"], selection)
                        except Exception:  # noqa: BLE001
                            continue
                        signature = tuple(
                            option_signature(options[index], node_observation)
                            for index in selection
                            if 0 <= index < len(options)
                        )
                        child_path = path + [selection]
                        child_signatures = signatures + [
                            (ours, int(node_select.get("context", -1)), signature)
                        ]
                        # A step reaches into the deck if any option it
                        # picked lives in the deck zone.  ``select.deck`` alone
                        # is not enough: the engine leaves it unset on several
                        # search contexts whose options still carry area=DECK.
                        child_deck = uses_deck or (
                            bool(selection)
                            and (
                                deck_offer
                                or any(
                                    int((options[i] or {}).get("area") or 0)
                                    == AREA_DECK
                                    for i in selection
                                    if 0 <= i < len(options)
                                )
                            )
                        )
                        if finished(child):
                            lines.append(describe(
                                child, child_path, child_signatures, True,
                                child_deck,
                            ))
                        else:
                            children.append((
                                child, child_path, child_signatures,
                                depth + 1, child_deck,
                            ))
                    if stop:
                        break
                # Diverse beam.  A plain top-k beam can prune every descendant
                # of the ranker's own first action before that branch reaches
                # the end of the turn, and then there is no baseline to judge
                # against - which is exactly what made the first build of this
                # layer refuse 24% of its searches.  Reserving a slot per
                # distinct first action guarantees every candidate opening
                # survives to a complete line.
                groups: dict[tuple, list] = {}
                for item in children:
                    groups.setdefault(item[2][0][2], []).append(item)
                for bucket in groups.values():
                    bucket.sort(key=rank, reverse=True)
                per_group = max(1, self.beam_width // max(1, len(groups)))
                frontier = []
                leftovers: list = []
                for bucket in groups.values():
                    frontier.extend(bucket[:per_group])
                    leftovers.extend(bucket[per_group:])
                if len(frontier) < self.beam_width and leftovers:
                    leftovers.sort(key=rank, reverse=True)
                    frontier.extend(leftovers[: self.beam_width - len(frontier)])
        finally:
            try:
                self.api.end()
            except Exception:  # noqa: BLE001
                pass
        return lines

    # ---- authority --------------------------------------------------------
    def suggest(
        self, observation: dict[str, Any], ranker_index: int
    ) -> int | None:
        """Return an index to play instead of ``ranker_index``, or None."""
        self.last = {}
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        if int(select.get("context", -1)) != MAIN_CONTEXT:
            return None
        if int(select.get("maxCount") or 0) != 1:
            return None
        options = select.get("option") or []
        if not 0 <= ranker_index < len(options):
            return None
        if int(current.get("turn", 0)) < MIN_TURN:
            return None
        if not observation.get("search_begin_input"):
            return None
        self.budget.considered += 1
        allowance = min(self.per_decision_seconds, self.budget.available())
        if allowance < 0.4:
            self.budget.skipped_budget += 1
            return None

        seat = int(current.get("yourIndex", 0))
        # ``first_signature`` is the tuple of signatures of every option picked
        # in the first step, so a single-pick MAIN choice is a 1-tuple.
        chosen_signature = (
            option_signature(options[ranker_index], observation),
        )
        started = time.monotonic()
        per_world = max(0.2, allowance / self.determinizations)
        winners: list[tuple] = []
        margins: list[int] = []
        plan_line: dict[str, Any] | None = None
        # Running the rule policy inside the search writes to its module-level
        # per-turn caches; the search must leave no trace on the live game.
        saved = None
        if self.state_guard is not None:
            try:
                saved = self.state_guard[0]()
            except Exception:  # noqa: BLE001
                saved = None
        try:
            for world in range(self.determinizations):
                rng = random.Random(
                    (int(current.get("turn", 0)) << 8)
                    ^ (int(current.get("turnActionCount", 0)) << 4)
                    ^ world
                    ^ 0x5EED
                )
                lines = self._lines(
                    observation, seat, rng, time.monotonic() + per_world
                )
                complete = [line for line in lines if line["complete"]]
                if not complete:
                    self.budget.no_lines += 1
                    return None
                # The baseline is the best the ranker's own opening can do,
                # over *any* line: judging it only on committable lines would
                # understate it and manufacture overrides.
                best_by_first: dict[tuple, tuple[int, int, int, int]] = {}
                challengers: dict[tuple, tuple[int, int, int, int]] = {}
                best_line: dict[tuple, dict[str, Any]] = {}
                for line in complete:
                    key = line["first_signature"]
                    score = (
                        1 if line["result"] == seat else 0,
                        line["prizes"],
                        line["threat"],
                        line["damage"],
                    )
                    if key not in best_by_first or score > best_by_first[key]:
                        best_by_first[key] = score
                    if line.get("uses_deck"):
                        continue
                    if key not in challengers or score > challengers[key]:
                        challengers[key] = score
                        best_line[key] = line
                if chosen_signature not in best_by_first:
                    self.budget.no_baseline += 1
                    return None
                if not challengers:
                    self.budget.no_committable += 1
                    return None
                baseline = best_by_first[chosen_signature]
                best_key, best_score = max(
                    challengers.items(), key=lambda item: item[1]
                )
                # Level 1: only a win or an extra prize may overrule a policy
                # that imitates a 1220-rated pilot.  Level 2 additionally
                # allows a line that leaves the opponent's Active inside
                # Shadow Bullet range when the ranker's line does not, which
                # is the one damage difference that is certain to be worth
                # something next turn.
                improvement = (
                    best_score[0] > baseline[0]
                    or best_score[1] > baseline[1]
                    or (
                        self.authority >= 2
                        and best_score[:2] == baseline[:2]
                        and best_score[2] > baseline[2]
                    )
                )
                if not improvement:
                    self.budget.no_improvement += 1
                    return None
                if world == 0:
                    self.budget.improved_first_world += 1
                    plan_line = best_line[best_key]
                winners.append(best_key)
                margins.append(best_score[1] - baseline[1])
        except SearchUnavailable:
            return None
        except Exception:  # noqa: BLE001
            self.budget.errors += 1
            return None
        finally:
            if self.state_guard is not None and saved is not None:
                try:
                    self.state_guard[1](saved)
                except Exception:  # noqa: BLE001
                    pass
            self.budget.charge(time.monotonic() - started)

        if len(set(winners)) != 1:
            self.budget.world_disagreement += 1
            return None
        target = winners[0]
        if len(target) != 1:
            return None
        for index, option in enumerate(options):
            if (option_signature(option, observation),) == target:
                if index == ranker_index:
                    self.budget.same_index += 1
                    return None
                self.budget.overrides += 1
                # Commit the rest of the line.  Measured on 90 stored games:
                # playing only the opening and handing the turn back to the
                # ranker collected 19 of the 36 prizes the line was worth and
                # did 73 less damage per turn than not overriding at all
                # (``probe_commitment.py``).  An opening without its
                # continuation is worse than no opening.
                if plan_line and len(plan_line["plan"]) > 1:
                    if os.environ.get("GRIMMSNARL_SEARCH_DEBUG") == "1":
                        print("PLAN", plan_line.get("uses_deck"),
                              [c for c, _ in plan_line["plan"]], flush=True)
                    self.plan = {
                        "turn": int(current.get("turn", -1)),
                        "steps": list(plan_line["plan"][1:]),
                        "cursor": 0,
                    }
                    self.budget.plans += 1
                self.last = {
                    "turn": int(current.get("turn", -1)),
                    "from": ranker_index,
                    "to": index,
                    "prize_margin": margins[0] if margins else 0,
                }
                return index
        return None

    # ---- committed plan replay -------------------------------------------
    def planned(self, observation: dict[str, Any]) -> list[int] | None:
        """The next selection of a committed line, or None to drop the plan."""
        if not self.plan:
            return None
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        if int(current.get("turn", -1)) != self.plan["turn"]:
            self.plan = None
            return None
        cursor = self.plan["cursor"]
        steps = self.plan["steps"]
        if cursor >= len(steps):
            self.plan = None
            return None
        wanted_context, wanted = steps[cursor]
        live_context = int(select.get("context", -1))
        if live_context != wanted_context:
            # The live turn is being asked something the plan does not
            # describe; the two have desynchronised.
            self.plan = None
            self.budget.plans_abandoned += 1
            key = f"desync live{live_context}!=plan{wanted_context}@{cursor}"
            self.budget.abandon_at[key] = self.budget.abandon_at.get(key, 0) + 1
            return None
        indices = resolve_selection(observation, wanted)
        minimum = int(select.get("minCount") or 0)
        maximum = int(select.get("maxCount") or 0)
        if (
            indices is None
            or len(indices) < minimum
            or (maximum and len(indices) > maximum)
        ):
            # The real deck order took the turn somewhere the plan does not
            # describe.  Abandon rather than play a half-understood line.
            self.plan = None
            self.budget.plans_abandoned += 1
            key = (
                f"ctx{live_context}/step{cursor}of{len(steps)}"
                f"/{'unresolved' if indices is None else 'count'}"
            )
            self.budget.abandon_at[key] = self.budget.abandon_at.get(key, 0) + 1
            return None
        self.plan["cursor"] = cursor + 1
        self.budget.plan_steps += 1
        return indices

    def drop_plan(self) -> None:
        self.plan = None

    def reset(self) -> None:
        self.budget.reset()
        self.plan = None
        self.last = {}

    def snapshot(self) -> dict[str, Any]:
        return self.budget.snapshot()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def build(deck_path: str, multi_pick=None, **kwargs) -> TurnSearch | None:
    """Construct the layer, letting an A/B run retune it without a code edit.

    The environment overrides exist so the arena can measure one setting
    against another; the shipped defaults are the constants at the top of this
    module and no submission depends on the environment being set.
    """
    if not ENABLED or os.environ.get("GRIMMSNARL_TURN_SEARCH_DISABLE") == "1":
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    path = deck_path if os.path.isabs(deck_path) else os.path.join(here, deck_path)
    with open(path, encoding="utf-8") as handle:
        deck = [int(line) for line in handle.read().split() if line.strip()]
    if len(deck) != 60:
        return None
    kwargs.setdefault("authority", _env_int("GRIMMSNARL_SEARCH_AUTHORITY", 1))
    kwargs.setdefault(
        "commit_plan", _env_int("GRIMMSNARL_SEARCH_COMMIT", 1) == 1
    )
    kwargs.setdefault(
        "determinizations",
        _env_int("GRIMMSNARL_SEARCH_WORLDS", DEFAULT_DETERMINIZATIONS),
    )
    kwargs.setdefault(
        "per_decision_seconds",
        _env_float("GRIMMSNARL_SEARCH_SECONDS", DEFAULT_PER_DECISION_SECONDS),
    )
    kwargs.setdefault("max_nodes", _env_int("GRIMMSNARL_SEARCH_NODES",
                                            DEFAULT_MAX_NODES))
    return TurnSearch(deck, multi_pick=multi_pick, **kwargs)
