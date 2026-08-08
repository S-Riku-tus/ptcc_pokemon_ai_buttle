"""Robust full-turn arithmetic search for Grimmsnarl v11.

The v9 ranker remains the default policy and supplies both the root candidates
and the continuation policy.  This module uses cabt's real Search API to play
a small set of semantically distinct root actions to the end of our turn.  It
differs from v7 in the two places that made v7's value search flat:

* the leaf is the last observation from our own perspective, so retained hand
  identities and the exact action sequence are still visible;
* leaves are compared with deterministic prize, attack, board, energy and
  survival arithmetic instead of a public-state win-probability model.

Search needs a complete hidden state even though an agent cannot know deck
order or prizes.  v11 therefore evaluates two legal determinizations and only
overrides v9 when the alternative is non-worse in both and meaningfully better
in at least one.  Any tie, engine error or incomplete branch returns v9.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import json
import os
import random
from collections import Counter
from typing import Any

import fallback_policy
import ml_features as mf
import ml_planner


MAIN_CONTEXT = 0
MIN_TURN = 1
TOP_K = 3
MAX_RANK_MARGIN = 8.0
TERMINAL_RANK_MARGIN = 12.0
DETERMINIZATIONS = 2
MAX_ROLLOUT_STEPS = 64
MIN_MEAN_UTILITY_GAIN = 1_000
DEFAULT_SEARCHES_PER_TURN = 1
ALAKAZAM_SECOND_SEARCHES_PER_TURN = 2
ALAKAZAM_LINE_IDS = frozenset({
    fallback_policy.C.ABRA,
    fallback_policy.C.KADABRA,
    fallback_policy.C.ALAKAZAM,
})


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item) for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


class _SearchApi:
    """Use the official wrapper on Kaggle and the low-level local shim here."""

    def __init__(self) -> None:
        from cg import api

        self.api = api
        self.high_level = all(hasattr(api, name) for name in (
            "search_begin", "search_step", "search_end",
        ))
        self.pointer = None
        self.lib = None
        if not self.high_level:
            self._bind_low_level()

    def _bind_low_level(self) -> None:
        from cg.sim import lib

        pointer = ctypes.POINTER(ctypes.c_int)
        lib.AgentStart.restype = ctypes.c_void_p
        lib.AgentStart.argtypes = []
        lib.SearchBegin.restype = ctypes.c_char_p
        lib.SearchBegin.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            pointer, pointer, pointer, pointer, pointer, pointer,
            ctypes.c_int,
        ]
        lib.SearchStep.restype = ctypes.c_char_p
        lib.SearchStep.argtypes = [
            ctypes.c_void_p, ctypes.c_int, pointer, ctypes.c_int,
        ]
        lib.SearchEnd.restype = None
        lib.SearchEnd.argtypes = [ctypes.c_void_p]
        self.lib = lib
        self.pointer = lib.AgentStart()
        if not self.pointer:
            raise RuntimeError("AgentStart failed")

    @staticmethod
    def _state(result: Any) -> dict[str, Any]:
        if hasattr(result, "error") and int(result.error or 0) != 0:
            raise RuntimeError(f"search error {int(result.error)}")
        if isinstance(result, dict):
            if int(result.get("error", 0) or 0) != 0:
                raise RuntimeError(f"search error {result.get('error')}")
            state = result.get("state", result)
        else:
            state = getattr(result, "state", result)
        plain = _plain(state)
        if not isinstance(plain, dict) or not plain:
            raise RuntimeError("search returned no state")
        return plain

    def begin(self, observation: dict[str, Any], hidden: list[list[int]]) -> dict:
        if self.high_level:
            try:
                result = self.api.search_begin(observation, *hidden, False)
            except (AttributeError, TypeError):
                converted = self.api.to_observation_class(observation)
                result = self.api.search_begin(converted, *hidden, False)
            return self._state(result)

        arrays = [(ctypes.c_int * len(values))(*values) for values in hidden]
        encoded = str(observation["search_begin_input"]).encode("ascii")
        raw = self.lib.SearchBegin(
            self.pointer, encoded, len(encoded), *arrays, 0,
        )
        if not raw:
            raise RuntimeError("SearchBegin returned null")
        return self._state(json.loads(raw))

    def step(self, search_id: int, selection: list[int]) -> dict:
        if self.high_level:
            return self._state(self.api.search_step(search_id, selection))
        array = (ctypes.c_int * len(selection))(*selection)
        raw = self.lib.SearchStep(
            self.pointer, int(search_id), array, len(selection),
        )
        if not raw:
            raise RuntimeError("SearchStep returned null")
        return self._state(json.loads(raw))

    def end(self) -> None:
        if self.high_level:
            self.api.search_end()
        else:
            self.lib.SearchEnd(self.pointer)


def _card_ids(value: Any):
    if isinstance(value, dict):
        card_id = value.get("id")
        if isinstance(card_id, int):
            yield card_id
        for nested in value.values():
            yield from _card_ids(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _card_ids(nested)


def _repeat(values: list[int], count: int, offset: int = 0) -> list[int]:
    source = values or list(fallback_policy.MY_DECK)
    return [source[(offset + index) % len(source)] for index in range(count)]


def _hidden_state(
    observation: dict[str, Any], perspective: int, salt: int = 0,
) -> list[list[int]]:
    """Construct one deterministic, engine-legal hidden-state hypothesis."""
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    opponent = 1 - perspective
    me = players[perspective]
    them = players[opponent]
    deck = list(fallback_policy.MY_DECK)
    remaining = Counter(deck)
    for zone in ("hand", "discard", "active", "bench"):
        for card_id in _card_ids(me.get(zone) or []):
            if remaining[card_id] > 0:
                remaining[card_id] -= 1
    for card in me.get("prize") or []:
        if isinstance(card, dict):
            card_id = card.get("id")
            if isinstance(card_id, int) and remaining[card_id] > 0:
                remaining[card_id] -= 1

    pool: list[int] = []
    for card_id, count in remaining.items():
        pool.extend([card_id] * max(int(count), 0))
    seed = (
        int(current.get("turn", 0) or 0) * 1009
        + int(current.get("turnActionCount", 0) or 0) * 97
        + perspective * 17
        + salt * 104_729
    )
    random.Random(seed).shuffle(pool)
    prize_count = len(me.get("prize") or [])
    deck_count = int(me.get("deckCount", 0) or 0)
    needed = deck_count + prize_count
    if len(pool) < needed:
        pool.extend(_repeat(deck, needed - len(pool), seed % len(deck)))

    prize_values: list[int] = []
    cursor = 0
    for card in me.get("prize") or []:
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            prize_values.append(int(card["id"]))
        else:
            prize_values.append(pool[cursor])
            cursor += 1
    own_deck = pool[cursor:cursor + deck_count]
    if len(own_deck) < deck_count:
        own_deck.extend(_repeat(deck, deck_count - len(own_deck), seed))

    # We never roll through the opponent's turn.  Their hidden identities only
    # matter to effects such as Unfair Stamp; a deterministic mirror prior is
    # legal and deliberately receives no strategic value in the leaf score.
    offset = (seed + sum(_card_ids(them.get("discard") or []))) % len(deck)
    opponent_deck = _repeat(deck, int(them.get("deckCount", 0) or 0), offset)
    opponent_prize = _repeat(deck, len(them.get("prize") or []), offset + 11)
    opponent_hand = _repeat(deck, int(them.get("handCount", 0) or 0), offset + 23)
    active = them.get("active") or []
    opponent_active = [mf.IMPIDIMP_ID] if active and active[0] is None else []
    return [
        own_deck, prize_values, opponent_deck, opponent_prize,
        opponent_hand, opponent_active,
    ]


def _simple_legal(select: dict[str, Any]) -> list[int]:
    count = len(select.get("option") or [])
    maximum = min(int(select.get("maxCount", 0) or 0), count)
    minimum = min(int(select.get("minCount", 0) or 0), maximum)
    return list(range(max(minimum, maximum)))


def _players(
    observation: dict[str, Any], perspective: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    if len(players) < 2:
        return {}, {}
    return players[perspective], players[1 - perspective]


def _damage_by_serial(player: dict[str, Any]) -> dict[int, float]:
    output: dict[int, float] = {}
    for card in mf._in_play(player):
        serial = int(card.get("serial", -1))
        if serial >= 0:
            output[serial] = max(
                0.0,
                float(card.get("maxHp", 0)) - float(card.get("hp", 0)),
            )
    return output


def _line_progress(card: dict[str, Any]) -> int:
    card_id = int(card.get("id", -1))
    return {
        mf.GRIMMSNARL_EX_ID: 5,
        mf.MORGREM_ID: 3,
        mf.IMPIDIMP_ID: 1,
        mf.FROSLASS_ID: 2,
        mf.SNORUNT_ID: 1,
        mf.MUNKIDORI_ID: 2,
    }.get(card_id, 0)


def _useful_energy(card: dict[str, Any]) -> tuple[int, int]:
    card_id = int(card.get("id", -1))
    energy = mf._dark_energy_count(card)
    if card_id in mf.MARNIE_LINE_IDS:
        return min(2, energy), max(0, energy - 2)
    if card_id == mf.MUNKIDORI_ID:
        return min(1, energy), max(0, energy - 1)
    return 0, energy


def _hand_plan(hand: list[dict[str, Any]], me: dict[str, Any]) -> int:
    ids = Counter(int(card.get("id", -1)) for card in hand)
    score = min(8, len(hand))
    score += 2 * min(2, ids[mf.DARK_ENERGY_ID])
    score += 2 * ids[mf.RARE_CANDY_ID]
    score += 2 * ids[mf.GRIMMSNARL_EX_ID]
    score += ids[mf.MORGREM_ID]
    score += 2 * ids[mf.POFFIN_ID]
    score += 2 * ids[mf.NIGHT_STRETCHER_ID]
    score += 2 * ids[mf.PETREL_ID]
    score += 3 * (ids[mf.LILLIE_ID] + ids[mf.DAWN_ID])
    score += 2 * ids[mf.BOSS_ID]
    score += 2 * ids[mf.UNFAIR_STAMP_ID]
    impidimps = sum(
        int(int(card.get("id", -1)) == mf.IMPIDIMP_ID)
        for card in mf._in_play(me)
    )
    if impidimps and ids[mf.RARE_CANDY_ID] and ids[mf.GRIMMSNARL_EX_ID]:
        score += 8
    return int(score)


@dataclasses.dataclass(frozen=True)
class LeafEvaluation:
    result: int
    prizes_taken: int
    prizes_conceded: int
    attacked: int
    damage_dealt: int
    active_ready: int
    ready_grimms: int
    bodies: int
    setup_progress: int
    useful_energy: int
    wasted_energy: int
    active_survival_margin: int
    hand_plan: int
    hand_count: int
    deck_count: int
    own_damage_added: int

    @property
    def utility(self) -> int:
        """A tie-break scale; acceptance is governed by `_grade_upgrade`."""
        body_safety = min(self.bodies, 3)
        return int(
            self.result * 100_000_000
            + self.prizes_taken * 1_000_000
            - self.prizes_conceded * 1_000_000
            + self.attacked * 50_000
            + self.damage_dealt * 100
            + self.active_ready * 20_000
            + self.ready_grimms * 8_000
            + body_safety * 2_000
            + self.setup_progress * 1_000
            + self.useful_energy * 800
            - self.wasted_energy * 400
            + self.active_survival_margin * 10
            + self.hand_plan * 250
            + self.hand_count * 50
            + self.deck_count * 15
            - self.own_damage_added * 50
        )

    def summary(self) -> dict[str, int]:
        output = dataclasses.asdict(self)
        output["utility"] = self.utility
        return output


def evaluate_leaf(
    root_observation: dict[str, Any],
    preterminal_observation: dict[str, Any],
    post_observation: dict[str, Any],
    perspective: int,
    terminal_action: str,
) -> LeafEvaluation:
    """Score a completed own turn from exact, observable arithmetic."""
    root_current = root_observation.get("current") or {}
    post_current = post_observation.get("current") or {}
    root_me, root_opp = _players(root_observation, perspective)
    post_me, post_opp = _players(post_observation, perspective)
    pre_me, _ = _players(preterminal_observation, perspective)

    game_result = int(post_current.get("result", -1))
    result = (
        1 if game_result == perspective
        else -1 if game_result in (0, 1)
        else 0
    )
    prizes_taken = max(
        0,
        len(root_me.get("prize") or []) - len(post_me.get("prize") or []),
    )
    prizes_conceded = max(
        0,
        len(root_opp.get("prize") or []) - len(post_opp.get("prize") or []),
    )

    before_opp = _damage_by_serial(root_opp)
    after_opp = _damage_by_serial(post_opp)
    damage_dealt = sum(
        max(0.0, damage - before_opp.get(serial, 0.0))
        for serial, damage in after_opp.items()
    )
    before_me = _damage_by_serial(root_me)
    after_me = _damage_by_serial(post_me)
    own_damage_added = sum(
        max(0.0, damage - before_me.get(serial, 0.0))
        for serial, damage in after_me.items()
    )

    bodies = mf._in_play(post_me)
    active = (mf._cards(post_me, "active") or [{}])[0]
    ready_grimms = sum(
        int(
            int(card.get("id", -1)) == mf.GRIMMSNARL_EX_ID
            and mf._dark_energy_count(card) >= mf.SHADOW_BULLET_COST
        )
        for card in bodies
    )
    active_ready = int(
        int(active.get("id", -1)) == mf.GRIMMSNARL_EX_ID
        and mf._dark_energy_count(active) >= mf.SHADOW_BULLET_COST
    )
    setup_progress = sum(_line_progress(card) for card in bodies)
    useful_energy = 0
    wasted_energy = 0
    for card in bodies:
        useful, wasted = _useful_energy(card)
        useful_energy += useful
        wasted_energy += wasted

    if int(active.get("id", -1)) < 0:
        survival = -300
    else:
        threat = mf.incoming_damage(mf._in_play(post_opp), active, 1)
        survival = int(max(
            -300.0,
            min(300.0, float(active.get("hp", 0)) - threat),
        ))

    hand = [
        card for card in (pre_me.get("hand") or [])
        if isinstance(card, dict)
    ]
    return LeafEvaluation(
        result=result,
        prizes_taken=int(prizes_taken),
        prizes_conceded=int(prizes_conceded),
        attacked=int(terminal_action == "attack"),
        damage_dealt=int(round(damage_dealt)),
        active_ready=active_ready,
        ready_grimms=ready_grimms,
        bodies=len(bodies),
        setup_progress=setup_progress,
        useful_energy=useful_energy,
        wasted_energy=wasted_energy,
        active_survival_margin=survival,
        hand_plan=_hand_plan(hand, post_me),
        hand_count=int(pre_me.get("handCount", len(hand)) or len(hand)),
        deck_count=int(post_me.get("deckCount", 0) or 0),
        own_damage_added=int(round(own_damage_added)),
    )


def _grade_upgrade(candidate: LeafEvaluation, incumbent: LeafEvaluation) -> int:
    """Return -3..3; positive means a proof-like, non-regressive upgrade."""
    if candidate.result != incumbent.result:
        return 3 if candidate.result > incumbent.result else -3
    if candidate.prizes_taken != incumbent.prizes_taken:
        if (
            candidate.prizes_taken > incumbent.prizes_taken
            and candidate.prizes_conceded <= incumbent.prizes_conceded
        ):
            return 3
        if candidate.prizes_taken < incumbent.prizes_taken:
            return -3
    if candidate.prizes_conceded != incumbent.prizes_conceded:
        if (
            candidate.prizes_conceded < incumbent.prizes_conceded
            and candidate.prizes_taken >= incumbent.prizes_taken
        ):
            return 3
        if candidate.prizes_conceded > incumbent.prizes_conceded:
            return -3

    # Without an immediate prize gain, none of these strategic invariants may
    # be sold for a softer setup or hand preference.
    if candidate.attacked < incumbent.attacked:
        return -2
    if candidate.active_ready < incumbent.active_ready:
        return -2
    if candidate.ready_grimms < incumbent.ready_grimms:
        return -2
    if candidate.bodies < incumbent.bodies:
        return -2 if candidate.bodies <= 1 else -1
    if candidate.useful_energy + 1 < incumbent.useful_energy:
        return -1

    major = 0
    medium = 0
    structural = 0
    major += int(candidate.attacked > incumbent.attacked)
    major += int(candidate.active_ready > incumbent.active_ready)
    major += int(candidate.ready_grimms > incumbent.ready_grimms)
    major += int(candidate.bodies > incumbent.bodies and incumbent.bodies <= 2)
    structural += major

    setup_gain = candidate.setup_progress - incumbent.setup_progress
    energy_gain = candidate.useful_energy - incumbent.useful_energy
    damage_gain = candidate.damage_dealt - incumbent.damage_dealt
    survival_gain = (
        candidate.active_survival_margin - incumbent.active_survival_margin
    )
    hand_gain = candidate.hand_plan - incumbent.hand_plan
    medium += int(setup_gain >= 2)
    medium += int(energy_gain >= 1 and candidate.wasted_energy <= incumbent.wasted_energy)
    medium += int(damage_gain >= 30)
    medium += int(survival_gain >= 40)
    medium += int(hand_gain >= 3 and candidate.hand_count >= incumbent.hand_count)
    structural += int(setup_gain >= 2) + int(energy_gain >= 1)
    structural += int(damage_gain >= 30) + int(survival_gain >= 40)

    delta = candidate.utility - incumbent.utility
    if delta <= 0:
        return 0
    if major:
        return 2
    if structural and medium >= 1 and delta >= MIN_MEAN_UTILITY_GAIN:
        return 1

    # A strictly better retained plan is allowed only when every public board
    # resource is equal or better.  This is the sequence-sensitive case the
    # public-state v7 head could not distinguish.
    resource_upgrade = (
        hand_gain >= 4
        and candidate.hand_count >= incumbent.hand_count
        and candidate.deck_count >= incumbent.deck_count
        and candidate.wasted_energy <= incumbent.wasted_energy
        and candidate.setup_progress >= incumbent.setup_progress
    )
    return 1 if resource_upgrade and delta >= MIN_MEAN_UTILITY_GAIN else 0


class ArithmeticSearch:
    def __init__(self) -> None:
        self.disabled = (
            os.environ.get("GRIMMSNARL_ARITHMETIC_SEARCH_DISABLE") == "1"
        )
        self.search = None if self.disabled else _SearchApi()
        self.reset()

    def reset(self) -> None:
        self._turn_search_counts: dict[tuple[int, int], int] = {}
        self._last_turn: int | None = None
        self._last_turn_action_count = -1
        self._override_records: list[dict[str, Any]] = []
        self.stats: dict[str, int | float] = {
            "considered": 0,
            "searched": 0,
            "determinizations": 0,
            "branches": 0,
            "overrides": 0,
            "branch_errors": 0,
            "incomplete_branches": 0,
            "skip_early": 0,
            "skip_already_searched_turn": 0,
            "alakazam_second_considered": 0,
            "alakazam_second_extra_searches": 0,
            "skip_planner_guard": 0,
            "skip_no_search_state": 0,
            "skip_no_candidates": 0,
            "skip_incomplete_matrix": 0,
            "skip_nonrobust": 0,
            "robust_candidates": 0,
            "utility_gain_sum": 0.0,
            "utility_gain_max": 0.0,
        }

    @staticmethod
    def _alakazam_going_second(observation: dict[str, Any]) -> bool:
        """Whether the narrow ladder failure mode is currently observable.

        The trigger is deliberately based only on public board cards and the
        resolved first-player field.  It cannot infer a hidden matchup before
        Abra, Kadabra or Alakazam appears, and it stands down while
        ``firstPlayer`` is still unresolved.
        """
        current = observation.get("current") or {}
        players = current.get("players") or []
        perspective = int(current.get("yourIndex", -1))
        first_player = int(current.get("firstPlayer", -1))
        if (
            len(players) < 2
            or perspective not in (0, 1)
            or first_player not in (0, 1)
            or first_player == perspective
        ):
            return False
        opponent = players[1 - perspective]
        visible_ids = {
            int(card.get("id", -1))
            for card in mf._in_play(opponent)
            if isinstance(card, dict)
        }
        return bool(visible_ids & ALAKAZAM_LINE_IDS)

    @staticmethod
    def _terminal_action(
        observation: dict[str, Any], selection: list[int],
    ) -> str:
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        if (
            int(select.get("context", -1)) == MAIN_CONTEXT
            and len(selection) == 1
            and 0 <= selection[0] < len(options)
        ):
            action = mf.action_type(
                observation.get("current") or {},
                options[selection[0]],
                select,
            )
            if action in ("attack", "end"):
                return action
        return ""

    def _branch_choice(
        self,
        observation: dict[str, Any],
        ranker,
        rule_agent,
        planner: ml_planner.Planner,
    ) -> list[int]:
        select = observation.get("select") or {}
        try:
            rule_choice = list(rule_agent(observation))
        except Exception:
            rule_choice = _simple_legal(select)
        if ranker.is_scorable(select):
            index = ranker.choose(observation)
            if index is not None:
                index = planner.adjust(
                    observation, select, index, ranker.last_scores,
                )
                ranker.commit(index)
                planner.note(observation, select, index)
                return [index]
        chosen = rule_choice[0] if len(rule_choice) == 1 else None
        if chosen is not None:
            ranker.observe_external(observation, chosen)
            planner.note(observation, select, chosen)
        return rule_choice

    def _rollout(
        self,
        root_observation: dict[str, Any],
        root_id: int,
        candidate: int,
        root_turn: int,
        perspective: int,
        ranker,
        root_ranker_state: dict[str, Any],
        live_planner: ml_planner.Planner | None,
    ) -> LeafEvaluation | None:
        ranker.restore_dynamic_state(root_ranker_state)
        ranker.teacher_forced = False
        ranker.commit(candidate)
        branch_planner = (
            copy.deepcopy(live_planner)
            if live_planner is not None else ml_planner.Planner()
        )
        branch_planner.note(
            root_observation,
            root_observation.get("select") or {},
            candidate,
        )
        branch_diag = fallback_policy._fresh_diag()
        rule_agent = fallback_policy.make_agent(
            fallback_policy.GrimmsnarlPolicy,
            fallback_policy.MY_DECK,
            branch_diag,
        )

        root_selection = [candidate]
        terminal_action = self._terminal_action(
            root_observation, root_selection,
        )
        preterminal = root_observation
        state = self.search.step(root_id, root_selection)
        for _ in range(MAX_ROLLOUT_STEPS + 1):
            observation = state.get("observation") or {}
            current = observation.get("current") or {}
            result = int(current.get("result", -1))
            turn_finished = (
                int(current.get("turn", -1)) != root_turn
                or int(current.get("yourIndex", -1)) != perspective
            )
            if result >= 0 or turn_finished:
                return evaluate_leaf(
                    root_observation,
                    preterminal,
                    observation,
                    perspective,
                    terminal_action,
                )
            select = observation.get("select") or {}
            if not select:
                return None
            selection = self._branch_choice(
                observation, ranker, rule_agent, branch_planner,
            )
            preterminal = observation
            next_action = self._terminal_action(observation, selection)
            if next_action:
                # A KO can be followed by prize selection before perspective
                # flips.  Do not erase the attack when that non-MAIN select is
                # processed.
                terminal_action = next_action
            state = self.search.step(int(state["searchId"]), selection)
        return None

    @staticmethod
    def _candidate_indices(
        observation: dict[str, Any],
        proposed: int,
        scores: dict[int, float],
    ) -> list[int]:
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        if proposed not in scores or proposed >= len(options):
            return []
        ordered = sorted(scores, key=lambda index: scores[index], reverse=True)
        top_score = scores[ordered[0]]
        candidates = [proposed]
        seen_actions = {
            mf.action_type(
                observation.get("current") or {}, options[proposed], select,
            )
        }

        # First cover distinct semantic action classes; then fill remaining
        # slots by ranker score.  This prevents four interchangeable Items from
        # crowding out an Energy, evolution or attack line.
        for index in ordered:
            if index == proposed or not 0 <= index < len(options):
                continue
            action = mf.action_type(
                observation.get("current") or {}, options[index], select,
            )
            margin_limit = (
                TERMINAL_RANK_MARGIN
                if action in ("attack", "end") else MAX_RANK_MARGIN
            )
            if top_score - scores[index] > margin_limit:
                continue
            if action in seen_actions:
                continue
            candidates.append(index)
            seen_actions.add(action)
            if len(candidates) >= TOP_K:
                return candidates
        for index in ordered:
            if index in candidates or not 0 <= index < len(options):
                continue
            if top_score - scores[index] <= MAX_RANK_MARGIN:
                candidates.append(index)
            if len(candidates) >= TOP_K:
                break
        return candidates

    def adjust(
        self,
        observation: dict[str, Any],
        proposed: int,
        scores: dict[int, float],
        ranker,
        planner: ml_planner.Planner | None = None,
    ) -> int:
        if self.disabled or self.search is None or len(scores) < 2:
            return proposed
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != MAIN_CONTEXT:
            return proposed
        self.stats["considered"] += 1
        current = observation.get("current") or {}
        turn = int(current.get("turn", 0) or 0)
        action_count = int(current.get("turnActionCount", 0) or 0)
        perspective = int(current.get("yourIndex", 0) or 0)
        new_game = (
            self._last_turn is not None
            and (
                turn < self._last_turn
                or (
                    turn == self._last_turn
                    and turn <= 2
                    and action_count < self._last_turn_action_count
                )
            )
        )
        if new_game:
            self._turn_search_counts.clear()
        self._last_turn = turn
        self._last_turn_action_count = action_count
        turn_key = (turn, perspective)
        alakazam_second = self._alakazam_going_second(observation)
        if alakazam_second:
            self.stats["alakazam_second_considered"] += 1
        search_limit = (
            ALAKAZAM_SECOND_SEARCHES_PER_TURN
            if alakazam_second else DEFAULT_SEARCHES_PER_TURN
        )
        searches_this_turn = self._turn_search_counts.get(turn_key, 0)
        if searches_this_turn >= search_limit:
            self.stats["skip_already_searched_turn"] += 1
            return proposed
        if turn < MIN_TURN:
            self.stats["skip_early"] += 1
            return proposed
        ordered = sorted(scores, key=lambda index: scores[index], reverse=True)
        if proposed != ordered[0]:
            self.stats["skip_planner_guard"] += 1
            return proposed
        if not observation.get("search_begin_input"):
            self.stats["skip_no_search_state"] += 1
            return proposed
        candidates = self._candidate_indices(observation, proposed, scores)
        if len(candidates) < 2:
            self.stats["skip_no_candidates"] += 1
            return proposed

        ranker_state = ranker.save_dynamic_state()
        diag_state = copy.deepcopy(fallback_policy.DIAG)
        immunity_state = copy.deepcopy(fallback_policy.TEMP_IMMUNITY)
        values: dict[int, list[LeafEvaluation]] = {
            candidate: [] for candidate in candidates
        }
        try:
            self._turn_search_counts[turn_key] = searches_this_turn + 1
            self.stats["searched"] += 1
            if alakazam_second and searches_this_turn > 0:
                self.stats["alakazam_second_extra_searches"] += 1

            # First determinization screens every candidate.  The second runs
            # only v9 and alternatives that were already strict upgrades.  A
            # candidate tied or worse in sample zero cannot pass the final
            # strict-consensus rule, so simulating it again only burns clock.
            active_candidates = list(candidates)
            for salt in range(DETERMINIZATIONS):
                started = False
                try:
                    hidden = _hidden_state(observation, perspective, salt)
                    root = self.search.begin(observation, hidden)
                    started = True
                    self.stats["determinizations"] += 1
                    root_id = int(root["searchId"])
                    for candidate in active_candidates:
                        self.stats["branches"] += 1
                        try:
                            leaf = self._rollout(
                                observation,
                                root_id,
                                candidate,
                                turn,
                                perspective,
                                ranker,
                                ranker_state,
                                planner,
                            )
                            if leaf is None:
                                self.stats["incomplete_branches"] += 1
                            else:
                                values[candidate].append(leaf)
                        except Exception:
                            self.stats["branch_errors"] += 1
                except Exception:
                    self.stats["branch_errors"] += 1
                finally:
                    if started:
                        try:
                            self.search.end()
                        except Exception:
                            self.stats["branch_errors"] += 1

                if salt == 0:
                    if len(values[proposed]) != 1:
                        break
                    first_incumbent = values[proposed][0]
                    survivors = [
                        candidate for candidate in candidates
                        if candidate != proposed
                        and len(values[candidate]) == 1
                        and _grade_upgrade(
                            values[candidate][0], first_incumbent,
                        ) > 0
                    ]
                    if not survivors:
                        break
                    active_candidates = [proposed, *survivors]
        finally:
            ranker.restore_dynamic_state(ranker_state)
            fallback_policy.DIAG.clear()
            fallback_policy.DIAG.update(diag_state)
            fallback_policy.TEMP_IMMUNITY.clear()
            fallback_policy.TEMP_IMMUNITY.update(immunity_state)

        if len(values[proposed]) == 0:
            self.stats["skip_incomplete_matrix"] += 1
            return proposed
        screened = [
            candidate for candidate in candidates
            if candidate != proposed and len(values[candidate]) == DETERMINIZATIONS
        ]
        if len(values[proposed]) != DETERMINIZATIONS:
            # No strict first-sample upgrade is a normal conservative skip;
            # only count an incomplete matrix if a surviving branch failed.
            if any(len(values[candidate]) == 1 for candidate in candidates
                   if candidate != proposed):
                self.stats["skip_nonrobust"] += 1
            else:
                self.stats["skip_incomplete_matrix"] += 1
            return proposed
        incumbent = values[proposed]
        robust: list[tuple[int, list[int], list[int]]] = []
        for candidate in screened:
            grades = [
                _grade_upgrade(candidate_leaf, incumbent_leaf)
                for candidate_leaf, incumbent_leaf
                in zip(values[candidate], incumbent)
            ]
            deltas = [
                candidate_leaf.utility - incumbent_leaf.utility
                for candidate_leaf, incumbent_leaf
                in zip(values[candidate], incumbent)
            ]
            if (
                min(grades) > 0
                and sum(deltas) / len(deltas) >= MIN_MEAN_UTILITY_GAIN
            ):
                robust.append((candidate, grades, deltas))
        if not robust:
            self.stats["skip_nonrobust"] += 1
            return proposed

        self.stats["robust_candidates"] += len(robust)
        best, grades, deltas = max(
            robust,
            key=lambda item: (
                min(item[1]),
                sum(item[1]),
                min(item[2]),
                sum(item[2]),
                scores.get(item[0], float("-inf")),
            ),
        )
        mean_gain = sum(deltas) / len(deltas)
        self.stats["overrides"] += 1
        self.stats["utility_gain_sum"] += mean_gain
        self.stats["utility_gain_max"] = max(
            float(self.stats["utility_gain_max"]), mean_gain,
        )
        pending = ranker_state.get("_pending") or []
        before = pending[proposed] if proposed < len(pending) else {}
        after = pending[best] if best < len(pending) else {}
        if len(self._override_records) < 32:
            self._override_records.append({
                "turn": turn,
                "from_index": proposed,
                "to_index": best,
                "from_action_type": int(before.get("action_type_id", -1)),
                "to_action_type": int(after.get("action_type_id", -1)),
                "from_card_id": int(before.get("candidate_card_id", -1)),
                "to_card_id": int(after.get("candidate_card_id", -1)),
                "grades": grades,
                "utility_deltas": deltas,
                "mean_utility_gain": round(mean_gain, 2),
                "incumbent": [leaf.summary() for leaf in incumbent],
                "candidate": [leaf.summary() for leaf in values[best]],
            })
        return best

    def snapshot(self) -> dict[str, Any]:
        output = dict(self.stats)
        for name in ("utility_gain_sum", "utility_gain_max"):
            output[name] = round(float(output[name]), 2)
        output.update({
            "enabled": not self.disabled,
            "min_turn": MIN_TURN,
            "top_k": TOP_K,
            "determinizations_per_search": DETERMINIZATIONS,
            "max_rank_margin": MAX_RANK_MARGIN,
            "min_mean_utility_gain": MIN_MEAN_UTILITY_GAIN,
            "default_searches_per_turn": DEFAULT_SEARCHES_PER_TURN,
            "alakazam_second_searches_per_turn": (
                ALAKAZAM_SECOND_SEARCHES_PER_TURN
            ),
            "override_records": list(self._override_records),
        })
        return output
