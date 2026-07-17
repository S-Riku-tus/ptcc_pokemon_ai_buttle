"""Pure, engine-free core for Champion-Challenger evaluation.

This module holds everything that can be tested without importing the cg
battle engine: configuration handling, fair matchup generation, per-game and
per-matchup metric aggregation, statistical confidence intervals, promotion
judgement, challenger auto-detection, and report rendering.

The battle-running orchestrator (``scripts/run_champion_challenger.py``) feeds
this module lightweight decision events + game metadata; the promotion CLI
(``scripts/promote_challenger.py``) reuses the report readers.  Keeping this
layer free of ``cg`` imports means the whole judgement pipeline is unit
testable with mock agents.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

CHAMPION_ROLE = "champion"
CHALLENGER_ROLE = "challenger"

VERDICT_PROMOTE = "PROMOTE_RECOMMENDED"
VERDICT_HOLD = "HOLD"
VERDICT_REJECT = "REJECT"
VERDICT_INVALID = "INVALID_EVALUATION"

# Card id of the archetype's key attacker (Alakazam).  Configurable per config.
DEFAULT_ALAKAZAM_CARD_ID = 743

DEFAULT_CONFIG: dict[str, Any] = {
    "archetype": "alakazam",
    "champion_agent": None,
    "challenger_agent": None,
    "baseline_agent": None,
    "games": 200,
    "seat_swap": True,
    "seed": 0,
    "max_steps": 8000,
    "parallel_workers": 1,
    "timeout_seconds_per_decision": 5.0,
    "save_replays": False,
    "save_trajectories": False,
    "run_baseline_comparison": False,
    "baseline_games": 100,
    "minimum_games": 1,
    "require_confidence_interval_above": 0.50,
    "alakazam_card_id": DEFAULT_ALAKAZAM_CARD_ID,
    "output_root": "artifacts/champion_challenger",
    "promotion_thresholds": {
        "minimum_head_to_head_win_rate": 0.53,
        "minimum_attack_turn_rate": 0.70,
        "minimum_alakazam_attacks_per_game": 3.8,
        "maximum_deckout_rate": 0.05,
        "maximum_boardout_rate": 0.05,
        "maximum_post_first_attack_idle_turns_in_losses": 0.8,
        "maximum_crashes": 0,
        "maximum_illegal_actions": 0,
        "maximum_timeouts": 0,
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: Path | None) -> dict[str, Any]:
    """Load a config file merged over the defaults (deep merge for thresholds)."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in raw.items():
            if key == "promotion_thresholds" and isinstance(value, dict):
                config["promotion_thresholds"].update(value)
            else:
                config[key] = value
    return config


# CLI overrides that map directly onto config keys.
_OVERRIDE_KEYS = {
    "champion": "champion_agent",
    "challenger": "challenger_agent",
    "baseline": "baseline_agent",
    "games": "games",
    "seed": "seed",
    "seat_swap": "seat_swap",
    "parallel_workers": "parallel_workers",
    "timeout_seconds_per_decision": "timeout_seconds_per_decision",
    "save_replays": "save_replays",
    "save_trajectories": "save_trajectories",
    "run_baseline_comparison": "run_baseline_comparison",
    "baseline_games": "baseline_games",
    "output_root": "output_root",
    "max_steps": "max_steps",
    "minimum_games": "minimum_games",
}


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Overlay non-None CLI overrides (keyed by CLI name) onto a config dict."""
    merged = json.loads(json.dumps(config))
    for cli_name, value in overrides.items():
        if value is None:
            continue
        config_key = _OVERRIDE_KEYS.get(cli_name, cli_name)
        merged[config_key] = value
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Raise ValueError on structurally invalid configuration."""
    if not config.get("champion_agent"):
        raise ValueError("champion_agent is required (config or --champion)")
    if not config.get("challenger_agent"):
        raise ValueError("challenger_agent is required (config or --challenger)")
    if config["champion_agent"] == config["challenger_agent"]:
        raise ValueError(
            "champion and challenger must differ; refusing to compare an agent to itself"
        )
    games = config.get("games")
    if not isinstance(games, int) or games < 1:
        raise ValueError(f"games must be a positive integer, got {games!r}")
    if config.get("seat_swap") and games % 2 != 0:
        raise ValueError(
            f"seat_swap requires an even game count so every seed plays both seats; got {games}"
        )
    thresholds = config.get("promotion_thresholds") or {}
    for key, value in thresholds.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"threshold {key!r} must be numeric, got {value!r}")
        if key.startswith("minimum_") or key.startswith("maximum_"):
            if value < 0:
                raise ValueError(f"threshold {key!r} must be non-negative, got {value}")
        if "win_rate" in key or "rate" in key:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"rate threshold {key!r} must be within [0,1], got {value}")
    ci = config.get("require_confidence_interval_above")
    if ci is not None and not 0.0 <= float(ci) <= 1.0:
        raise ValueError(f"require_confidence_interval_above must be within [0,1], got {ci}")


# ---------------------------------------------------------------------------
# Agent resolution / hashing / detection
# ---------------------------------------------------------------------------


def resolve_agent_dir(spec: str) -> Path:
    """Resolve an agent spec to its directory, searching agents/ then archive."""
    direct = Path(spec)
    if direct.is_dir():
        return direct.resolve()
    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        candidate = base / spec
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve agent {spec!r}; checked agents/ and archive/agents/."
    )


def read_metadata(agent_dir: Path) -> dict[str, Any]:
    path = agent_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def deck_hash(agent_dir: Path) -> str:
    """Order-independent hash of the 60-card deck (matches self_play semantics)."""
    deck_path = agent_dir / "deck.csv"
    if not deck_path.exists():
        return ""
    deck = [int(x) for x in deck_path.read_text(encoding="utf-8-sig").split()]
    text = ",".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


def model_hash(agent_dir: Path) -> str:
    """Hash of the ranker model file, when the agent ships one."""
    return sha256_file(agent_dir / "ranker_model.json")[:16]


_CHALLENGER_NAME_PATTERNS = (re.compile(r"_challenger$"), re.compile(r"_candidate$"))


def detect_challengers(
    champion: str,
    archetype: str | None = None,
    agents_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return candidate challenger agents, best-guess ordered.

    A directory is a candidate when its name matches ``*_challenger`` /
    ``*_candidate`` OR its metadata ``role`` is ``challenger``.  The champion
    itself is always excluded.  Ordering is by metadata ``created_at`` when
    available (newest first), then by directory modification time.
    """
    agents_root = agents_root or (ROOT / "agents")
    candidates: list[dict[str, Any]] = []
    if not agents_root.is_dir():
        return candidates
    for child in sorted(agents_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if child.name == champion:
            continue
        if not (child / "main.py").exists():
            continue
        metadata = read_metadata(child)
        role = str(metadata.get("role", "")).lower()
        name_match = any(pat.search(child.name) for pat in _CHALLENGER_NAME_PATTERNS)
        if not name_match and role != CHALLENGER_ROLE:
            continue
        if archetype and metadata.get("archetype") and metadata["archetype"] != archetype:
            continue
        candidates.append(
            {
                "name": child.name,
                "role": role or None,
                "archetype": metadata.get("archetype"),
                "parent_agent": metadata.get("parent_agent"),
                "created_at": metadata.get("created_at"),
                "model_version": metadata.get("model_version"),
                "mtime": child.stat().st_mtime,
            }
        )
    candidates.sort(
        key=lambda c: (c.get("created_at") or "", c["mtime"]),
        reverse=True,
    )
    return candidates


# ---------------------------------------------------------------------------
# Fair matchup generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchAssignment:
    game_index: int  # 1-based
    pair_id: int  # groups the two seat-swapped games of one seed
    seed: int  # per-pair logical seed (see note below)
    seat0_role: str  # role occupying seat 0
    seat1_role: str  # role occupying seat 1


def build_pairs(
    games: int,
    base_seed: int,
    seat_swap: bool,
    role_a: str = CHALLENGER_ROLE,
    role_b: str = CHAMPION_ROLE,
) -> list[MatchAssignment]:
    """Generate the game schedule.

    With ``seat_swap`` and an even ``games`` count, each of ``games // 2`` seed
    pairs plays two games: one with ``role_a`` in seat 0 and one with the seats
    swapped.  This guarantees both agents occupy each seat exactly the same
    number of times.

    NOTE: the bundled cg engine's ``BattleStart`` accepts no RNG seed, so the
    per-pair ``seed`` here is a *logical* identifier used for deterministic
    scheduling and for seeding Python-level baselines (random/first).  It does
    not force identical shuffles across the two games of a pair; seat swap is
    what keeps the comparison fair.  This limitation is stated in the report.
    """
    assignments: list[MatchAssignment] = []
    if seat_swap:
        pair_count = games // 2
        for pair in range(pair_count):
            seed = base_seed + pair
            assignments.append(
                MatchAssignment(
                    game_index=2 * pair + 1,
                    pair_id=pair,
                    seed=seed,
                    seat0_role=role_a,
                    seat1_role=role_b,
                )
            )
            assignments.append(
                MatchAssignment(
                    game_index=2 * pair + 2,
                    pair_id=pair,
                    seed=seed,
                    seat0_role=role_b,
                    seat1_role=role_a,
                )
            )
        # Odd remainder game (only when games is odd and seat_swap slipped past
        # validation, e.g. programmatic callers): play one unpaired game.
        if games % 2 == 1:
            assignments.append(
                MatchAssignment(
                    game_index=games,
                    pair_id=pair_count,
                    seed=base_seed + pair_count,
                    seat0_role=role_a,
                    seat1_role=role_b,
                )
            )
    else:
        for game in range(games):
            a_first = game % 2 == 0
            assignments.append(
                MatchAssignment(
                    game_index=game + 1,
                    pair_id=game,
                    seed=base_seed + game,
                    seat0_role=role_a if a_first else role_b,
                    seat1_role=role_b if a_first else role_a,
                )
            )
    return assignments


# ---------------------------------------------------------------------------
# Per-game metric derivation (from decision events)
# ---------------------------------------------------------------------------


def _empty_role_metrics() -> dict[str, Any]:
    return {
        "decisions": 0,
        "attacks": 0,
        "alakazam_attacks": 0,
        "search_uses": 0,
        "first_attack_turn": None,
        "acting_turns": set(),
        "attack_turns": set(),
        "hand_size_samples": [],
        "hand_at_alakazam_attack": [],
        "overkill_total": 0.0,
        "overkill_samples": 0,
        "decision_ms_total": 0.0,
        "decision_ms_max": 0.0,
        "action_type_counts": {},
    }


def summarize_game(events: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Reduce a game's per-decision events + terminal metadata to a record.

    ``events`` items carry: role, seat, turn, action_type, is_attack,
    is_alakazam_attack, is_search, hand_count, overkill, decision_ms.
    ``meta`` carries: pair_id, game_index, seed, champion_seat, challenger_seat,
    first_player_role, winner_role, result, termination, turns, and per-role
    terminal fields (loss_reason, crash, illegal, timeout, final_deck_count,
    in_play_count_final, ml diag).
    """
    roles = {CHAMPION_ROLE: _empty_role_metrics(), CHALLENGER_ROLE: _empty_role_metrics()}
    for event in events:
        role = event.get("role")
        acc = roles.get(role)
        if acc is None:
            continue
        turn = int(event.get("turn", 0))
        acc["decisions"] += 1
        acc["acting_turns"].add(turn)
        acc["decision_ms_total"] += float(event.get("decision_ms", 0.0))
        acc["decision_ms_max"] = max(acc["decision_ms_max"], float(event.get("decision_ms", 0.0)))
        action = str(event.get("action_type") or "other")
        acc["action_type_counts"][action] = acc["action_type_counts"].get(action, 0) + 1
        acc["hand_size_samples"].append(int(event.get("hand_count", 0)))
        if event.get("is_search"):
            acc["search_uses"] += 1
        if event.get("is_attack"):
            acc["attacks"] += 1
            acc["attack_turns"].add(turn)
            if acc["first_attack_turn"] is None or turn < acc["first_attack_turn"]:
                acc["first_attack_turn"] = turn
            if event.get("is_alakazam_attack"):
                acc["alakazam_attacks"] += 1
                acc["hand_at_alakazam_attack"].append(int(event.get("hand_count", 0)))
            overkill = float(event.get("overkill", 0.0) or 0.0)
            if overkill > 0:
                acc["overkill_total"] += overkill
                acc["overkill_samples"] += 1

    per_role: dict[str, Any] = {}
    for role, acc in roles.items():
        acting = acc["acting_turns"]
        attack_turns = acc["attack_turns"]
        first = acc["first_attack_turn"]
        idle_after_first = 0
        if first is not None:
            after = {t for t in acting if t >= first}
            idle_after_first = len(after - attack_turns)
        role_meta = meta.get(role, {}) if isinstance(meta.get(role), dict) else {}
        per_role[role] = {
            "decisions": acc["decisions"],
            "attacks": acc["attacks"],
            "alakazam_attacks": acc["alakazam_attacks"],
            "search_uses": acc["search_uses"],
            "first_attack_turn": first,
            "acting_turns": len(acting),
            "attack_turns": len(attack_turns),
            "idle_turns_after_first_attack": idle_after_first,
            "hand_size_samples": acc["hand_size_samples"],
            "hand_at_alakazam_attack": acc["hand_at_alakazam_attack"],
            "overkill_total": acc["overkill_total"],
            "overkill_samples": acc["overkill_samples"],
            "decision_ms_total": acc["decision_ms_total"],
            "decision_ms_max": acc["decision_ms_max"],
            "action_type_counts": acc["action_type_counts"],
            # terminal / safety fields provided by the orchestrator
            "lost": bool(role_meta.get("lost", False)),
            "loss_reason": role_meta.get("loss_reason"),
            "crash": bool(role_meta.get("crash", False)),
            "illegal": bool(role_meta.get("illegal", False)),
            "timeout": bool(role_meta.get("timeout", False)),
            "final_deck_count": role_meta.get("final_deck_count"),
            "in_play_count_final": role_meta.get("in_play_count_final"),
            "ml": role_meta.get("ml"),
        }

    return {
        "pair_id": meta.get("pair_id"),
        "game_index": meta.get("game_index"),
        "seed": meta.get("seed"),
        "champion_seat": meta.get("champion_seat"),
        "challenger_seat": meta.get("challenger_seat"),
        "first_player_role": meta.get("first_player_role"),
        "winner_role": meta.get("winner_role"),
        "result": meta.get("result"),
        "termination": meta.get("termination"),
        "turns": meta.get("turns"),
        CHAMPION_ROLE: per_role[CHAMPION_ROLE],
        CHALLENGER_ROLE: per_role[CHALLENGER_ROLE],
    }


# ---------------------------------------------------------------------------
# Matchup aggregation
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_role(records: list[dict[str, Any]], role: str) -> dict[str, Any]:
    """Aggregate one role's tactical/safety metrics across games."""
    games = len(records)
    rows = [r[role] for r in records]

    attacks = sum(r["attacks"] for r in rows)
    alakazam_attacks = sum(r["alakazam_attacks"] for r in rows)
    acting_turns = sum(r["acting_turns"] for r in rows)
    attack_turns = sum(r["attack_turns"] for r in rows)
    searches = sum(r["search_uses"] for r in rows)
    decisions = sum(r["decisions"] for r in rows)

    first_turns = [r["first_attack_turn"] for r in rows if r["first_attack_turn"] is not None]
    games_with_attack = len(first_turns)
    t2_attacks = sum(1 for t in first_turns if t <= 2)

    losses = [r for r in rows if r["lost"]]
    idle_in_losses = [
        r["idle_turns_after_first_attack"]
        for r in losses
        if r["first_attack_turn"] is not None
    ]

    deckouts = sum(1 for r in rows if r["loss_reason"] == "deckout")
    boardouts = sum(1 for r in rows if r["loss_reason"] == "boardout")
    prize_losses = sum(1 for r in rows if r["loss_reason"] == "prizes")

    hand_samples = [h for r in rows for h in r["hand_size_samples"]]
    hand_at_alakazam = [h for r in rows for h in r["hand_at_alakazam_attack"]]
    overkill_total = sum(r["overkill_total"] for r in rows)
    overkill_samples = sum(r["overkill_samples"] for r in rows)

    decision_ms_total = sum(r["decision_ms_total"] for r in rows)
    decision_ms_max = max((r["decision_ms_max"] for r in rows), default=0.0)

    crashes = sum(1 for r in rows if r["crash"])
    illegals = sum(1 for r in rows if r["illegal"])
    timeouts = sum(1 for r in rows if r["timeout"])

    return {
        "role": role,
        "games": games,
        # tactical
        "avg_first_attack_turn": _mean(first_turns),
        "games_with_attack": games_with_attack,
        "t2_attack_rate": _safe_div(t2_attacks, games),
        "attack_turn_rate": _safe_div(attack_turns, acting_turns),
        "attacks_per_game": _safe_div(attacks, games),
        "alakazam_attacks_per_game": _safe_div(alakazam_attacks, games),
        "idle_turns_after_first_attack_in_losses": _mean(idle_in_losses),
        "avg_game_turns": _mean([r for r in (rec["turns"] for rec in records) if r is not None]),
        "search_uses_per_attack": _safe_div(searches, max(1, attacks)),
        "avg_hand_size": _mean(hand_samples),
        "avg_hand_at_alakazam_attack": _mean(hand_at_alakazam),
        "avg_overkill": _safe_div(overkill_total, max(1, overkill_samples)),
        # deck / board outcomes
        "deckouts": deckouts,
        "deckout_rate": _safe_div(deckouts, games),
        "boardouts": boardouts,
        "boardout_rate": _safe_div(boardouts, games),
        "prize_losses": prize_losses,
        # safety
        "crashes": crashes,
        "illegal_actions": illegals,
        "timeouts": timeouts,
        "avg_decision_ms": _safe_div(decision_ms_total, max(1, decisions)),
        "max_decision_ms": decision_ms_max,
        "decisions": decisions,
    }


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    phat = wins / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def aggregate_matchup(
    records: list[dict[str, Any]],
    start_errors: int = 0,
) -> dict[str, Any]:
    """Aggregate head-to-head champion-vs-challenger results."""
    games = len(records)
    champion_wins = sum(1 for r in records if r["winner_role"] == CHAMPION_ROLE)
    challenger_wins = sum(1 for r in records if r["winner_role"] == CHALLENGER_ROLE)
    draws = sum(1 for r in records if r["winner_role"] == "draw")
    decided = champion_wins + challenger_wins

    # seat-based win rate for the challenger
    chal_seat0 = [r for r in records if r["challenger_seat"] == 0]
    chal_seat1 = [r for r in records if r["challenger_seat"] == 1]
    seat0_wins = sum(1 for r in chal_seat0 if r["winner_role"] == CHALLENGER_ROLE)
    seat1_wins = sum(1 for r in chal_seat1 if r["winner_role"] == CHALLENGER_ROLE)
    seat0_decided = sum(1 for r in chal_seat0 if r["winner_role"] != "draw")
    seat1_decided = sum(1 for r in chal_seat1 if r["winner_role"] != "draw")

    # first/second player win rate for the challenger
    chal_first = [r for r in records if r["first_player_role"] == CHALLENGER_ROLE]
    chal_second = [r for r in records if r["first_player_role"] == CHAMPION_ROLE]
    first_wins = sum(1 for r in chal_first if r["winner_role"] == CHALLENGER_ROLE)
    second_wins = sum(1 for r in chal_second if r["winner_role"] == CHALLENGER_ROLE)
    first_decided = sum(1 for r in chal_first if r["winner_role"] != "draw")
    second_decided = sum(1 for r in chal_second if r["winner_role"] != "draw")

    # seed-pair analysis
    pairs: dict[Any, list[dict[str, Any]]] = {}
    for r in records:
        pairs.setdefault(r["pair_id"], []).append(r)
    both_seat_wins = 0
    one_seat_wins = 0
    full_pairs = 0
    for pair_records in pairs.values():
        if len(pair_records) < 2:
            continue
        full_pairs += 1
        chal_pair_wins = sum(1 for r in pair_records if r["winner_role"] == CHALLENGER_ROLE)
        if chal_pair_wins == 2:
            both_seat_wins += 1
        elif chal_pair_wins == 1:
            one_seat_wins += 1

    win_rate = _safe_div(challenger_wins, decided)
    ci_low, ci_high = wilson_interval(challenger_wins, decided)

    return {
        "games": games,
        "champion_wins": champion_wins,
        "challenger_wins": challenger_wins,
        "draws": draws,
        "decided_games": decided,
        "start_errors": start_errors,
        "challenger_win_rate": win_rate,
        "challenger_win_rate_ci_low": ci_low,
        "challenger_win_rate_ci_high": ci_high,
        "challenger_seat0_win_rate": _safe_div(seat0_wins, seat0_decided),
        "challenger_seat1_win_rate": _safe_div(seat1_wins, seat1_decided),
        "challenger_first_player_win_rate": _safe_div(first_wins, first_decided),
        "challenger_second_player_win_rate": _safe_div(second_wins, second_decided),
        "seed_pairs": full_pairs,
        "challenger_won_both_seats_pairs": both_seat_wins,
        "challenger_won_one_seat_pairs": one_seat_wins,
    }


def aggregate_ml_diagnostics(records: list[dict[str, Any]], role: str) -> dict[str, Any]:
    """Sum ML diagnostic deltas across games for a role, if present."""
    totals: dict[str, float] = {}
    present = False
    for record in records:
        ml = record[role].get("ml")
        if not isinstance(ml, dict):
            continue
        present = True
        for key, value in ml.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + value
    if not present:
        return {}
    decisions = totals.get("decisions", 0.0)
    if decisions:
        totals["adoption_rate"] = _safe_div(totals.get("model_selected", 0.0), decisions)
        totals["fallback_rate"] = _safe_div(totals.get("fallback", 0.0), decisions)
    return totals


# ---------------------------------------------------------------------------
# Promotion judgement
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    observed: Any
    threshold: Any
    comparator: str  # ">=" or "<="
    category: str  # "winrate" | "tactical" | "safety" | "sample"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pass": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "comparator": self.comparator,
            "category": self.category,
        }


def _ge(name, observed, threshold, category) -> Check:
    return Check(name, observed >= threshold, observed, threshold, ">=", category)


def _le(name, observed, threshold, category) -> Check:
    return Check(name, observed <= threshold, observed, threshold, "<=", category)


def judge_promotion(
    matchup: dict[str, Any],
    challenger_role_metrics: dict[str, Any],
    config: dict[str, Any],
    preflight_ok: bool = True,
) -> dict[str, Any]:
    """Return the final verdict + per-condition PASS/FAIL breakdown.

    Priority: INVALID_EVALUATION > REJECT > (PROMOTE_RECOMMENDED | HOLD).
    """
    thresholds = config.get("promotion_thresholds") or {}
    minimum_games = int(config.get("minimum_games", 1))
    ci_floor = config.get("require_confidence_interval_above")

    checks: list[Check] = []

    # --- sample sufficiency ---
    games_check = _ge("minimum_games", matchup["games"], minimum_games, "sample")
    checks.append(games_check)

    # --- head-to-head win rate ---
    win_rate = matchup["challenger_win_rate"]
    win_rate_check = _ge(
        "minimum_head_to_head_win_rate",
        round(win_rate, 4),
        thresholds.get("minimum_head_to_head_win_rate", 0.53),
        "winrate",
    )
    checks.append(win_rate_check)

    ci_check = None
    if ci_floor is not None:
        ci_check = _ge(
            "confidence_interval_lower_above",
            round(matchup["challenger_win_rate_ci_low"], 4),
            float(ci_floor),
            "winrate",
        )
        checks.append(ci_check)

    # --- tactical gates ---
    tactical_specs = [
        ("minimum_attack_turn_rate", challenger_role_metrics["attack_turn_rate"], _ge),
        (
            "minimum_alakazam_attacks_per_game",
            challenger_role_metrics["alakazam_attacks_per_game"],
            _ge,
        ),
        ("maximum_deckout_rate", challenger_role_metrics["deckout_rate"], _le),
        ("maximum_boardout_rate", challenger_role_metrics["boardout_rate"], _le),
        (
            "maximum_post_first_attack_idle_turns_in_losses",
            challenger_role_metrics["idle_turns_after_first_attack_in_losses"],
            _le,
        ),
    ]
    for name, observed, comparator in tactical_specs:
        if name in thresholds:
            checks.append(comparator(name, round(float(observed), 4), thresholds[name], "tactical"))

    # --- safety gates ---
    safety_specs = [
        ("maximum_crashes", challenger_role_metrics["crashes"]),
        ("maximum_illegal_actions", challenger_role_metrics["illegal_actions"]),
        ("maximum_timeouts", challenger_role_metrics["timeouts"]),
    ]
    for name, observed in safety_specs:
        if name in thresholds:
            checks.append(_le(name, observed, thresholds[name], "safety"))

    check_dicts = [c.to_dict() for c in checks]

    def failed(category: str) -> list[str]:
        return [c.name for c in checks if c.category == category and not c.passed]

    reasons: list[str] = []

    # --- INVALID: evaluation did not really happen ---
    invalid = False
    if not preflight_ok:
        invalid = True
        reasons.append("preflight_failed")
    if matchup["decided_games"] == 0:
        invalid = True
        reasons.append("no_decided_games")
    if matchup["games"] == 0:
        invalid = True
        reasons.append("no_games_completed")
    if matchup.get("start_errors", 0) and matchup["start_errors"] >= matchup["games"]:
        invalid = True
        reasons.append("engine_start_errors")

    if invalid:
        return _verdict(VERDICT_INVALID, check_dicts, reasons)

    # --- REJECT: clear regression ---
    safety_failed = failed("safety")
    if safety_failed:
        reasons.append("safety_violation:" + ",".join(safety_failed))
    # clearly losing: point estimate below 0.5 AND even the optimistic CI bound
    # does not clear 0.5
    if win_rate < 0.5 and matchup["challenger_win_rate_ci_high"] < 0.5:
        reasons.append("head_to_head_regression")
    # catastrophic play regressions
    for name in ("maximum_deckout_rate", "maximum_boardout_rate"):
        if name in failed("tactical"):
            reasons.append(f"play_regression:{name}")

    reject = bool(safety_failed) or "head_to_head_regression" in reasons or any(
        r.startswith("play_regression:") for r in reasons
    )
    if reject:
        return _verdict(VERDICT_REJECT, check_dicts, reasons)

    # --- PROMOTE vs HOLD ---
    all_pass = all(c.passed for c in checks)
    if all_pass:
        return _verdict(VERDICT_PROMOTE, check_dicts, reasons or ["all_conditions_met"])

    failing = [c.name for c in checks if not c.passed]
    reasons.append("unmet_conditions:" + ",".join(failing))
    return _verdict(VERDICT_HOLD, check_dicts, reasons)


def _verdict(verdict: str, checks: list[dict[str, Any]], reasons: list[str]) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "checks": checks,
        "reasons": reasons,
        "note": (
            "This evaluation never overwrites the Champion, edits models, or performs "
            "git/Kaggle actions. Formal promotion is a separate, human-invoked step "
            "(scripts/promote_challenger.py)."
        ),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    """Render the full promotion_report.md from a resolved report dict."""
    meta = report["meta"]
    matchup = report["head_to_head"]
    champ = report["champion_metrics"]
    chal = report["challenger_metrics"]
    judgement = report["judgement"]
    baseline = report.get("baseline", {})

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines: list[str] = []
    lines.append(f"# Champion-Challenger Report: {meta['champion']} vs {meta['challenger']}")
    lines.append("")
    lines.append(f"**Final verdict: `{judgement['verdict']}`**")
    lines.append("")
    lines.append("## 1. Identity")
    lines.append("")
    lines.append(f"- Champion: `{meta['champion']}`")
    lines.append(f"- Challenger: `{meta['challenger']}`")
    lines.append(f"- Baseline: `{meta.get('baseline') or 'n/a'}`")
    lines.append(f"- Champion model hash: `{meta.get('champion_model_hash') or 'n/a'}`")
    lines.append(f"- Challenger model hash: `{meta.get('challenger_model_hash') or 'n/a'}`")
    lines.append(f"- Champion deck hash: `{meta.get('champion_deck_hash') or 'n/a'}`")
    lines.append(f"- Challenger deck hash: `{meta.get('challenger_deck_hash') or 'n/a'}`")
    lines.append(f"- Run timestamp: `{meta.get('timestamp')}`")
    lines.append(f"- Git commit: `{meta.get('git_commit') or 'n/a'}`")
    lines.append(f"- Environment: `{meta.get('python_version')}` on `{meta.get('platform')}`")
    lines.append(f"- Games: `{matchup['games']}` (seat_swap=`{meta.get('seat_swap')}`)")
    lines.append("")
    lines.append(
        "> Seeding note: the bundled cg engine exposes no RNG seed, so the two "
        "games of each seed pair are not bit-identical shuffles. Fairness is "
        "enforced by seat swap (each agent plays seat 0 and seat 1 equally)."
    )
    lines.append("")

    lines.append("## 2. Head-to-Head")
    lines.append("")
    lines.append(f"- Champion wins: `{matchup['champion_wins']}`")
    lines.append(f"- Challenger wins: `{matchup['challenger_wins']}`")
    lines.append(f"- Draws: `{matchup['draws']}`")
    lines.append(
        f"- Challenger win rate (ex-draws): **{pct(matchup['challenger_win_rate'])}**"
    )
    lines.append(
        f"- 95% CI (Wilson): "
        f"{pct(matchup['challenger_win_rate_ci_low'])}"
        f"–{pct(matchup['challenger_win_rate_ci_high'])}"
    )
    lines.append("")
    lines.append("## 3. Seat / Turn-Order Breakdown")
    lines.append("")
    lines.append(f"- Challenger seat 0 win rate: {pct(matchup['challenger_seat0_win_rate'])}")
    lines.append(f"- Challenger seat 1 win rate: {pct(matchup['challenger_seat1_win_rate'])}")
    lines.append(
        f"- Challenger first-player win rate: {pct(matchup['challenger_first_player_win_rate'])}"
    )
    lines.append(
        f"- Challenger second-player win rate: {pct(matchup['challenger_second_player_win_rate'])}"
    )
    lines.append(f"- Seed pairs: {matchup['seed_pairs']}")
    lines.append(f"- Challenger won both seats: {matchup['challenger_won_both_seats_pairs']} pairs")
    lines.append(f"- Challenger won one seat only: {matchup['challenger_won_one_seat_pairs']} pairs")
    lines.append("")

    lines.append("## 4. Tactical Metrics (Challenger vs Champion)")
    lines.append("")
    lines.append("| Metric | Challenger | Champion |")
    lines.append("|---|---|---|")
    tactical_keys = [
        ("avg_first_attack_turn", "Avg first attack turn"),
        ("t2_attack_rate", "Attack-by-T2 rate"),
        ("attack_turn_rate", "Attack rate (per own turn)"),
        ("attacks_per_game", "Attacks / game"),
        ("alakazam_attacks_per_game", "Alakazam attacks / game"),
        ("idle_turns_after_first_attack_in_losses", "Idle turns post-1st-attack (losses)"),
        ("avg_game_turns", "Avg game turns"),
        ("search_uses_per_attack", "Search uses / attack"),
        ("avg_hand_size", "Avg hand size"),
        ("avg_hand_at_alakazam_attack", "Avg hand @ Alakazam attack"),
        ("avg_overkill", "Avg overkill damage"),
        ("deckout_rate", "Deckout rate"),
        ("boardout_rate", "Boardout rate"),
        ("prize_losses", "Normal prize losses"),
    ]
    for key, label in tactical_keys:
        lines.append(f"| {label} | {chal.get(key)} | {champ.get(key)} |")
    lines.append("")

    lines.append("## 5. Safety Metrics (Challenger vs Champion)")
    lines.append("")
    lines.append("| Metric | Challenger | Champion |")
    lines.append("|---|---|---|")
    for key, label in [
        ("crashes", "Crashes"),
        ("illegal_actions", "Illegal actions"),
        ("timeouts", "Timeouts"),
        ("avg_decision_ms", "Avg decision ms"),
        ("max_decision_ms", "Max decision ms"),
    ]:
        lines.append(f"| {label} | {chal.get(key)} | {champ.get(key)} |")
    preflight = report.get("preflight", {})
    lines.append(
        f"| Import failures | {preflight.get('challenger_import_failed')} "
        f"| {preflight.get('champion_import_failed')} |"
    )
    lines.append(
        f"| Model load failures | {preflight.get('challenger_model_failed')} "
        f"| {preflight.get('champion_model_failed')} |"
    )
    lines.append("")

    if baseline:
        lines.append("## 6. Baseline Comparison")
        lines.append("")
        lines.append(f"Baseline agent: `{meta.get('baseline')}`")
        lines.append("")
        lines.append("| Matchup | Games | Win rate |")
        lines.append("|---|---|---|")
        for label, data in baseline.items():
            lines.append(
                f"| {label} | {data.get('games')} | {pct(data.get('win_rate', 0.0))} |"
            )
        note = report.get("baseline_note")
        if note:
            lines.append("")
            lines.append(f"> {note}")
        lines.append("")

    ml = report.get("ml_diagnostics", {})
    if ml.get("challenger"):
        lines.append("## 7. ML Diagnostics (Challenger)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key in sorted(ml["challenger"].keys()):
            lines.append(f"| {key} | {ml['challenger'][key]} |")
        lines.append("")

    lines.append("## 8. Promotion Conditions (PASS / FAIL)")
    lines.append("")
    lines.append("| Condition | Observed | Threshold | Result |")
    lines.append("|---|---|---|---|")
    for check in judgement["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(
            f"| {check['name']} | {check['observed']} "
            f"| {check['comparator']} {check['threshold']} | {status} |"
        )
    lines.append("")

    lines.append("## 9. Final Verdict")
    lines.append("")
    lines.append(f"**`{judgement['verdict']}`**")
    lines.append("")
    if judgement.get("reasons"):
        lines.append("Reasons / remaining concerns:")
        lines.append("")
        for reason in judgement["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("## 10. Formal Promotion Procedure (human-only)")
    lines.append("")
    lines.append(
        "This report does NOT change the Champion. To promote after human review:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/promote_challenger.py \\")
    lines.append(f"  --report {report.get('report_path', '<this report>.json')} \\")
    lines.append(f"  --new-agent-name {meta['challenger']}_promoted \\")
    lines.append("  --dry-run   # inspect, then re-run with --apply")
    lines.append("```")
    lines.append("")
    lines.append(
        "`--apply` only copies files into a new agent directory. It never edits the "
        "existing Champion, commits, tags, pushes, or submits to Kaggle."
    )
    lines.append("")

    if report.get("failures"):
        lines.append("## 11. Failures Observed")
        lines.append("")
        for failure in report["failures"]:
            lines.append(f"- {failure}")
        lines.append("")

    return "\n".join(lines) + "\n"
