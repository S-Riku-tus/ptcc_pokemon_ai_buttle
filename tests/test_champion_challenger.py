"""Fast, engine-free tests for the Champion-Challenger core logic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cc_core  # noqa: E402
import promote_challenger  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_event(role, turn, *, attack=False, alakazam=False, hand=5, search=False, ms=1.0):
    return {
        "role": role,
        "seat": 0,
        "turn": turn,
        "action_type": "attack" if attack else "other",
        "is_attack": attack,
        "is_alakazam_attack": alakazam,
        "is_search": search,
        "hand_count": hand,
        "overkill": 10.0 if alakazam else 0.0,
        "decision_ms": ms,
    }


def make_game(
    *,
    pair_id=0,
    game_index=1,
    seed=1,
    champion_seat=1,
    challenger_seat=0,
    first_player_role=cc_core.CHALLENGER_ROLE,
    winner_role=cc_core.CHALLENGER_ROLE,
    result="win",
    turns=10,
    challenger_events=None,
    champion_events=None,
    challenger_meta=None,
    champion_meta=None,
):
    events = list(challenger_events or []) + list(champion_events or [])
    meta = {
        "pair_id": pair_id,
        "game_index": game_index,
        "seed": seed,
        "champion_seat": champion_seat,
        "challenger_seat": challenger_seat,
        "first_player_role": first_player_role,
        "winner_role": winner_role,
        "result": result,
        "termination": "normal",
        "turns": turns,
        cc_core.CHAMPION_ROLE: champion_meta or {},
        cc_core.CHALLENGER_ROLE: challenger_meta or {},
    }
    return cc_core.summarize_game(events, meta)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_load_config_merges_over_defaults(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(
        json.dumps(
            {
                "champion_agent": "champ",
                "challenger_agent": "chal",
                "games": 4,
                "promotion_thresholds": {"minimum_head_to_head_win_rate": 0.6},
            }
        ),
        encoding="utf-8",
    )
    config = cc_core.load_config(cfg_file)
    assert config["champion_agent"] == "champ"
    assert config["games"] == 4
    # deep-merge keeps other default thresholds
    assert config["promotion_thresholds"]["minimum_head_to_head_win_rate"] == 0.6
    assert "maximum_crashes" in config["promotion_thresholds"]


def test_cli_overrides_take_precedence():
    config = dict(cc_core.DEFAULT_CONFIG)
    config["champion_agent"] = "from_config"
    merged = cc_core.apply_overrides(config, {"champion": "from_cli", "games": None})
    assert merged["champion_agent"] == "from_cli"
    assert merged["games"] == config["games"]  # None override ignored


def test_validate_config_rejects_missing_agents():
    config = cc_core.load_config(None)
    with pytest.raises(ValueError):
        cc_core.validate_config(config)


def test_validate_config_rejects_same_champion_challenger():
    config = cc_core.load_config(None)
    config["champion_agent"] = config["challenger_agent"] = "same"
    config["games"] = 4
    with pytest.raises(ValueError, match="differ"):
        cc_core.validate_config(config)


def test_validate_config_rejects_bad_games():
    config = cc_core.load_config(None)
    config["champion_agent"] = "a"
    config["challenger_agent"] = "b"
    config["games"] = 0
    with pytest.raises(ValueError, match="positive"):
        cc_core.validate_config(config)


def test_validate_config_rejects_odd_games_with_seat_swap():
    config = cc_core.load_config(None)
    config.update({"champion_agent": "a", "challenger_agent": "b", "games": 3, "seat_swap": True})
    with pytest.raises(ValueError, match="even"):
        cc_core.validate_config(config)


def test_validate_config_rejects_bad_threshold():
    config = cc_core.load_config(None)
    config.update({"champion_agent": "a", "challenger_agent": "b", "games": 2})
    config["promotion_thresholds"]["minimum_head_to_head_win_rate"] = 1.5
    with pytest.raises(ValueError, match="rate threshold"):
        cc_core.validate_config(config)


def test_validate_config_rejects_nonexistent_agent():
    with pytest.raises(FileNotFoundError):
        cc_core.resolve_agent_dir("this_agent_does_not_exist_xyz")


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------


def test_build_pairs_seat_swap_balances_seats():
    pairs = cc_core.build_pairs(200, base_seed=7, seat_swap=True)
    assert len(pairs) == 200
    seat0_roles = [p.seat0_role for p in pairs]
    # each role occupies seat 0 exactly half the time
    assert seat0_roles.count(cc_core.CHALLENGER_ROLE) == 100
    assert seat0_roles.count(cc_core.CHAMPION_ROLE) == 100


def test_build_pairs_pairs_share_seed_and_swap():
    pairs = cc_core.build_pairs(4, base_seed=100, seat_swap=True)
    # games 0,1 form pair 0 with same seed and swapped seats
    assert pairs[0].pair_id == pairs[1].pair_id == 0
    assert pairs[0].seed == pairs[1].seed == 100
    assert pairs[0].seat0_role != pairs[1].seat0_role


def test_build_pairs_reproducible():
    a = cc_core.build_pairs(10, 42, True)
    b = cc_core.build_pairs(10, 42, True)
    assert a == b


def test_build_pairs_odd_count_handled():
    pairs = cc_core.build_pairs(5, 0, True)
    assert len(pairs) == 5  # 2 full pairs + 1 remainder


def test_build_pairs_no_seat_swap_still_alternates():
    pairs = cc_core.build_pairs(4, 0, False)
    seat0 = [p.seat0_role for p in pairs]
    assert seat0.count(cc_core.CHALLENGER_ROLE) == 2
    assert seat0.count(cc_core.CHAMPION_ROLE) == 2


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_summarize_game_tactical_counts():
    events = [
        make_event(cc_core.CHALLENGER_ROLE, 3, attack=True, alakazam=True, hand=6),
        make_event(cc_core.CHALLENGER_ROLE, 5, attack=False),  # idle after first attack
        make_event(cc_core.CHALLENGER_ROLE, 7, attack=True, alakazam=True, hand=4),
    ]
    record = make_game(challenger_events=events, challenger_meta={"lost": True})
    chal = record[cc_core.CHALLENGER_ROLE]
    assert chal["attacks"] == 2
    assert chal["alakazam_attacks"] == 2
    assert chal["first_attack_turn"] == 3
    assert chal["acting_turns"] == 3
    assert chal["attack_turns"] == 2
    assert chal["idle_turns_after_first_attack"] == 1


def test_aggregate_matchup_winrate_and_seats():
    records = [
        make_game(challenger_seat=0, champion_seat=1, winner_role=cc_core.CHALLENGER_ROLE),
        make_game(challenger_seat=1, champion_seat=0, winner_role=cc_core.CHAMPION_ROLE),
        make_game(challenger_seat=0, champion_seat=1, winner_role=cc_core.CHALLENGER_ROLE),
        make_game(challenger_seat=1, champion_seat=0, winner_role="draw", result="draw"),
    ]
    agg = cc_core.aggregate_matchup(records)
    assert agg["games"] == 4
    assert agg["challenger_wins"] == 2
    assert agg["champion_wins"] == 1
    assert agg["draws"] == 1
    assert agg["decided_games"] == 3
    assert agg["challenger_win_rate"] == pytest.approx(2 / 3)
    # challenger won both its seat-0 games
    assert agg["challenger_seat0_win_rate"] == pytest.approx(1.0)


def test_aggregate_role_rates():
    records = []
    for i in range(10):
        lost = i < 2  # 2 deckouts
        meta = {"lost": lost, "loss_reason": "deckout" if lost else None}
        events = [make_event(cc_core.CHALLENGER_ROLE, 2, attack=True, alakazam=True)]
        records.append(make_game(challenger_events=events, challenger_meta=meta, turns=8))
    role = cc_core.aggregate_role(records, cc_core.CHALLENGER_ROLE)
    assert role["deckouts"] == 2
    assert role["deckout_rate"] == pytest.approx(0.2)
    assert role["attack_turn_rate"] == pytest.approx(1.0)
    assert role["alakazam_attacks_per_game"] == pytest.approx(1.0)


def test_aggregate_role_counts_crashes_and_boardouts():
    records = [
        make_game(challenger_meta={"lost": True, "crash": True}),
        make_game(challenger_meta={"lost": True, "loss_reason": "boardout"}),
        make_game(challenger_meta={"illegal": True, "lost": True}),
    ]
    role = cc_core.aggregate_role(records, cc_core.CHALLENGER_ROLE)
    assert role["crashes"] == 1
    assert role["illegal_actions"] == 1
    assert role["boardouts"] == 1
    assert role["boardout_rate"] == pytest.approx(1 / 3)


def test_wilson_interval():
    low, high = cc_core.wilson_interval(55, 100)
    assert 0.45 <= low <= 0.46
    assert 0.64 <= high <= 0.65
    assert cc_core.wilson_interval(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# judgement
# ---------------------------------------------------------------------------


def _clean_challenger_metrics(**overrides):
    base = {
        "attack_turn_rate": 0.8,
        "alakazam_attacks_per_game": 4.0,
        "deckout_rate": 0.0,
        "boardout_rate": 0.0,
        "idle_turns_after_first_attack_in_losses": 0.0,
        "crashes": 0,
        "illegal_actions": 0,
        "timeouts": 0,
    }
    base.update(overrides)
    return base


def _matchup(win_rate=0.6, ci_low=0.55, ci_high=0.65, games=200, decided=200, start_errors=0):
    return {
        "games": games,
        "decided_games": decided,
        "challenger_win_rate": win_rate,
        "challenger_win_rate_ci_low": ci_low,
        "challenger_win_rate_ci_high": ci_high,
        "start_errors": start_errors,
    }


def _config():
    return cc_core.load_config(None) | {"minimum_games": 200, "require_confidence_interval_above": 0.50}


def test_judge_promote_recommended():
    result = cc_core.judge_promotion(_matchup(), _clean_challenger_metrics(), _config())
    assert result["verdict"] == cc_core.VERDICT_PROMOTE


def test_judge_reject_on_crash():
    result = cc_core.judge_promotion(
        _matchup(win_rate=0.7, ci_low=0.65), _clean_challenger_metrics(crashes=1), _config()
    )
    assert result["verdict"] == cc_core.VERDICT_REJECT
    assert any("safety" in r for r in result["reasons"])


def test_judge_reject_on_illegal():
    result = cc_core.judge_promotion(
        _matchup(), _clean_challenger_metrics(illegal_actions=1), _config()
    )
    assert result["verdict"] == cc_core.VERDICT_REJECT


def test_judge_reject_on_losing_winrate():
    result = cc_core.judge_promotion(
        _matchup(win_rate=0.4, ci_low=0.33, ci_high=0.47), _clean_challenger_metrics(), _config()
    )
    assert result["verdict"] == cc_core.VERDICT_REJECT


def test_judge_reject_on_deckout():
    result = cc_core.judge_promotion(
        _matchup(), _clean_challenger_metrics(deckout_rate=0.2), _config()
    )
    assert result["verdict"] == cc_core.VERDICT_REJECT


def test_judge_hold_on_insufficient_games():
    result = cc_core.judge_promotion(
        _matchup(games=20, decided=20), _clean_challenger_metrics(), _config()
    )
    assert result["verdict"] == cc_core.VERDICT_HOLD


def test_judge_hold_on_wide_ci():
    result = cc_core.judge_promotion(
        _matchup(win_rate=0.55, ci_low=0.45, ci_high=0.65), _clean_challenger_metrics(), _config()
    )
    assert result["verdict"] == cc_core.VERDICT_HOLD


def test_judge_invalid_on_no_decided_games():
    result = cc_core.judge_promotion(
        _matchup(games=0, decided=0), _clean_challenger_metrics(), _config()
    )
    assert result["verdict"] == cc_core.VERDICT_INVALID


def test_judge_invalid_on_preflight_failure():
    result = cc_core.judge_promotion(
        _matchup(), _clean_challenger_metrics(), _config(), preflight_ok=False
    )
    assert result["verdict"] == cc_core.VERDICT_INVALID


# ---------------------------------------------------------------------------
# output / report
# ---------------------------------------------------------------------------


def test_render_markdown_has_required_sections():
    records = [
        make_game(
            winner_role=cc_core.CHALLENGER_ROLE,
            challenger_events=[make_event(cc_core.CHALLENGER_ROLE, 2, attack=True, alakazam=True)],
        )
        for _ in range(4)
    ]
    matchup = cc_core.aggregate_matchup(records)
    champ = cc_core.aggregate_role(records, cc_core.CHAMPION_ROLE)
    chal = cc_core.aggregate_role(records, cc_core.CHALLENGER_ROLE)
    judgement = cc_core.judge_promotion(matchup, chal, _config())
    report = {
        "meta": {
            "champion": "champ", "challenger": "chal", "baseline": "base",
            "champion_model_hash": "h1", "challenger_model_hash": "h2",
            "champion_deck_hash": "d1", "challenger_deck_hash": "d2",
            "timestamp": "20260717_000000", "git_commit": "abc123",
            "python_version": "3.11.0", "platform": "test", "seat_swap": True, "seed": 1,
        },
        "head_to_head": matchup,
        "champion_metrics": champ,
        "challenger_metrics": chal,
        "ml_diagnostics": {"challenger": {"model_selected": 5}, "champion": {}},
        "baseline": {"challenger_vs_baseline": {"games": 4, "win_rate": 0.75}},
        "baseline_note": "note",
        "judgement": judgement,
        "preflight": {},
        "failures": [],
        "report_path": "x.json",
    }
    md = cc_core.render_markdown(report)
    assert "Champion-Challenger Report" in md
    assert "Head-to-Head" in md
    assert "Promotion Conditions" in md
    assert "Final Verdict" in md
    assert "model hash" in md
    assert "Formal Promotion Procedure" in md
    assert judgement["verdict"] in md


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_detect_challengers_by_name_and_role(tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    for name, meta in [
        ("champ", {"role": "champion"}),
        ("alakazam_v4_candidate", {}),
        ("some_challenger", {}),
        ("role_marked", {"role": "challenger", "created_at": "2026-07-17T10:00:00"}),
        ("plain_agent", {}),
    ]:
        d = agents / name
        d.mkdir()
        (d / "main.py").write_text("def agent(o):\n    return []\n", encoding="utf-8")
        (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    found = {c["name"] for c in cc_core.detect_challengers("champ", agents_root=agents)}
    assert "alakazam_v4_candidate" in found
    assert "some_challenger" in found
    assert "role_marked" in found
    assert "champ" not in found
    assert "plain_agent" not in found


# ---------------------------------------------------------------------------
# promotion (dry-run / no file changes)
# ---------------------------------------------------------------------------


def test_promotion_dry_run_changes_nothing(tmp_path, monkeypatch):
    # build a fake challenger agent + a report pointing at it
    agents = tmp_path / "agents"
    (agents / "chal_agent").mkdir(parents=True)
    (agents / "chal_agent" / "main.py").write_text("def agent(o):\n    return []\n", encoding="utf-8")
    (agents / "chal_agent" / "deck.csv").write_text("\n".join(["1"] * 60), encoding="utf-8")
    monkeypatch.setattr(cc_core, "ROOT", tmp_path)
    monkeypatch.setattr(promote_challenger, "ROOT", tmp_path)

    report = {
        "meta": {"challenger": "chal_agent"},
        "judgement": {"verdict": cc_core.VERDICT_PROMOTE},
    }
    report_path = tmp_path / "promotion_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    rc = promote_challenger.main(
        ["--report", str(report_path), "--new-agent-name", "new_champ", "--dry-run"]
    )
    assert rc == 0
    assert not (agents / "new_champ").exists()  # dry-run created nothing


def test_promotion_apply_creates_new_dir_only(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    (agents / "chal_agent").mkdir(parents=True)
    (agents / "chal_agent" / "main.py").write_text("def agent(o):\n    return []\n", encoding="utf-8")
    (agents / "chal_agent" / "deck.csv").write_text("\n".join(["1"] * 60), encoding="utf-8")
    (agents / "chal_agent" / "metadata.json").write_text(json.dumps({"name": "chal_agent"}), encoding="utf-8")
    monkeypatch.setattr(cc_core, "ROOT", tmp_path)
    monkeypatch.setattr(promote_challenger, "ROOT", tmp_path)

    report = {"meta": {"challenger": "chal_agent"}, "judgement": {"verdict": cc_core.VERDICT_PROMOTE}}
    report_path = tmp_path / "promotion_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    rc = promote_challenger.main(
        ["--report", str(report_path), "--new-agent-name", "new_champ", "--apply"]
    )
    assert rc == 0
    assert (agents / "new_champ" / "main.py").exists()
    # original challenger untouched
    assert (agents / "chal_agent" / "main.py").exists()
    new_meta = json.loads((agents / "new_champ" / "metadata.json").read_text(encoding="utf-8"))
    assert new_meta["name"] == "new_champ"
    assert new_meta["promoted_from"] == "chal_agent"


def test_promotion_refuses_non_promote_verdict(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    (agents / "chal_agent").mkdir(parents=True)
    (agents / "chal_agent" / "main.py").write_text("def agent(o):\n    return []\n", encoding="utf-8")
    monkeypatch.setattr(cc_core, "ROOT", tmp_path)
    monkeypatch.setattr(promote_challenger, "ROOT", tmp_path)

    report = {"meta": {"challenger": "chal_agent"}, "judgement": {"verdict": cc_core.VERDICT_HOLD}}
    report_path = tmp_path / "promotion_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    rc = promote_challenger.main(
        ["--report", str(report_path), "--new-agent-name", "new_champ", "--apply"]
    )
    assert rc == 1  # blocked
    assert not (agents / "new_champ").exists()
