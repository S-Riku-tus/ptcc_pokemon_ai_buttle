from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paired_gauntlet import (  # noqa: E402
    Opponent,
    _paired_exact_p,
    _parse_agent_spec,
    build_schedule,
    paired_summary,
)
from policy_impact_gate import evaluate_impact  # noqa: E402


def test_schedule_is_weighted_paired_and_seat_mirrored():
    opponents = [Opponent("mirror", "a", 0.75), Opponent("other", "b", 0.25)]
    jobs = build_schedule(opponents, blocks=8, base_seed=10, both_seats=True, calibration_blocks=0)
    primary = [job for job in jobs if job.repeat == 0]
    assert len(primary) == 8 * 2 * 2
    counts = {label: sum(job.opponent == label for job in primary) for label in ("mirror", "other")}
    assert counts == {"mirror": 24, "other": 8}
    for block_id in range(8):
        block = [job for job in primary if job.block_id == block_id]
        assert {job.evaluated_seat for job in block} == {0, 1}
        assert {job.treatment for job in block} == {"champion", "challenger"}
        assert len({job.seed for job in block}) == 1


def test_small_schedule_never_omits_configured_opponent():
    opponents = [
        Opponent("mirror", "a", 0.46), Opponent("alakazam", "b", 0.12),
        Opponent("kangaskhan", "c", 0.06), Opponent("crustle", "d", 0.06),
    ]
    jobs = build_schedule(opponents, blocks=4, base_seed=10, both_seats=True, calibration_blocks=0)
    assert {job.opponent for job in jobs} == {item.label for item in opponents}


def test_agent_spec_can_pin_an_existing_teacher_code():
    assert _parse_agent_spec("grimmsnarl_ml_v8") == ("grimmsnarl_ml_v8", None)
    assert _parse_agent_spec("grimmsnarl_ml_v8@teacher_code=0") == (
        "grimmsnarl_ml_v8", 0
    )


def test_paired_exact_p_uses_only_discordant_pairs():
    assert _paired_exact_p(0, 0) == 1.0
    assert _paired_exact_p(5, 5) == 1.0
    assert _paired_exact_p(10, 0) == pytest.approx(2 / 1024)


def test_paired_summary_uses_seed_clusters():
    rows = []
    # Seed 1: challenger improves both seats. Seed 2: tied both seats.
    for seed, champion_wins, challenger_wins in ((1, [0, 0], [1, 1]), (2, [1, 0], [1, 0])):
        for seat in (0, 1):
            rows.extend(
                [
                    {"repeat": 0, "error": None, "opponent": "x", "seed": seed,
                     "evaluated_seat": seat, "treatment": "champion",
                     "evaluated_win": champion_wins[seat]},
                    {"repeat": 0, "error": None, "opponent": "x", "seed": seed,
                     "evaluated_seat": seat, "treatment": "challenger",
                     "evaluated_win": challenger_wins[seat]},
                ]
            )
    summary = paired_summary(rows)
    assert summary["complete_pairs"] == 4
    assert summary["seed_clusters"] == 2
    assert summary["challenger_minus_champion"] == pytest.approx(0.5)
    assert summary["discordant"]["challenger_only_win"] == 2


@pytest.mark.parametrize(
    ("changed", "verdict"),
    [
        (49, "REJECT_TOO_SMALL"),
        (50, "MEASURE_WITH_2000_PAIRED_GAMES"),
        (200, "MEASURE_WITH_2000_PAIRED_GAMES"),
        (201, "LARGE_ENOUGH_TO_IMPLEMENT"),
    ],
)
def test_policy_impact_gate_boundaries(changed, verdict):
    assert evaluate_impact(100, changed)["verdict"] == verdict
