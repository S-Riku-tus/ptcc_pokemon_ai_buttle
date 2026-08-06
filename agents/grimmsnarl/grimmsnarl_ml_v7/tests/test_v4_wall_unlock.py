"""The v4 additions: the wall-unlock rule and the corrected shroud ledger.

Every case is built from the shapes the ladder measurement isolated, and each
asserts the *narrowness* as well as the firing, because a dominance rule that
fires one board wider than its proof is how a shell starts costing more than it
saves (the Alakazam line lost 5.16 points of agreement that way).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import ml_features as mf  # noqa: E402
from ml_planner import Planner  # noqa: E402

CRUSTLE_ID = 345          # prevents all damage from a Pokemon ex
KANGASKHAN_ID = 1  # stand-in benched body Shadow Bullet can damage


def _dark(count: int) -> list[dict[str, int]]:
    return [{"id": mf.DARK_ENERGY_ID} for _ in range(count)]


def _board(
    *,
    bench: list[dict],
    active_id: int = CRUSTLE_ID,
    active_hp: float = 100.0,
    stadium: int | None = None,
    our_energy: int = mf.SHADOW_BULLET_COST,
    our_active_id: int = mf.GRIMMSNARL_EX_ID,
) -> dict:
    current = {
        "turn": 5,
        "yourIndex": 0,
        "players": [
            {
                "active": [{
                    "id": our_active_id,
                    "hp": 340.0,
                    "maxHp": 340.0,
                    "energies": _dark(our_energy),
                }],
                "bench": [],
                "hand": [],
                "prize": [{}] * 4,
            },
            {
                "active": [{
                    "id": active_id, "hp": active_hp, "maxHp": active_hp,
                }],
                "bench": bench,
                "prize": [{}] * 4,
            },
        ],
    }
    if stadium is not None:
        current["stadium"] = {"id": stadium}
    return {"current": current}


ATTACK = {"type": 13, "attackId": mf.SHADOW_BULLET_ID}
BOSS = {"type": 7, "area": mf.AREA_HAND, "index": 0}
END = {"type": 14}


def _select(options: list[dict]) -> dict:
    return {"context": mf.MAIN_CONTEXT, "option": options}


def _hand(observation: dict, cards: list[dict]) -> dict:
    observation["current"]["players"][0]["hand"] = cards
    return observation


def test_fires_when_gust_unlocks_the_wall() -> None:
    """Crustle Active, a damageable body benched, Boss in hand."""
    planner = Planner()
    observation = _hand(
        _board(bench=[{"id": KANGASKHAN_ID, "hp": 340.0, "maxHp": 340.0}]),
        [{"id": mf.BOSS_ID}],
    )
    select = _select([ATTACK, BOSS, END])
    assert planner.adjust(observation, select, 0) == 1
    assert planner.stats["wall_unlock_overrides"] == 1


def test_stands_down_without_boss() -> None:
    """The swing is worthless, but there is no gust: the teachers swing here
    88.6% of the time and so should we."""
    planner = Planner()
    observation = _board(
        bench=[{"id": KANGASKHAN_ID, "hp": 340.0, "maxHp": 340.0}]
    )
    select = _select([ATTACK, END])
    assert planner.adjust(observation, select, 0) == 0
    assert planner.stats["wall_unlock_overrides"] == 0


def test_stands_down_when_the_active_is_damageable() -> None:
    planner = Planner()
    observation = _hand(
        _board(
            active_id=KANGASKHAN_ID, active_hp=200.0,
            bench=[{"id": KANGASKHAN_ID, "hp": 340.0, "maxHp": 340.0}],
        ),
        [{"id": mf.BOSS_ID}],
    )
    select = _select([ATTACK, BOSS, END])
    assert planner.adjust(observation, select, 0) == 0
    assert planner.stats["wall_unlock_considered"] == 0


def test_stands_down_when_the_bench_30_takes_a_prize() -> None:
    """A Bench-30 kill is a prize in hand; trading it for damage is a
    preference, not a dominance, so the ranker keeps it."""
    planner = Planner()
    observation = _hand(
        _board(bench=[{"id": KANGASKHAN_ID, "hp": 20.0, "maxHp": 340.0}]),
        [{"id": mf.BOSS_ID}],
    )
    select = _select([ATTACK, BOSS, END])
    assert planner.adjust(observation, select, 0) == 0
    assert planner.stats["wall_unlock_overrides"] == 0


def test_stands_down_when_the_whole_bench_is_walled() -> None:
    planner = Planner()
    observation = _hand(
        _board(bench=[{"id": CRUSTLE_ID, "hp": 100.0, "maxHp": 100.0}]),
        [{"id": mf.BOSS_ID}],
    )
    select = _select([ATTACK, BOSS, END])
    assert planner.adjust(observation, select, 0) == 0
    assert planner.stats["wall_unlock_overrides"] == 0


def test_stands_down_when_our_active_cannot_swing_after_the_gust() -> None:
    """Retreating to find an attacker costs energy this deck has no spare of,
    so an unfuelled Active means the gust does not convert this turn."""
    planner = Planner()
    observation = _hand(
        _board(
            bench=[{"id": KANGASKHAN_ID, "hp": 340.0, "maxHp": 340.0}],
            our_energy=mf.SHADOW_BULLET_COST - 1,
        ),
        [{"id": mf.BOSS_ID}],
    )
    select = _select([ATTACK, BOSS, END])
    assert planner.adjust(observation, select, 0) == 0


def test_fires_on_end_too() -> None:
    """The trigger that matters. Probing v3 over its own 65 games: it never
    picked the swing at a decision where the gust was also offered, so a rule
    keyed on the attack alone fires never - the same way v3's Boss rule did."""
    planner = Planner()
    observation = _hand(
        _board(bench=[{"id": KANGASKHAN_ID, "hp": 340.0, "maxHp": 340.0}]),
        [{"id": mf.BOSS_ID}],
    )
    select = _select([ATTACK, BOSS, END])
    assert planner.adjust(observation, select, 2) == 1
    assert planner.stats["wall_unlock_overrides"] == 1


def test_leaves_other_actions_alone() -> None:
    """Only the two turn-closing picks are dominated; a bench play is not."""
    planner = Planner()
    observation = _hand(
        _board(bench=[{"id": KANGASKHAN_ID, "hp": 340.0, "maxHp": 340.0}]),
        [{"id": mf.BOSS_ID}],
    )
    bench_play = {"type": 7, "area": mf.AREA_HAND, "index": 1}
    select = _select([ATTACK, BOSS, bench_play, END])
    assert planner.adjust(observation, select, 2) == 2
    assert planner.stats["wall_unlock_considered"] == 0


# ----- the corrected shroud ledger -------------------------------------------

def _side(cards: list[dict]) -> dict:
    return {"active": cards[:1], "bench": cards[1:]}


def test_battle_cage_blocks_their_bench_not_ours() -> None:
    """Battle Cage prevents damage counters on Benched Pokemon from *the
    opponent's* Abilities, so our Froslass is stopped on their Bench and not on
    our own. v3 counted both benches regardless of the stadium."""
    side = _side([
        {"id": mf.MUNKIDORI_ID, "hp": 70.0, "maxHp": 70.0},
        {"id": mf.MUNKIDORI_ID, "hp": 70.0, "maxHp": 70.0},
        {"id": mf.GRIMMSNARL_EX_ID, "hp": 340.0, "maxHp": 340.0},
    ])
    open_field = mf.shroud_side(side, -1, is_own_side=False)
    caged_theirs = mf.shroud_side(side, mf.BATTLE_CAGE_ID, is_own_side=False)
    caged_ours = mf.shroud_side(side, mf.BATTLE_CAGE_ID, is_own_side=True)
    assert len(open_field) == 3
    assert len(caged_theirs) == 1
    assert len(caged_ours) == 3
    # Stadium-blind agrees with the open field and overstates the caged one.
    assert len(mf.shroud_targets(mf._in_play(side))) == len(open_field)


def test_shroud_ledger_turns_negative_under_battle_cage() -> None:
    """The whole point: the same board flips from favourable to a net loss."""
    def net(stadium: int | None) -> int:
        current = {
            "turn": 5, "yourIndex": 0,
            "players": [
                _side([
                    {"id": mf.MUNKIDORI_ID, "hp": 70.0, "maxHp": 70.0},
                    {"id": mf.MUNKIDORI_ID, "hp": 70.0, "maxHp": 70.0},
                ]) | {"hand": [], "prize": [{}] * 4},
                _side([
                    {"id": mf.MUNKIDORI_ID, "hp": 70.0, "maxHp": 70.0},
                    {"id": mf.MUNKIDORI_ID, "hp": 70.0, "maxHp": 70.0},
                    {"id": mf.MUNKIDORI_ID, "hp": 70.0, "maxHp": 70.0},
                ]) | {"prize": [{}] * 4},
            ],
        }
        if stadium is not None:
            current["stadium"] = {"id": stadium}
        return int(mf.state_features(current)["shroud_net"])

    assert net(None) == 1          # 3 of theirs against 2 of ours
    assert net(mf.BATTLE_CAGE_ID) == -1   # only their Active; both of ours


@pytest.mark.parametrize(
    ("hp", "expected"), [(10.0, 1), (20.0, 2), (70.0, 7), (340.0, 9)]
)
def test_shroud_checkups_to_kill(hp: float, expected: int) -> None:
    assert mf.shroud_checkups_to_kill({"hp": hp}) == expected
