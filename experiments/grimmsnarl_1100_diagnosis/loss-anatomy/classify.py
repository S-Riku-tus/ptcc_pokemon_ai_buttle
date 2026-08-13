"""Loss taxonomy over the pooled v15/v19a/v19b/v20/v21 ladder runs."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402


def num(value: str) -> float | None:
    if value in ("", "None", None):
        return None
    return float(value)


def integer(value: str) -> int | None:
    got = num(value)
    return None if got is None else int(got)


def load() -> list[dict[str, Any]]:
    rows = list(csv.DictReader((HERE / "episodes.csv").open(encoding="utf-8-sig")))
    for row in rows:
        row["won"] = row["won"] == "True"
        row["went_first"] = (
            None if row["went_first"] in ("", "None") else row["went_first"] == "True"
        )
        for key in (
            "our_prize_left", "opp_prize_left", "our_deck_left", "opp_deck_left",
            "our_bodies_left", "opp_bodies_left", "our_attacks", "opp_attacks",
            "our_damage", "opp_damage", "our_zero_damage_attacks", "our_kos",
            "opp_kos", "opp_one_shot_kos", "our_one_shot_kos", "first_ready_turn",
            "first_shadow_turn", "first_attack_turn", "opp_first_attack_turn",
            "turns", "our_turns", "our_mulligans", "opp_mulligans",
            "opening_impidimp", "opening_rare_candy", "opening_grim_ex",
            "opening_basics", "opening_dark_energy", "opening_morgrem",
            "our_empty_actions", "our_active_steps", "our_draws", "log_turns",
        ):
            row[key] = integer(row[key])
        row["opponent_rating"] = num(row["opponent_rating"])
        row["our_rating"] = num(row["our_rating"])
        row["min_overage"] = num(row["min_overage"])
    return rows


def loss_class(row: dict[str, Any]) -> str:
    """Primary mechanism, mutually exclusive, terminal condition first.

    Order matters and is chosen so each test is a fact about how the game
    ENDED before any test about how it was played:

    * ``error/timeout``  - engine never reported DONE|DONE, or we were ACTIVE
      with ``minCount>0`` and submitted nothing.
    * ``board-out``      - zero bodies on our side in the final observation.
    * ``deck-out``       - our ``deckCount`` reached 0 with bodies still up.
    * everything below is a prize-out: the opponent took the 6th prize.
    * ``no ready Grimmsnarl ex`` - no Grimmsnarl ex ever carried Shadow
      Bullet's energy cost anywhere on our board.
    * ``walled``         - at least two of our attacks resolved for 0 damage
      and those were at least half of all the attacks we made.
    * ``run over``       - the opponent needed <=4 attacks for all six prizes
      (>=1.5 prizes per swing), i.e. they were one-shotting rule-box bodies.
    * ``prize race``     - the remainder: both sides traded and they got there
      first.
    """
    if row["statuses"] != "DONE|DONE" or row["our_empty_actions"]:
        return "error/timeout"
    if (row["our_bodies_left"] or 0) == 0:
        return "board-out"
    if (row["our_deck_left"] if row["our_deck_left"] is not None else 1) == 0:
        return "deck-out"
    if row["first_ready_turn"] is None:
        return "no ready Grimmsnarl ex"
    zero = row["our_zero_damage_attacks"] or 0
    if zero >= 2 and row["our_attacks"] and zero * 2 >= row["our_attacks"]:
        return "walled (majority of attacks dealt 0)"
    if (row["opp_attacks"] or 99) <= 4:
        return "run over (<=4 opposing attacks for 6 prizes)"
    return "prize race lost while attacking"


def block(rows: list[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    n = len(rows)
    base = total if total is not None else n
    low, high = wilson(n, base) if base else (0.0, 0.0)
    return {
        "n": n,
        "share": round(n / base, 4) if base else None,
        "wilson95": [low, high],
    }


def main() -> int:
    rows = load()
    losses = [r for r in rows if not r["won"]]
    wins = [r for r in rows if r["won"]]
    n, w = len(rows), len(wins)

    out: dict[str, Any] = {}
    out["pool"] = {
        "games": n, "wins": w, "losses": len(losses),
        "win_rate": round(w / n, 4), "wilson95": wilson(w, n),
        "per_version": {
            v: {
                "games": len(g), "wins": sum(1 for r in g if r["won"]),
                "win_rate": round(sum(1 for r in g if r["won"]) / len(g), 4),
                "wilson95": wilson(sum(1 for r in g if r["won"]), len(g)),
            }
            for v, g in sorted(
                _group(rows, lambda r: r["version"]).items()
            )
        },
    }

    by_class = _group(losses, loss_class)
    out["loss_classes"] = {
        name: {
            **block(g, len(losses)),
            "share_of_all_games": round(len(g) / n, 4),
            "mean_prizes_we_took": round(
                sum(6 - (r["our_prize_left"] or 6) for r in g) / len(g), 2
            ),
            "mean_our_attacks": round(
                sum(r["our_attacks"] for r in g) / len(g), 2
            ),
            "mean_our_damage": round(
                sum(r["our_damage"] for r in g) / len(g), 1
            ),
            "mean_turns": round(sum(r["turns"] for r in g) / len(g), 1),
            "went_second": sum(1 for r in g if r["went_first"] is False),
            "families": dict(Counter(r["opponent_family"] for r in g).most_common()),
            "episodes": sorted(int(r["episode_id"]) for r in g),
        }
        for name, g in sorted(by_class.items(), key=lambda kv: -len(kv[1]))
    }

    # --- overlays (non-exclusive) -------------------------------------------
    out["overlays_on_losses"] = {
        "opp_one_shot_kos>=1": block(
            [r for r in losses if (r["opp_one_shot_kos"] or 0) >= 1], len(losses)
        ),
        "opp_one_shot_kos>=2": block(
            [r for r in losses if (r["opp_one_shot_kos"] or 0) >= 2], len(losses)
        ),
        "we_took_0_prizes": block(
            [r for r in losses if (r["our_prize_left"] or 6) == 6], len(losses)
        ),
        "we_took_5_prizes (1 away)": block(
            [r for r in losses if (r["our_prize_left"] or 6) == 1], len(losses)
        ),
        "we_took_>=4_prizes": block(
            [r for r in losses if (r["our_prize_left"] or 6) <= 2], len(losses)
        ),
        "any_zero_damage_attack": block(
            [r for r in losses if (r["our_zero_damage_attacks"] or 0) >= 1],
            len(losses),
        ),
        "never_shadow_bullet": block(
            [r for r in losses if r["first_shadow_turn"] is None], len(losses)
        ),
        "our_deck_left<=3": block(
            [r for r in losses if (r["our_deck_left"] or 99) <= 3], len(losses)
        ),
        "we_mulliganed": block(
            [r for r in losses if (r["our_mulligans"] or 0) >= 1], len(losses)
        ),
        "went_second": block(
            [r for r in losses if r["went_first"] is False], len(losses)
        ),
    }

    # --- coin-flip / pre-decision layer -------------------------------------
    out["pre_decision"] = _pre_decision(rows)

    # --- matchup table ------------------------------------------------------
    out["by_family"] = {
        name: {
            "games": len(g),
            "wins": sum(1 for r in g if r["won"]),
            "win_rate": round(sum(1 for r in g if r["won"]) / len(g), 4),
            "wilson95": wilson(sum(1 for r in g if r["won"]), len(g)),
        }
        for name, g in sorted(
            _group(rows, lambda r: r["opponent_family"]).items(),
            key=lambda kv: -len(kv[1]),
        )
    }
    out["by_family_turn_order"] = {
        f"{name} | {order}": {
            "games": len(g),
            "wins": sum(1 for r in g if r["won"]),
            "win_rate": round(sum(1 for r in g if r["won"]) / len(g), 4),
            "wilson95": wilson(sum(1 for r in g if r["won"]), len(g)),
        }
        for (name, order), g in sorted(
            _group(
                rows,
                lambda r: (
                    r["opponent_family"],
                    {True: "first", False: "second", None: "?"}[r["went_first"]],
                ),
            ).items(),
            key=lambda kv: -len(kv[1]),
        )
    }

    (HERE / "loss_classes.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _group(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return dict(out)


def _pre_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(sub):
        w = sum(1 for r in sub if r["won"])
        return {
            "games": len(sub), "wins": w,
            "win_rate": round(w / len(sub), 4) if sub else None,
            "wilson95": wilson(w, len(sub)) if sub else None,
        }

    out: dict[str, Any] = {}
    out["turn_order"] = {
        "first": rate([r for r in rows if r["went_first"] is True]),
        "second": rate([r for r in rows if r["went_first"] is False]),
        "unknown": rate([r for r in rows if r["went_first"] is None]),
    }
    out["is_first_select_offered"] = {
        "offered": rate([r for r in rows if r["is_first_offered"] == "True"]),
        "not_offered": rate([r for r in rows if r["is_first_offered"] != "True"]),
    }
    out["our_mulligans"] = {
        str(k): rate([r for r in rows if (r["our_mulligans"] or 0) == k])
        for k in sorted({min(r["our_mulligans"] or 0, 3) for r in rows})
    }
    out["opening_hand"] = {
        "impidimp>=1": rate([r for r in rows if (r["opening_impidimp"] or 0) >= 1]),
        "impidimp==0": rate([r for r in rows if (r["opening_impidimp"] or 0) == 0]),
        "impidimp==0 and candy==0": rate(
            [r for r in rows
             if (r["opening_impidimp"] or 0) == 0
             and (r["opening_rare_candy"] or 0) == 0]
        ),
        "no_dark_energy": rate(
            [r for r in rows if (r["opening_dark_energy"] or 0) == 0]
        ),
        "basics==1": rate([r for r in rows if (r["opening_basics"] or 0) == 1]),
        "basics>=3": rate([r for r in rows if (r["opening_basics"] or 0) >= 3]),
    }
    out["opening_hand_x_turn_order"] = {
        f"impidimp{'>=1' if has else '==0'} | {'first' if first else 'second'}": rate(
            [r for r in rows
             if ((r["opening_impidimp"] or 0) >= 1) == has
             and r["went_first"] is first]
        )
        for has in (True, False) for first in (True, False)
    }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
