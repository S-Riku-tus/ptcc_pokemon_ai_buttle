"""Emit the card-rule tables `ml_features` needs to compute real damage.

`ml_features` is standard-library only and the observation exposes just
`id/hp/maxHp/energies/tools` per Pokemon - no weakness, no ability text. So
the model's damage columns were computed as `0 < hp <= 180`, which is wrong
in three ways the ladder actually punishes:

* Crustle, Sylveon and Cornerstone Mask Ogerpon ex prevent **all** damage from
  a Pokemon ex. Shadow Bullet does 0 to them, but the feature said the Active
  dies whenever it had <= 180 HP.
* A Darkness-weak Active takes 360, so bodies up to 360 HP die and the feature
  said they survive.
* Shadow Bullet's Bench-30 does not land on a benched wall, and Rabsca /
  Shaymin / Neutralization Zone shield the Bench outright.

The rule policy models all of this (`shadow_damage`, `bench_damage_lands`);
the imitation features did not, so the ranker was blind in exactly the spots
that decide games. This script resolves the tables from the vendor card
database once and writes them into `ml_features.py` between the generated
markers, keeping the runtime module import-free and fast.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

BEGIN = "# --- BEGIN GENERATED CARD RULE TABLES ---"
END = "# --- END GENERATED CARD RULE TABLES ---"

# Always present so the tables are correct even if the database is missing a
# skill text; these three are the bodies the archive actually walls us with.
KNOWN_EX_BLOCKERS = {345, 330, 117}   # Crustle, Sylveon, Cornerstone Ogerpon
BENCH_SHIELD_ALL = {74}               # Rabsca: protects the whole Bench
BENCH_SHIELD_NON_RULE_BOX = {343}     # Shaymin: protects non-Rule-Box Bench
NEUTRALIZATION_ZONE = 1247            # zeroes ex damage to non-Rule-Box bodies
BATTLE_CAGE = 1264                    # blocks damage *counters* on the Bench


def ex_damage_blockers(card_table) -> set[int]:
    """Abilities that prevent all damage from our attacker.

    Grimmsnarl ex is a Stage 2 Pokemon ex with an Ability that attacks with
    basic Darkness Energy, so clauses restricted to Basic ex, Tera or Special
    Energy attackers do not apply, and bench-only wording is handled by
    ``bench_damage_lands`` instead. This mirrors
    ``fallback_policy._compute_active_wall_blockers`` exactly.
    """
    blockers = set(KNOWN_EX_BLOCKERS)
    for data in card_table.values():
        for skill in (getattr(data, "skills", None) or []):
            text = getattr(skill, "text", "") or ""
            low = text.lower()
            if "prevent all damage" not in low or "this pok" not in low:
                continue
            if "basic pokémon" in low or "basic pokemon" in low:
                continue
            if "tera" in low or "special energy" in low:
                continue
            if "your bench" in low or "benched pok" in low:
                continue
            if ("{ex}" in text or "pokémon ex" in low or "pokemon ex" in low
                    or "have an ability" in low):
                card_id = getattr(data, "cardId", None)
                if card_id is not None:
                    blockers.add(int(card_id))
    return blockers


def ability_pokemon(card_table, froslass_id: int = 104) -> set[int]:
    """Every Pokemon Freezing Shroud puts a damage counter on.

    v2 counted three card ids - Froslass, Munkidori, Grimmsnarl ex - so the
    "who does Freezing Shroud hurt more" comparison only worked in the mirror.
    The card reads "each Pokemon that has an Ability (both yours and your
    opponent's), except any Froslass", so the real set is every Pokemon card
    with a non-empty skill list, minus the Froslass line itself.
    """
    return {
        int(card_id)
        for card_id, data in card_table.items()
        if getattr(data, "cardType", -1) == 0
        and (getattr(data, "skills", None) or [])
        and int(card_id) != froslass_id
    }


def attack_cost_damage(card_table, attack_table) -> dict[int, tuple]:
    """Pokemon card id -> Pareto frontier of (energy cost, printed damage).

    Used to estimate what the opponent's board can do to us next turn, which
    is what turns "heal 30" into "heal enough to survive". Printed damage only:
    conditional bonuses, damage-counter placement and "for each" scaling are
    not modelled, so this is a floor on the real number for scaling attacks and
    exact for the flat ones that decide most turns.
    """
    out: dict[int, tuple] = {}
    for card_id, data in card_table.items():
        if getattr(data, "cardType", -1) != 0:
            continue
        pairs: list[tuple[int, int]] = []
        for attack_id in (getattr(data, "attacks", None) or []):
            attack = attack_table.get(attack_id)
            if attack is None:
                continue
            damage = int(getattr(attack, "damage", 0) or 0)
            if damage <= 0:
                continue
            cost = len(getattr(attack, "energies", None) or [])
            pairs.append((cost, damage))
        frontier = []
        for cost, damage in sorted(pairs):
            if any(c <= cost and d >= damage for c, d in frontier):
                continue
            frontier.append((cost, damage))
        if frontier:
            out[int(card_id)] = tuple(frontier)
    return out


def type_map(card_table, attribute: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for card_id, data in card_table.items():
        if getattr(data, "cardType", -1) != 0:
            continue
        value = getattr(data, attribute, None)
        if isinstance(value, int) and value:
            out[int(card_id)] = int(value)
    return out


def dict_block(name: str, mapping: dict, comment: str) -> str:
    """Render a mapping as wrapped source, values already repr-able."""
    if not mapping:
        return f"# {comment}\n{name} = {{}}\n"
    tokens = [f"{key}:{value!r}," for key, value in sorted(mapping.items())]
    tokens = [token.replace(" ", "") for token in tokens]
    lines: list[str] = []
    line = "    "
    for token in tokens:
        if len(line) + len(token) > 78:
            lines.append(line.rstrip())
            line = "    "
        line += token
    lines.append(line.rstrip())
    inner = "\n".join(lines)
    return f"# {comment}\n{name} = {{\n{inner}\n}}\n"


def render_v3_extra(card_table, attack_table) -> list[str]:
    return [
        block_frozenset(
            "ABILITY_POKEMON_IDS", sorted(ability_pokemon(card_table)),
            "Every Pokemon with an Ability except the Froslass line: exactly"
            "\n# the bodies Freezing Shroud puts a counter on, both sides.",
        ),
        dict_block(
            "POKEMON_TYPE_IDS", type_map(card_table, "energyType"),
            "Pokemon card id -> its own energy type, for Weakness matching.",
        ),
        dict_block(
            "POKEMON_WEAKNESS_IDS", type_map(card_table, "weakness"),
            "Pokemon card id -> Weakness energy type (doubles damage).",
        ),
        dict_block(
            "POKEMON_RESISTANCE_IDS", type_map(card_table, "resistance"),
            "Pokemon card id -> Resistance energy type (-30 damage).",
        ),
        dict_block(
            "ATTACK_COST_DAMAGE", attack_cost_damage(card_table, attack_table),
            "Pokemon card id -> ((energy cost, printed damage), ...), Pareto"
            "\n# frontier. Printed damage only; scaling attacks read low.",
        ),
    ]


def block_frozenset(name: str, values: list[int], comment: str) -> str:
    if not values:
        return f"# {comment}\n{name} = frozenset()\n"
    lines: list[str] = []
    line = "    "
    for value in values:
        token = f"{value},"
        if len(line) + len(token) + 1 > 79:
            lines.append(line.rstrip())
            line = "    "
        line += token + " "
    lines.append(line.rstrip())
    inner = "\n".join(lines)
    return f"# {comment}\n{name} = frozenset({{\n{inner}\n}})\n"


def render(card_table, energy_dark, attack_table=None,
           variant: str = "v2") -> str:
    blockers = sorted(ex_damage_blockers(card_table))
    rule_box = sorted(
        int(card_id) for card_id, data in card_table.items()
        if getattr(data, "ex", False) or getattr(data, "megaEx", False)
    )
    dark_weak = sorted(
        int(card_id) for card_id, data in card_table.items()
        if getattr(data, "weakness", None) == energy_dark
    )
    dark_resist = sorted(
        int(card_id) for card_id, data in card_table.items()
        if getattr(data, "resistance", None) == energy_dark
    )
    mega_ex = sorted(
        int(card_id) for card_id, data in card_table.items()
        if getattr(data, "megaEx", False)
    )

    def block(name: str, values: list[int], comment: str) -> str:
        if not values:
            return f"# {comment}\n{name} = frozenset()\n"
        body = ", ".join(str(v) for v in values)
        wrapped: list[str] = []
        line = "    "
        for token in body.split(", "):
            piece = token + ","
            if len(line) + len(piece) + 1 > 79:
                wrapped.append(line.rstrip())
                line = "    "
            line += piece + " "
        wrapped.append(line.rstrip())
        inner = "\n".join(wrapped)
        return f"# {comment}\n{name} = frozenset({{\n{inner}\n}})\n"

    parts = [
        BEGIN,
        "# Generated by scripts/build_grimmsnarl_damage_tables.py from the",
        "# vendor card database. Do not edit by hand.",
        "",
        block(
            "EX_DAMAGE_BLOCKER_IDS", blockers,
            "Prevent ALL damage from our Pokemon ex: Shadow Bullet does 0 to "
            "these,\n# in the Active spot and as the Bench-30.",
        ),
        block(
            "BENCH_SHIELD_ALL_IDS", sorted(BENCH_SHIELD_ALL),
            "Shields the opponent's whole Bench from our Bench-30.",
        ),
        block(
            "BENCH_SHIELD_NON_RULE_BOX_IDS", sorted(BENCH_SHIELD_NON_RULE_BOX),
            "Shields the opponent's non-Rule-Box Bench from our Bench-30.",
        ),
        block(
            "RULE_BOX_IDS", rule_box,
            "Pokemon ex / Mega ex: they keep taking damage under "
            "Neutralization Zone\n# and are not covered by Shaymin.",
        ),
        block("MEGA_EX_IDS", mega_ex, "Mega Pokemon ex: three prizes."),
        block(
            "DARK_WEAK_IDS", dark_weak,
            "Weak to Darkness: Shadow Bullet does 360, not 180.",
        ),
        block(
            "DARK_RESIST_IDS", dark_resist,
            "Resistant to Darkness: Shadow Bullet does 150.",
        ),
        f"NEUTRALIZATION_ZONE_ID = {NEUTRALIZATION_ZONE}",
        f"BATTLE_CAGE_ID = {BATTLE_CAGE}",
    ]
    if variant in ("v3", "v4"):
        # v4 needs the same generated tables as v3: the ability set, the
        # attack-cost frontier and the type/weakness maps. Its new work is in
        # the hand-written ledger that reads them, not in the tables.
        parts.append("")
        parts.extend(render_v3_extra(card_table, attack_table or {}))
    parts.append(END)
    return "\n".join(parts) + "\n"


DEFAULT_TARGETS = {
    "v2": ["agents/grimmsnarl/grimmsnarl_ml_v2/ml_features.py"],
    "v3": ["agents/grimmsnarl/grimmsnarl_ml_v3/ml_features.py"],
    "v4": ["agents/grimmsnarl/grimmsnarl_ml_v4/ml_features.py"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["v2", "v3", "v4"], default="v2"
    )
    parser.add_argument("--targets", nargs="*")
    args = parser.parse_args()
    targets = args.targets or DEFAULT_TARGETS[args.variant]

    # policy_base needs to resolve from an agent directory that ships it.
    agent_dir = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v2"
    sys.path.insert(0, str(agent_dir))
    from cg.api import EnergyType  # noqa: E402
    from policy_base import attack_table, card_table  # noqa: E402

    generated = render(
        card_table, EnergyType.DARKNESS,
        attack_table=attack_table, variant=args.variant,
    )
    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", re.DOTALL
    )
    for target in targets:
        path = ROOT / target
        source = path.read_text(encoding="utf-8")
        if BEGIN not in source:
            print(f"{path}: no marker block, skipped", file=sys.stderr)
            return 1
        path.write_text(pattern.sub(generated, source), encoding="utf-8")
        print(f"updated {path}")
    counts = generated.count("frozenset")
    print(f"{counts} tables, {len(card_table)} cards scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
