"""Emit the card-rule tables the Dragapult features need to compute real damage.

The observation exposes only ``id/hp/maxHp/energies/tools`` per Pokemon: no
weakness, no prize value, no ability text, no attack costs.  So a feature that
wants to say "this Phantom Dive knocks the Active out" or "these 6 damage
counters take two prizes" has to carry resolved card data.

This is the Grimmsnarl ``build_grimmsnarl_damage_tables.py`` idea with the
deck-specific parts removed: nothing here mentions Darkness or Shadow Bullet,
so the emitted block is a plain description of the card database and the
feature module decides what to do with it.  Regenerate whenever the vendor
card database changes.

Usage:
  python scripts/build_dragapult_card_tables.py \
      --targets agents/dragapult/dragapult_ml_v2/ml_features.py
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

# Bodies that prevent all damage from a Pokemon ex. Dragapult ex is one, so
# Phantom Dive's 200 does nothing to these in the Active spot. They are listed
# explicitly as well as text-matched because a missing skill text here turns
# an unwinnable attack into a feature that says "lethal".
KNOWN_EX_BLOCKERS = {345, 330, 117}   # Crustle, Sylveon, Cornerstone Ogerpon
BENCH_SHIELD_ALL = {74}               # Rabsca: shields the whole Bench
BENCH_SHIELD_NON_RULE_BOX = {343}     # Shaymin: shields the non-Rule-Box Bench
NEUTRALIZATION_ZONE = 1247
BATTLE_CAGE = 1264                    # blocks damage *counters* on the Bench


def ex_damage_blockers(card_table) -> set[int]:
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


def ability_pokemon(card_table) -> set[int]:
    return {
        int(card_id)
        for card_id, data in card_table.items()
        if getattr(data, "cardType", -1) == 0
        and (getattr(data, "skills", None) or [])
    }


def attack_cost_damage(card_table, attack_table) -> dict[int, tuple]:
    """Pokemon card id -> Pareto frontier of (energy cost, printed damage).

    Printed damage only, so "for each" scaling and damage-counter placement
    read low. It is a floor on what a body can do, which is the number that
    decides whether one Adrena-Brain move is enough to survive a turn.
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
            pairs.append((len(getattr(attack, "energies", None) or []), damage))
        frontier: list[tuple[int, int]] = []
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


def hp_map(card_table) -> dict[int, int]:
    out: dict[int, int] = {}
    for card_id, data in card_table.items():
        if getattr(data, "cardType", -1) != 0:
            continue
        value = getattr(data, "hp", None)
        if isinstance(value, int) and value:
            out[int(card_id)] = int(value)
    return out


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
    return f"# {comment}\n{name} = frozenset({{\n" + "\n".join(lines) + "\n})\n"


def dict_block(name: str, mapping: dict, comment: str) -> str:
    if not mapping:
        return f"# {comment}\n{name} = {{}}\n"
    tokens = [f"{key}:{value!r},".replace(" ", "") for key, value in sorted(mapping.items())]
    lines: list[str] = []
    line = "    "
    for token in tokens:
        if len(line) + len(token) > 78:
            lines.append(line.rstrip())
            line = "    "
        line += token
    lines.append(line.rstrip())
    return f"# {comment}\n{name} = {{\n" + "\n".join(lines) + "\n}\n"


def render(card_table, attack_table) -> str:
    rule_box = sorted(
        int(card_id) for card_id, data in card_table.items()
        if getattr(data, "ex", False) or getattr(data, "megaEx", False)
    )
    mega_ex = sorted(
        int(card_id) for card_id, data in card_table.items()
        if getattr(data, "megaEx", False)
    )
    parts = [
        BEGIN,
        "# Generated by scripts/build_dragapult_card_tables.py from the vendor",
        "# card database. Do not edit by hand.",
        "",
        block_frozenset(
            "EX_DAMAGE_BLOCKER_IDS", sorted(ex_damage_blockers(card_table)),
            "Prevent all damage from a Pokemon ex. Dragapult ex is one, so"
            "\n# Phantom Dive's 200 does nothing to these in the Active spot.",
        ),
        block_frozenset(
            "BENCH_SHIELD_ALL_IDS", sorted(BENCH_SHIELD_ALL),
            "Shields the opponent's whole Bench from attack damage.",
        ),
        block_frozenset(
            "BENCH_SHIELD_NON_RULE_BOX_IDS", sorted(BENCH_SHIELD_NON_RULE_BOX),
            "Shields the opponent's non-Rule-Box Bench from attack damage.",
        ),
        block_frozenset(
            "RULE_BOX_IDS", rule_box,
            "Pokemon ex / Mega ex: two or three prizes, and not covered by"
            "\n# Shaymin or Neutralization Zone.",
        ),
        block_frozenset("MEGA_EX_IDS", mega_ex, "Mega Pokemon ex: three prizes."),
        block_frozenset(
            "ABILITY_POKEMON_IDS", sorted(ability_pokemon(card_table)),
            "Every Pokemon card that has at least one Ability.",
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
            "POKEMON_MAX_HP", hp_map(card_table),
            "Pokemon card id -> printed HP, so a benched body's remaining HP"
            "\n# can be read as a fraction even before it is damaged.",
        ),
        dict_block(
            "ATTACK_COST_DAMAGE", attack_cost_damage(card_table, attack_table),
            "Pokemon card id -> ((energy cost, printed damage), ...), Pareto"
            "\n# frontier. Printed damage only; scaling attacks read low.",
        ),
        f"NEUTRALIZATION_ZONE_ID = {NEUTRALIZATION_ZONE}",
        f"BATTLE_CAGE_ID = {BATTLE_CAGE}",
        END,
    ]
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="*",
        default=["agents/dragapult/dragapult_ml_v2/ml_features.py"],
    )
    args = parser.parse_args()

    agent_dir = ROOT / "agents" / "dragapult" / "dragapult_ml_v2"
    sys.path.insert(0, str(agent_dir))
    sys.path.insert(0, str(ROOT / "agents" / "_base"))
    from policy_base import attack_table, card_table  # noqa: E402

    generated = render(card_table, attack_table)
    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", re.DOTALL
    )
    for target in args.targets:
        path = ROOT / target
        source = path.read_text(encoding="utf-8")
        if BEGIN not in source:
            print(f"{path}: no marker block, skipped", file=sys.stderr)
            return 1
        path.write_text(pattern.sub(generated, source), encoding="utf-8")
        print(f"updated {path}")
    print(f"{len(card_table)} cards scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
