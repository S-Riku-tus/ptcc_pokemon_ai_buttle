# Marnie's Grimmsnarl ex v6 Strategy

## Objective

Convert every Shadow Bullet into the largest reachable prize route while preserving
one current attacker and one near-term backup, and never waste a turn hitting a wall
for 0. The deck list is fixed so v5→v6 results isolate policy quality.

## The engine (unchanged core)

Grimmsnarl ex evolves with Punk Up (attach up to 5 basic Darkness to Marnie's
Pokémon) and swings for 180 Active + 30 to a Benched Pokémon. Munkidori's
Adrena-Brain moves damage onto a target, and Froslass chips Ability-holders each
Checkup. The plan is not "180 every turn" — it is *multi-prize turns*: line up the
Bench-30 and Adrena-Brain so one Shadow Bullet takes two or three prizes.

## v6 fundamentals

### Damage-immune walls ("locked")

An opposing Active prevents all Shadow Bullet damage when it is Crustle / Sylveon
(any ex), **Cornerstone Mask Ogerpon ex (any Ability — Grimmsnarl has Punk Up)**, or
any Pokémon under Neutralization Zone that lacks a Rule Box. In that state:

- The Shadow Bullet is **not** a live attack: development, backup building and
  searches are no longer held below it.
- **Boss's Orders is the unlock** — gust a reachable Bench Pokémon into the Active to
  turn 0 into 180.
- The 0-damage Shadow Bullet ranks **below END** unless its Bench-30 (± a ready
  Adrena-Brain) takes a prize this turn, so we stop hammering the wall.
- A benched wall is never a Bench-30 target (its Ability prevents that 30 too).
- If a powered Morgrem is Active, its Corkscrew Punch (60, no Ability / not an ex) is
  a *real* attack into the wall and outranks the 0.

### Attacker ETA and the T3 opening

`first_attacker_eta` measures whole self-turns to the first live Shadow Bullet
(0 = this turn, Punk Up powers a fresh evolution or a Rare-Candy skip; 1 = next
turn; 2 = partial line; 99 = nothing). While it is ≥ 2 with no attacker, digging for
the line beats optional engine setup (second Munkidori, Froslass, Handheld Fan). The
single biggest win correlator in the log is attacking by T3.

`backup_attacker_eta ≤ 1` is the bar for a *sufficient* backup — merely having an
Impidimp or Morgrem is not enough.

### Initial Active and no board wipes

The opening Active prefers **Impidimp > Munkidori > Snorunt** (ladder data: Impidimp
starts attack fastest at 2.42, Snorunt slowest at 3.91, because a Snorunt Active must
both clear itself and complete the attacker). With only one Pokémon in play we force
a Basic onto the Bench before anything else, so a single KO cannot wipe us.

## Action tiers

1. **EMERGENCY** — with a lone board, develop a Basic (avoid a board wipe).
2. **BUILD_ATTACKER** — complete the first Grimmsnarl ex and its two Energy (T3 focus).
3. **PRE_ATTACK_SAFE** — profitable Unfair Stamp, useful second Munkidori,
   Adrena-Brain, exact recovery — actions that preserve the attack.
4. **BUILD_BACKUP** — complete a distinct second attacker (`backup_attacker_eta ≤ 1`).
5. **ATTACK** — Shadow Bullet with the best reachable Bench-30 route (never a wall for 0).
6. **OPTIONAL_SETUP** — third line, redundant search, extra Froslass, board filling.

## Target priority (Bench-30 / Adrena-Brain / Boss / gust)

1. **Anti-Grimmsnarl tech** — Shaymin / Rabsca (they lock our Bench-30) and walls
   (Crustle line, Sylveon, Ogerpon).
2. **The opponent's main Pokémon** — high-prize ex/mega attackers and draw/damage
   engines (Munkidori, Fezandipiti, Dudunsparce, Alakazam, Kilowattrel).
3. **Pre-evolutions of main Pokémon** — Abra/Kadabra, Riolu, Duraludon, Dunsparce,
   Staryu, Cynthia's Gible line, etc. (detected dynamically from the evolution chain
   and boosted when the evolved form is already on board).
4. **Everything else.**

Within a tier, current HP, prize count, and a one-turn Bullet+Brain KO route decide.

## Boss's Orders (two-turn value)

Boss is valued as *two-turn prizes*: it will not gust a 1-prize chip target off a
2-3 prize Active that a two-attack Shadow Bullet already KOs (the Makuhita-over-Mega
Lucario mistake), and it will not replace an equal-or-better Active KO. It *will*
unlock a wall or gust up a KO-able engine.

## Fixed behavior

- Go first; keep the 60-card list unchanged.
- Punk Up: current attacker to two Energy, then a distinct backup to two Energy.
- Do not treat a damage-immune Active (Crustle / Sylveon / Ogerpon / Neutralization
  Zone) as taking 180 from Grimmsnarl ex.
