# Marnie’s Grimmsnarl ex v5 Strategy

## Objective

Convert every Shadow Bullet into the largest reachable prize route while preserving one current attacker and one near-term backup. The deck list remains fixed so that v4→v5 results isolate policy quality.

## v5 fundamentals (new)

### Damage-immune walls ("locked")

Some opposing Actives prevent all damage from our Pokémon ex — Crustle and
Sylveon by Ability, and Neutralization Zone against non-Rule-Box Pokémon. Shadow
Bullet then does 0 to that Active (its Bench-30 can still land). In this state:

- The Shadow Bullet is **not** a "live attack": pre-attack development, backup
  building and searches are no longer held below it.
- **Boss's Orders is the unlock.** Gusting any reachable Bench Pokémon into the
  Active — even one we cannot KO this turn — turns 0 into 180 and is valued above
  the wall attack.
- The 0-damage Shadow Bullet ranks last among real actions (only the Bench-30
  gives it any value), so we stop hammering the wall.

### Evolve on the Bench, not into a grave

Basic TCG play: build the next attacker where it survives, not where it is fed to
the opponent.

- **Morgrem** has no Punk Up and needs two Energy already attached to attack, so
  it is only evolved in the Active when it can attack **this turn**; otherwise a
  Bench body is preferred, especially when the opponent's Active can KO it.
- **Grimmsnarl ex** normally attacks the turn it evolves (Punk Up fuels it), so
  the Active preference is kept — **except** into a wall it cannot damage, where
  the Bench copy is built so Boss's Orders can later open a real target.

## Action tiers

1. **BUILD_ATTACKER** — complete the first Grimmsnarl ex and its two Energy.
2. **PRE_ATTACK_SAFE** — actions that preserve the attack and have immediate value: profitable Unfair Stamp, useful second Munkidori, Adrena-Brain, and exact recovery.
3. **BUILD_BACKUP** — complete a distinct second attacker through Morgrem or Rare Candy.
4. **ATTACK** — Shadow Bullet, including the best reachable Bench-30 route.
5. **OPTIONAL_SETUP** — third line, redundant search, extra Froslass, or low-impact board filling.

## Unfair Stamp

Stamp is not triggered by the opponent hand alone. It is promoted before Shadow Bullet when it refills a small self-hand, strips a very large opposing hand, or specifically disrupts Alakazam without sacrificing a unique backup route. A large, useful self-hand is protected.

## Munkidori

The first Munkidori is a core engine. A second is played before Shadow Bullet only when it has an immediate relocation/prize route and does not consume a bench slot reserved for the Grimmsnarl line. Third copies remain optional.

## Rare Candy and search

Impidimp + Rare Candy + Grimmsnarl ex is a complete route. Poké Pad and Spikemuth Gym must not search Morgrem in that state. Candy completing the second attacker is allowed before the live attack.

## Prize planning

Target ranking combines current HP, prize count, route-piece value, Shadow Bullet 30, and one powered Adrena-Brain. A reachable two-prize target can outrank an isolated one-prize chip target.

## Fixed behavior

- Go first.
- Keep the 60-card list unchanged.
- Punk Up prioritizes current attacker to two Energy, then a distinct backup to two Energy.
- Do not treat a damage-immune Active (Crustle / Sylveon / Neutralization Zone) as taking 180 from Grimmsnarl ex.
- Do not replace an equal-or-better Active KO with Boss’s Orders.
