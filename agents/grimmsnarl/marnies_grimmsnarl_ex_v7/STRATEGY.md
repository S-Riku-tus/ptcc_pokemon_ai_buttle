# Marnie's Grimmsnarl ex v7 Strategy

## Objective

Convert every Shadow Bullet into the largest reachable prize route while keeping
one live attacker and one *genuinely completable* backup, never wasting a turn
hitting a wall for 0, and using Boss's Orders whenever a gust has a real purpose.
The deck list is fixed so v6→v7 results isolate policy quality.

## The engine (unchanged core)

Grimmsnarl ex evolves with Punk Up (attach up to 5 basic Darkness to Marnie's
Pokémon) and swings 180 Active + 30 to a Benched Pokémon. Munkidori's Adrena-Brain
moves damage; Froslass chips Ability-holders each Checkup. The plan is
*multi-prize turns*: line up the Bench-30, Adrena-Brain and Boss so one Shadow
Bullet takes two or three prizes.

## Kept fixed from v6

- **Damage-immune walls ("locked").** Crustle / Sylveon (any ex), Cornerstone Mask
  Ogerpon ex (any Ability — Grimmsnarl has Punk Up), and Neutralization Zone vs
  non-Rule-Box. A 0-damage Shadow Bullet ranks **below END** unless its Bench-30
  (± Adrena-Brain) takes a prize this turn; a benched wall is never a Bench-30
  target. Boss's Orders is the unlock.
- **Meta target priority** for Bench-30 / Adrena-Brain / Boss / gust: anti-Grimm
  tech (Shaymin/Rabsca, walls) → main Pokémon (ex/mega by prizes, then draw/damage
  engines) → pre-evolutions → other.

## v7 fundamentals

### Boss's Orders by purpose

`best_boss_value()` scores each bench gust by purpose, each with its own bar, and
sets `self._boss_mode`:

1. **WIN_NOW** — the gusted KO takes our remaining prize(s). Always top.
2. **WALL_UNLOCK** — the Active is a wall; move a hittable body into range.
3. **HIGHER_PRIZE_KO** — KO a benched body worth more prizes than the Active.
4. **ENGINE_KO** — KO a key engine / tech / sole attacker at ≥ equal prizes,
   unless that throws away a *confirmed* ≥-prize Active KO this turn.
5. **TEMPO_GUST** — strand a hard-to-retreat body we cannot KO to buy a turn, only
   when the Active route is not itself a KO this turn.

A confirmed / near-confirmed high-prize Active KO route is protected: only WIN_NOW
and WALL_UNLOCK bypass the "beat continuing to attack the current Active" guard.
The old flat `>= 10_000` gate is gone, so engine/tempo gusts finally fire.

### Honest attacker ETAs

`first_attacker_eta` = whole self-turns to a Shadow Bullet that **actually
happens**: a ready Grimmsnarl is Active, or one evolves in the Active spot, or a
benched ready one can be promoted by retreat (the only switch this deck runs).
It refuses ETA 0 when the evolve base appeared this turn or the finished attacker
cannot legally reach the Active.

`backup_attacker_eta` = a distinct SECOND attacker; it need not reach the Active
this turn but must be a real route. A lone Morgrem with two Energy and no
Grimmsnarl ex in hand is **not** ETA 0. `backup_is_close` = ETA ≤ 1 (≤ 0 in a
fast race).

### Fast race (Mega Lucario / Archaludon)

`fast_race()` (a Mega ex on board, or a Riolu/Lucario/Duraludon/Archaludon line)
switches gears: flood Impidimp early so a single KO cannot wipe the line, hold
optional engines (2nd Munkidori / Froslass) until the first attacker is live, and
require a completed backup. Detection is name/megaEx based — deliberately not the
DB's name-unioned prize-potential map, which collides.

### Initial Active

Impidimp > (Munkidori that can retreat / has a spare) > (Snorunt with its Froslass
route) > a sole Munkidori with no escape. A lone board still force-develops a Basic.

### Temporary immunity

A coin Dodge/Hide attack that prevents all damage during our next turn zeroes that
Pokémon (`temp_immune`, by serial) for the turn, folded into `shadow_damage`,
`active_target_immune_to_ex` and `bench_damage_lands`. Best-effort and fully
defensive: an unknown log schema leaves it inert.

## Action tiers

1. **EMERGENCY** — lone board: develop a Basic.
2. **BUILD_ATTACKER** — complete the first Grimmsnarl and its two Energy.
3. **PRE_ATTACK_SAFE** — profitable Unfair Stamp, useful second Munkidori,
   Adrena-Brain, exact recovery.
4. **BUILD_BACKUP** — a genuinely completable second attacker.
5. **ATTACK / BOSS+ATTACK** — Shadow Bullet with the best Bench-30 route, or a
   purposeful Boss gust followed by the attack.
6. **OPTIONAL_SETUP** — held until the first attacker is live (harder in a fast race).

## Fixed behavior

- Go first; keep the 60-card list unchanged.
- Punk Up: current attacker to two Energy, then a distinct backup to two Energy.
- Never treat a damage-immune / dodged Active as taking 180 from Grimmsnarl ex.
