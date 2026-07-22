# Marnie's Grimmsnarl ex v6

Rule-based Kaggle agent. v6 keeps v5's route-aware sequencing and target model and
fixes the errors that remained in the 723.3-rating ladder log (17-15), plus the two
the pilot called out directly.

1. **Complete wall detection.** Grimmsnarl ex is our only Shadow Bullet attacker and
   it is *both* a Pokémon ex *and* a Pokémon with an Ability (Punk Up). Cornerstone
   Mask Ogerpon ex blocks "Pokémon that have an Ability", so it walls us just like
   Crustle/Sylveon — but v5 only checked the *ex* wording and hit it for 0. v6
   recognises both the ex clause and the ability clause (and Neutralization Zone),
   and never treats a benched wall as a Bench-30 target.
2. **Stop hammering walls.** A 0-damage Shadow Bullet now ranks *below ending the
   turn* unless its Bench-30 (optionally + a ready Adrena-Brain) takes a prize this
   turn. Otherwise we keep developing and look for the Boss's Orders unlock.
3. **Two-turn Boss's Orders.** Boss will not gust a low-prize chip target away from a
   high-prize Active that a two-attack Shadow Bullet already KOs, but it *will*
   remove a key engine or unlock a wall.
4. **Meta targeting.** Bench-30 / Adrena-Brain / Boss / gust targets follow a fixed
   ranking: anti-Grimmsnarl tech (Shaymin bench-lock, walls) → the opponent's main
   Pokémon (high-prize ex/mega, draw/damage engines) → their pre-evolutions →
   everything else.
5. **Faster, safer opening.** Explicit `first_attacker_eta` / `backup_attacker_eta`
   focus T3 completion of the first Grimmsnarl over optional engine setup, the
   initial Active prefers Impidimp > Munkidori > Snorunt, and a lone board
   force-develops a Basic so a single KO cannot wipe us.

The 60-card list is **unchanged** from v4/v5 so v5→v6 results isolate policy quality.

## Submission payload

- `main.py`
- `policy_base.py`
- `deck.csv`

The full package also contains strategy, changelog, metadata, tests, and validation notes.
