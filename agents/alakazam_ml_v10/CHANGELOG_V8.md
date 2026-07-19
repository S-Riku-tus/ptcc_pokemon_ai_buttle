# v8 changelog

## v7 ladder evidence

- Audited all 46 replays from submission `54811136`: 24 wins and 22 losses.
- In all 35 turns where both Active and Bench Abra could evolve into Kadabra,
  v7 selected the Active target. The rank-1 reference selected the Bench target
  in 120 of 129 comparable decisions.
- Found two strict two-turn multi-prize Boss opportunities. v7 played Boss in
  both turns but selected a one-prize target and lost both games.
- Found three Marnie's Grimmsnarl games and no wins. The 30 bench damage comes
  from Grimmsnarl's attack; repeated 10-damage increments are Froslass damage
  counters during Pokemon Checkup.
- Corrected stale card ID `675`: current replays identify it as Lunatone, while
  Team Rocket's Articuno is `414`.

## Runtime

- Prefer Bench Kadabra development when both target areas are legal, preserving
  Active evolution only for an immediate Kadabra KO.
- Add a conservative two-hit Boss route for urgent multi-prize races. It never
  gives up a game-winning Active KO and rejects ordinary early-game two-prize
  pulls.
- Add Froslass removal value and reduce extra Dunsparce development into its
  repeated damage counters.
- Replace the third Dudunsparce with one Shaymin. Flower Curtain is deployed
  only when the opponent shows attack-based bench damage; Max Rod is retained.

## ML

- Add Froslass, Grimmsnarl, Kadabra target-area, target-damage, and conditional
  Shaymin interaction features.
- Add the 46 v7 ladder replays to the v8 training corpus.
- Rebuilt 6,608 usable trajectories into 323,889 decisions and 3,736,551
  candidate rows. Time-holdout top-1 improved from 0.60284 to 0.60689; Bench
  top-1 improved from 0.72337 to 0.72451.
- Keep the model shadow-only by default; strategic actions remain rule-owned.
