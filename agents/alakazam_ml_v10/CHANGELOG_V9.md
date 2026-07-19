# v9 changelog

## Evidence used

- v8 went 26-21 in the reconstructed 47-game public sample, but averaged 1.51
  non-attacking turns after its first attack and fell to 7-8 in the mirror.
- In three v8 Mist games, only about one third of Hammer selections hit Mist.
  Majkel's 1,000-game sample selected Mist first in 171 of 184 Mist games.
- Fezandipiti ex became Active in 12 v8 games. It received no Energy, made no
  attacks, and produced seven long Active stalls.
- Boss connected to an attack or KO on 74.1% of v8 plays versus 89.2% in v7.
- Dual-target Kadabra evolution was absolute in both versions: v7 chose Active
  35/35, while v8 chose Bench 34/34. The top reference chose Bench 120/129.

Source: `data/runs/ml_v8_evaluation/ML_v6_v8_top_comparison_report.md` and
the CSV/JSON files under `data/runs/ml_v8_evaluation/results/`.

## Runtime changes

### Enhanced Hammer

- Resolve attached Energy options through `energyIndex`; v8 accidentally scored
  the owning Pokemon instead of the selected Energy in this context.
- Rank Mist and other effect-prevention Energy above ordinary Special Energy.
- Infer Mist likelihood from visible Crustle and Hop Trevenant signatures, with
  a medium prior for Mega Starmie variants.
- Reserve the fourth Hammer when three are already discarded and Mist likelihood
  is at least 30%. The reservation is released for a visible Mist/Rock-Fighting
  Energy, an immediate attack denial, or a Grow Grass Energy KO.

### Fezandipiti ex

- Add four explicit modes: `DO_NOT_BENCH`, `DRAW_ONLY`, `PIVOT`, and
  `ALTERNATE_ATTACKER`.
- Do not expose a new two-prizer when the opponent has two prizes remaining, or
  at three to four prizes when a powered Active has a confirmed KO route.
- Preserve Bench slots for the first Alakazam line and one Dudunsparce line.
- When Fezandipiti is stranded Active with a ready Benched Alakazam, attach one
  Energy for retreat instead of leaving it unfunded.
- Develop Cruel Arrow only when its completion ETA is at most two attachments
  and the matchup is Spidops/protection-heavy or a concrete 100-damage prize is
  available. Cruel Arrow target selection now values KOs and prize closure.
- Track consecutive Active turns and expose stalls beyond two own turns.

### Evolution and Boss

- Keep Bench Kadabra as the default dual-target evolution, but use Active when
  it is the only immediate attack, produces a KO, or prevents a fragile Active
  line from being stranded.
- Keep all same-turn Boss KO rules. A two-hit route now requires a target that
  closes the prize race, cannot currently retreat, and is a three-prizer,
  protection engine, or powered main attacker.

## ML and deck

- The v8 deck is byte-for-byte unchanged.
- `ranker_model.json` is byte-for-byte unchanged (SHA-256
  `e9a47de0bf27e9a5528eb09b846f9e63a530e1e8ab6c142eaf62faba5a3cdba3`).
- The ranker remains shadow-only by default. v9 does not claim a model promotion
  without action-specific offline and live gates.

## Promotion targets

- At least 5.2 attacks and 4.0 Alakazam attacks per game.
- At most 0.5 non-attacking turns per game after the first attack.
- Deck-out and board-out loss rates each at most 5%.
- First Hammer targets Mist in at least 85% of Mist matchups.
- No unexplained Fezandipiti Active stall beyond two own turns.
