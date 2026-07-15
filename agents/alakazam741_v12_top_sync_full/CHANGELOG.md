# CHANGELOG — alakazam741_v12_top_sync

## v12.0.0

### Deck

- Rebuilt the list from the current rank-one replay bundle and kept it at exactly 60 cards.
- Added Enriching Energy 1, Boss's Orders 3, and Nighttime Mine 2.
- Adjusted Basic Psychic Energy to 2, Dudunsparce to 2, Rare Candy to 3, and Night Stretcher to 1.
- Removed the previous ACE SPEC Item and Battle Cage. Enriching Energy is the only ACE SPEC.
- Kept all other v11 counts unchanged.

### Tempo and development

- Kept `go_first()` fixed to first.
- Split opening Abra bodies, complete attack routes, and `backup_eta` into separate goals.
- Set the opening plan to three Abra bodies plus one Dunsparce without automatically filling the
  remaining bench slots with Shaymin or Fezandipiti ex.
- Made Rare Candy beat Kadabra when it creates the first same-turn Alakazam attack; later lines
  retain Kadabra's evolution draw preference.
- Added a hard optional-search stop after an Active KO is secured with `backup_eta <= 1`, plus an
  18-card overdraw guard.
- Prevented Enriching Energy's draw attachment from consuming a Psychic attachment that creates
  the first/current Alakazam attack.

### Card-specific decisions

- Implemented Enriching Energy as one Colorless provision, draw-four hand attachment, and a
  high-value Dunsparce → Dudunsparce recycling resource.
- Added Dunsparce handoff/redeployment around the last live Dudunsparce cycle.
- Split Fezandipiti ex into `DRAW_ONLY` and `ALTERNATE_ATTACKER`. The draw mode gets no energy;
  the attack mode requires effect lock, a concrete 100-damage target, and deterministic energy
  access before any partial funding.
- Added strict Boss target scoring for KO, prizes, protection engines, main attackers, scarce
  evolution lines, current-Active opportunity cost, and the post-spend Powerful Hand threshold.
- Added a Boss commit guard: after Boss resolves, every non-attack MAIN action is blocked.
- Reworked Enhanced Hammer around effect-removing energy, actual one- or two-energy attack
  deficits, active tempo, and follow-up denial. Target value survives the Hammer sub-selection.
- Allowed non-mirror Xerosic only when a meaningful attack remains, the opponent has at least six
  cards and a hand-dependent board, Boss is worse, deckout is distant, and the state is not locked.
- Restricted Shaymin to a payable opposing Active bench-damage attack that would really KO a
  protected non-rule-box bench Pokémon.
- Added Nighttime Mine only for a real Tera tax or a valuable opposing-Stadium overwrite; it never
  pretends to extend the deck and never spends away a current KO.
- Reduced Night Stretcher handling to one-copy, before/after direct improvement checks.

### Safety and validation

- Preserved attack reservation, attackable-END blocking, zero-damage Powerful Hand blocking,
  last-body Dudunsparce safety, retreat safety, PrizeTracker, legal fallbacks, and `backup_eta`.
- Preserved startup checks for 60 cards, known IDs, the four-copy rule, and at most one ACE SPEC.
- Kept `policy_base.py` byte-identical to v11; all new behavior is deck-specific in `main.py`.
- Added 42 API-state tests, including all 20 requested Golden states and resolution-time Boss and
  Hammer regressions.

