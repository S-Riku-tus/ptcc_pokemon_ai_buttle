# Alakazam ML v4 candidate

This Challenger preserves `alakazam_ml_v3` as the Champion and applies the
user-reviewed ladder fixes in the deterministic fallback. Its distilled model
was retrained from the existing top-50 corpus plus the latest 40 episodes from
v2 submission `54644578` and v3 submission `54681320`.

## Deck delta from v3

- remove: Shaymin x1, Boss's Orders x3, Enriching Energy x1
- add: Dunsparce x1, Genesect x1, Psyduck x1, Lucky Helmet x1, Max Rod x1
- total: 60 cards; Max Rod is the only ACE SPEC

## New hard rules

- A Psychic attachment that enables the Active Alakazam attacks is a survival-tier action;
  once the attack exists, END remains blocked.
- Fezandipiti ex is established on the Bench from the midgame and its draw Ability is used
  after an opposing-turn KO unless doing so loses the deckout race.
- With only one Pokémon in play, a playable Basic is benched before an ordinary attack/end.
- Search cards are blocked when no role-relevant target remains in the deck.
- Genesect is a Bench-only ACE lock and receives Lucky Helmet before the opponent's ACE is seen.
- Psyduck is benched only against a visible self-KO Ability, except for last-body survival.
- Max Rod is held for a critical recovery or at least two useful Pokémon/basic Energy returns.

## Runtime responsibility split

`fallback_v12` exclusively decides strategic or irreversible actions:

- Dudunsparce and Fezandipiti ex abilities
- all end-turn decisions
- trainers, energy attachments, Retreat, Xerosic, and Hammer
- Fezandipiti ex, Genesect, and Psyduck deployment
- nested target/search and multi-select decisions
- fallback-confirmed immediate KOs

ML is restricted to high-confidence ranking of low-risk board-construction
choices after fallback has already entered the same safe scope:

- benching Abra or Dunsparce
- evolution
- attack choice only when fallback also selected an attack

This guarded scope addresses the first ladder run, where all 92 draw-ability
uses were ML decisions and several high-confidence Dudunsparce uses occurred
with only 4–6 cards left in deck.

The retrained model hash is recorded in `metadata.json`. Promotion still
requires Champion–Challenger evidence; this directory is not an automatic
replacement.
