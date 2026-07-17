# Alakazam ML v2 expanded — guarded runtime

This agent retains the exact Majkel-style 60-card deck and the existing
`fallback_v12.py` policy. The distilled model is unchanged.

## Runtime responsibility split

`fallback_v12` exclusively decides strategic or irreversible actions:

- Dudunsparce and Fezandipiti ex abilities
- all end-turn decisions
- trainers, energy attachments, Boss, Retreat, Xerosic, and Hammer
- Fezandipiti ex and Shaymin deployment
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

The submitted model is still the existing distilled LightGBM model. Retraining
with the new ladder trajectories must be performed separately; this runtime
change does not claim a new trained policy.
