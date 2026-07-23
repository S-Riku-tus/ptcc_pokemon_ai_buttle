# v13 ladder analysis and v14 design

## Evidence set

- v13 submission: `54868871`
- supplied rating: `807.1`
- audited saved games: `52`
- reference behavior: v11, especially its last-action attack fallback

The audit reads decoded observations and the option actually selected on every
decision. It does not infer choices from final board state alone.

## Confirmed failure modes

| Failure | Observed count | Root cause |
|---|---:|---|
| END while Powerful Hand was offered | 36 turns | v13 added an explicit `-1` score for protected Powerful Hand; v11 used `500` |
| Of those, Mist Energy | 21 turns | effect lock converted the attack from final action to rejected action |
| Of those, Rock Fighting Energy | 6 turns | same effect-lock branch |
| Of those, global protector | 9 turns | same effect-lock branch |
| Another attachment chosen while Active attacker could be fueled | 11 turns | attachment scores did not represent the shared once-per-turn resource |
| Specifically, Rich diverted Active Alakazam fuel | 6 turns | Rich-cycle score `24800` beat Alakazam fuel score near `8200` |
| Boss offered but protected Active left in place | 3 turns | low-value Basic KO filter ran even when current attack did zero |
| Non-Mist Hammer after Mist had been public | 12 uses | only the literal fourth Hammer was reserved |

The three Boss failures were turns 18, 20, and 22 of episode `87216813`.
The opponent's Active Trevenant carried Mist, while two unprotected 70 HP
Basics were on the Bench. Boss was legal, and the post-Boss hand supported a KO.

## Design decisions

### 1. Model the attachment opportunity cost

Energy attachment is a once-per-turn resource. Therefore an attachment cannot
be scored only by its destination value. If Psychic fuel makes the Active
Alakazam attack now, any other attachment also has the hidden cost of losing an
entire attack. v14 makes that route authoritative and blocks Rich for the turn.

### 2. Separate target quality from attack completion

Powerful Hand into protection does zero, so Hammer and Boss must outrank it.
But the attack itself is harmless and costs no card. v14 restores the v11 score
of 500 so it happens only after all productive actions, never instead of them.
An explicit END invariant protects this behavior from future score changes.

### 3. Let Boss optimize against the real alternative

The normal alternative to a low-value Boss target is often a better Active KO,
so the existing anti-waste filter is sound. Under effect lock, however, the
alternative is zero progress. v14 admits a low-value target only when it is
unprotected and is a concrete same-turn KO; ordinary Boss restrictions remain.

### 4. Give Hammer persistent matchup memory

Once Mist is observed, the matchup has demonstrated the exact card Hammer was
included to answer. v14 reserves every remaining Hammer while the public-copy,
archetype-aware estimate remains at least 0.30. Reservation releases when the
estimated package is exhausted, and any currently attached prevention Energy
is removed immediately.

## Scope deliberately unchanged

- Deck list and ACE SPEC choice
- v13 Rich-cycle draw/deck-floor rules outside attack-route conflicts
- v13 Boss protection against abandoning a superior Active KO
- v13 Articuno scope and Fezandipiti alternate-attacker logic
- Shadow-only ML policy and model weights
