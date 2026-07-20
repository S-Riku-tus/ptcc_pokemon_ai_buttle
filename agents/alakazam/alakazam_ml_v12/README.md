# alakazam_ml_v12

v12 is a narrow policy branch from `alakazam_ml_v11`. The 60-card deck, ML
model, ML features, and shadow-only model authority are unchanged. Only the two
requested tactical areas are modified: Abra's Teleport destination and Team
Rocket's Articuno counterplay.

## Abra Teleport

The engine reports an attack-effect switch as `SWITCH` with Abra (`741`) in the
effect field. v12 uses that signal to keep this selection separate from normal
KO promotion and retreat selection.

With multiple choices, the order is Dunsparce, Dudunsparce, spare Abra, then a
Fezandipiti ex that is unlikely to be Knocked Out next turn. The safety check
uses only visible state and allows the opponent's next Energy attachment.
Kadabra and either Alakazam are last-resort mandatory choices. If only one
Bench Pokemon is legal, the normal `minCount=1` rule still selects it.

## Team Rocket's Articuno

Powerful Hand places damage counters, so Articuno prevents it only against
Basic Team Rocket Pokemon. Team Rocket Evolutions and non-Team-Rocket Pokemon
remain valid Powerful Hand targets.

v12 follows two branches:

1. If an unprotected Bench Pokemon can be Knocked Out now, Boss's Orders moves
   it Active and Alakazam attacks it.
2. If every visible opponent is Articuno-protected, v12 builds a damage-based
   attacker. Fezandipiti ex is primary because Cruel Arrow can hit Articuno on
   the Bench; Dudunsparce, Dunsparce, and Shaymin are progressively weaker
   fallbacks. Once ready, the policy brings that attacker Active and does not
   use Powerful Hand or zero-damage Trading Places into the lock.

The complete-lock condition is intentionally strict so the normal v11 Energy
discipline remains unchanged in other matchups.

See `CHANGELOG_V12.md` and `VALIDATION_REPORT_V12.md`.
