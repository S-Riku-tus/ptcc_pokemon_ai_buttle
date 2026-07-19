# v4 candidate second logic audit

## Executive conclusion

The revised v4 had the correct broad direction, but still contained several cross-system contradictions. The most important were not matchup-specific: they were accounting and priority errors that could suppress a strong Alakazam turn or let ML replace the deterministic plan with a different role.

This pass keeps the requested deck and Fezandipiti design, but makes the agent more internally consistent with the attack-first behavior that made v3 strong.

## 1. Evolution was incorrectly treated as a hand loss

Powerful Hand depends directly on hand size. The previous attack-preservation gate treated every evolution as `-1` card, even though:

- Kadabra: play 1, draw 2, net +1
- Alakazam: play 1, draw 3, net +2
- Rare Candy + Alakazam: spend 2, draw 3, net +1

That could block an evolution as though it reduced damage even when it actually increased damage. The calculation now models the full effect.

## 2. Generic search was credited with outcomes it did not create

Previously, a missing backup could let a search action bypass deckout protection without proving that the searched cards completed a near-term attacker. A Buddy-Buddy Poffin body, for example, is not automatically an ETA-1 backup.

Each search now has:

- exact deck cost,
- exact net hand delta,
- legal target set,
- hypothetical `backup_eta`, and
- exact lethal contribution.

Only a result that reduces backup ETA to 1 or less receives the backup safety exception.

## 3. Fezandipiti's early deployment conflicted with core Bench capacity

The user-requested policy is to Bench Fezandipiti early and keep it as a draw engine. The naive form of that rule could occupy the final slot needed for Abra, Dunsparce, or a real backup line.

The revised predicate therefore keeps the early-deployment default, but reserves capacity for:

- the first Abra attacker line,
- one Dunsparce/Dudunsparce engine body, and
- a real backup attacker when the current Alakazam is already ready.

Fezandipiti is not proactively searched by Telepath Energy. It is a naturally drawn support body, not a setup objective.

## 4. Optional support could worsen the prize clock

Current-KO protection alone was too narrow. Playing a role card from a five-card hand can reduce Powerful Hand from 100 to 80. Against 200 HP, that changes a two-hit line into a three-hit line even though neither attack is a current KO.

For optional role spends—Fezandipiti, Genesect, Lucky Helmet, Nighttime Mine, and extra Buddy-Buddy Poffin—the attack gate now also preserves the practical number of hits required to KO the Active.

This is intentionally narrower than blocking every hand spend. Evolution, direct recovery, and meaningful disruption can still be worth a temporary damage trade.

## 5. Survival Bench logic ignored card roles

When only one Pokémon remained, every Basic previously received essentially the same survival score. Depending on option order, the policy could Bench Fezandipiti or Genesect before Abra.

The revised order is:

1. Abra — restores the attack plan
2. Dunsparce — restores the reusable draw engine
3. Fezandipiti ex — preserves future post-KO recovery
4. Genesect — emergency body, higher only when Helmet lock is immediately available

Any legal body still outranks accepting a board-out, but role quality now matters.

## 6. Genesect and Nighttime Mine were still too speculative

Genesect now requires Lucky Helmet in hand and enough Bench capacity after reserving the core plan. Nighttime Mine now requires the opposing Active to be Tera and the additional Colorless cost to make its currently payable attack unpayable.

This prevents spending a hand card for a nominal interaction that has no immediate game effect.

## 7. ML could change the fallback's strategic objective

Although ML was already restricted to low-risk action types, it could still replace:

- fallback Abra Bench with Dunsparce Bench, or
- fallback Alakazam evolution with Dudunsparce evolution.

Those are not equivalent choices. The first can delay the attacker; the second can trade attack completion for draw-engine development.

ML now ranks only inside the exact fallback intent:

- same Bench card ID,
- same evolution card ID, or
- attack candidates when fallback chose attack.

This intentionally lowers ML freedom. The prior 200-game comparison showed that the model changed only a small fraction of all decisions, so protecting deterministic strategic intent has more expected value than allowing broad low-confidence substitutions.

## 8. Reachable damage was optimistic

Dawn and Hilda were counted as future hand gain whenever present, even if all relevant search targets were gone. They now contribute to reachable damage only when the search has a concrete current goal.

## What was deliberately not changed

- The model was not retrained.
- Fezandipiti remains an early persistent Bench support card, per the user's direction.
- The 3/3 Dunsparce/Dudunsparce split remains.
- No Boss's Orders or alternative attacker was added.
- The general attack-reservation, legal fallback, deckout, and last-body protections remain.

## Remaining limitations

This codebase is still substantially more complex than v3. The explicit safety gates improve local correctness, but every extra branch increases interaction risk. The correct next evaluation is therefore not another large logic expansion. It is an ablation-based match test separating:

1. revised deck + deterministic fallback,
2. revised deck + deterministic fallback + ML,
3. v3 champion.

The official `cg` package and real battle harness were not available in this environment, so this pass validates syntax, isolated policy states, ML scope, metadata/deck invariants, and randomized legal-return robustness—not actual ladder strength.
