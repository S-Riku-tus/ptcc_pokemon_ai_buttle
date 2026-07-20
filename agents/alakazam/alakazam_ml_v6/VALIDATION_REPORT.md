# v4 candidate logic-audit validation report

## Deck validation

- Total: 60 cards
- Dunsparce: 3
- Dudunsparce: 3
- Psyduck: 0
- Basic Psychic Energy: 3
- Maximum non-basic copies: 4
- ACE SPEC: Maximum Rod only
- Deck SHA-256: `a79d4317d6dfcafefc397c9ee88066afc13b1f65a9dda0392d48461f3f97225e`

## Model validation

- Model retrained: no
- Model SHA-256: `63e68716c9a3dcfbb638dd994255135c1f95cc65a370c651ac56ecfbcc42f231`
- Confidence/margin thresholds: unchanged
- Runtime ML scope: `guarded_intent_preserving_v2`

## Logic changes validated

- full evolution/search hand deltas
- exact search deck cost and target-dependent backup result
- exact search lethal attribution
- Fezandipiti core-slot reservation
- no proactive Telepath fetch of Fezandipiti
- Fezandipiti draw usefulness/deck-safety gate
- Genesect immediate Helmet/core-slot gate
- immediate-effect Nighttime Mine gate
- role-aware survival Bench selection
- practical KO-clock preservation for optional role spends
- fallback Bench/evolution intent preservation in ML
- unusable Supporters excluded from reachable-hand estimate

## Automated validation

- Python compilation: passed
- Test suite: **29 passed**
- Randomized synthetic policy states: **1,000/1,000 returned legal option indices**
- Randomized exceptions: **0**
- Duplicate function definitions in modified runtime files: none
- Model load under the local API-compatible stub: passed
- `main.py` deck return: 60 cards

## Packaging validation

- caches and bytecode excluded
- runtime ZIP contains only submission runtime files
- full ZIP contains runtime, tests, metadata, and audit documents
- archive integrity checked after creation

## Limitation

The official `cg` package, real battle harness, Kaggle Validation Episode, and ladder execution were not available here. The local API-compatible stub validates policy control flow but cannot prove compatibility with every engine-side effect. Run repository static validation and a small real-engine smoke match before submission, followed by a seat-swapped Champion–Challenger evaluation.
