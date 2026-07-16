# Alakazam ML v2 expanded hybrid

This agent keeps the existing `fallback_v12.py` policy and uses the expanded legal-option
ranker only for high-confidence ACTIVE MAIN decisions.

Safety routing:

- Boss, Retreat, Xerosic, and Hammer always use the fallback.
- Energy uses ML only at probability >= 0.85 and margin >= 0.12.
- Nested target/search selections and multi-select decisions use the fallback.
- A fallback-confirmed immediate KO is never overridden.
- The distilled runtime has no LightGBM/NumPy/pandas dependency.

The model was trained on 95,254 decisions and 1,097,481 legal candidates from 20 submissions,
19 teams, and 8 Alakazam deck clusters. Actual Kaggle Rating improvement is not claimed until
official-engine ladder validation.
