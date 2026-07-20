# v6 Changelog

- Deck: Genesect -1, Lucky Helmet -1, basic Psychic Energy -1, Boss's Orders +3.
- Added an absolute last-body Dudunsparce ability guard in the authoritative `fallback_v3.py`.
- Reworked Boss's Orders into a same-turn-KO, strategic-value gate.
- Added explicit Team Rocket's Articuno protection-role priority plus generic protector detection.
- Boss now accounts for the one-card Powerful Hand damage loss.
- Boss will not replace an equal or better Active KO with a lower-value Bench KO.
- Boss and Ability remain rule-only; the distilled ML model is unchanged.
- Added v5 ladder analysis artifacts and v6 regression tests.
