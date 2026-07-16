# Validation Episode failure fix

## Root cause

The submission `main.py` defined these public functions in this order:

1. `agent(observation)`
2. `diag_reset()`
3. `diag_snapshot()`

Kaggle's Python loader executes the module and uses the last callable in the
module namespace. It therefore selected `diag_snapshot`, not `agent`.
During the initial deck request the selected function either raised a
signature error or returned a diagnostic dictionary rather than a 60-card
list. The CABT environment consequently marked the submission INVALID with
`Player 0's deck does not have 60 cards`, surfaced by Kaggle as
`Validation Episode failed`.

## Fix

- Moved the public `agent(observation)` definition to the end of `main.py`.
- Added an export-time AST guard requiring `main.py` to end with `agent`.
- Added a regression test for the Kaggle entrypoint rule.
- Removed documentation and package-manifest files from the submission ZIP;
  only runtime files remain at the ZIP root.

## Verification

- Old package under Kaggle-compatible loader: INVALID at deck request.
- Fixed package: 10/10 CABT smoke games completed with no import errors,
  invalid actions, or engine rejections.
- Deck count: 60.
- Distilled model loaded successfully.
- All runtime files are at ZIP root.
