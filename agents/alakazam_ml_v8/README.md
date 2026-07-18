# alakazam_ml_v8

v8 is a v7 ladder-driven revision of the Alakazam agent. The default runtime
still uses the deterministic policy for strategic choices; the trained ranker is
shadow-only unless an experiment explicitly sets `ALAKAZAM_ML_V8_ENABLE_OVERRIDE=1`.

## Main changes

- Boss's Orders keeps the reliable same-turn KO rule and adds a narrow two-hit
  route for urgent multi-prize races.
- Kadabra evolves on the Bench when both Active and Bench targets are legal,
  unless evolving the Active creates an immediate KO.
- Current card IDs are used for Team Rocket's Articuno (`414`), Froslass (`104`),
  and Marnie's Grimmsnarl ex (`648`).
- Froslass is a priority engine target, and extra damage-counter-vulnerable draw
  bodies are limited while it is present.
- The deck uses two Dudunsparce and one conditional Shaymin while retaining Max
  Rod. Flower Curtain addresses attack damage to the Bench; it does not prevent
  Froslass damage counters.
- The v7 submission `54811136` is included in the v8 training corpus together
  with the v5 ladder logs and the top-rank Alakazam archives.

See `CHANGELOG_V8.md` and the generated strategy audit stored inside
`data/runs/ml_v7_evaluation/submission_54811136_training.zip` for the evidence.
