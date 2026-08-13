# Grimmsnarl imitation v8 experiments

This directory targets strict, leakage-free Top-1 imitation of a current
high-rated same-deck pilot.  The primary gate is chronological held-out
accuracy for team `16452116`; pooled accuracy is retained as a secondary
diagnostic because the deployed runtime uses one pinned teacher at a time.

The production candidate is not changed until a model clears the accuracy and
runtime gates documented here.
