# Alakazam replay imitation pipeline

This directory is isolated from the existing ladder agents. It audits every available top-team
bundle, trains only on episodes that contain the acting observation, legal options, and recorded
action, and exports a dependency-free hybrid runtime.

## Reproduce

Run from the repository root with the system Python that provides pandas, PyArrow, LightGBM,
scikit-learn, and NumPy:

```powershell
python -m ml_alakazam.src.run_pipeline
python -m pytest ml_alakazam/tests -q
```

Raw ZIP files are read directly and never modified or extracted in place. Generated datasets,
models, reports, and packages live below this directory.
