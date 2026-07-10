# PTCG AI Battle Agent

Repository for Pokemon TCG AI Battle Challenge agents.

The current development focus is the Alakazam line:

- `agents/alakazam741_v1`: strong baseline from the first Alakazam ladder run.
- `agents/alakazam741_v2`: current candidate. It fixes several v1 ladder issues, but should still be compared against v1 before becoming the only active line.

Other deck lines are kept locally under `archive/agents/` for reference and regression checks. The archive directory is intentionally ignored by Git.

## Layout

```text
agents/
  alakazam741_v1/
  alakazam741_v2/
  _base/
  _opponents/
archive/
  agents/                  # local-only retired deck lines
  legacy_v0/               # local-only older baseline
data/
  logs/                    # local-only generated logs
  replays/                 # local-only replay downloads
  runs/                    # local-only experiment runs
  submissions/             # local-only submission logs
  summaries/               # local-only generated analysis
docs/
experiments/
kaggle/
scripts/
tests/
```

Only source code, strategies, small experiment records, docs, and directory keepers should be tracked. Generated Kaggle logs, replays, ZIP files, and archived agents stay local.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pytest kaggle
```

The full `requirements-dev.txt` may require extra native build tools on Windows because `kaggle-environments` can pull heavier dependencies.

## Validate

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_agent.py --agent alakazam741_v2
.\.venv\Scripts\python.exe -m pytest
```

## Local Battles

Use the lightweight local arena when `vendor/cg` is available:

```powershell
.\.venv\Scripts\python.exe .\scripts\local_arena.py alakazam741_v2 alakazam741_v1 --games 40
.\.venv\Scripts\python.exe .\scripts\local_arena.py alakazam741_v2 generic:grimmsnarl --games 80
```

Archived agents can still be used by path or by name because `local_arena.py` checks `archive/agents/` after `agents/`.

## Build a Submission

```powershell
.\.venv\Scripts\python.exe .\scripts\build_submission.py --agent alakazam741_v2
```

On Kaggle, use:

```powershell
python kaggle/create_submission_from_git.py
```

That helper currently builds submissions for:

- `alakazam741_v1`
- `alakazam741_v2`

## Fetch Ladder Logs

Always fetch logs into `data/runs/` with a run name and deck snapshot:

```powershell
.\.venv\Scripts\python.exe .\scripts\fetch_submission_logs.py `
  --submission 54523210 `
  --run-name alakazam741_v2 `
  --deck-name "Alakazam v2" `
  --deck-dir .\agents\alakazam741_v2 `
  --sleep 0 `
  --zip
```

Each run writes:

- `run_meta.json`
- `episodes.csv`
- `manifest.csv`
- `deck_snapshot/`
- per-episode replay JSON and extracted observation logs
- optional ZIP archive

These outputs are ignored by Git.
