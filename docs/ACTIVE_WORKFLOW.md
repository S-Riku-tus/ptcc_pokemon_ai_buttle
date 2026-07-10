# Active Workflow

## Current Focus

The active competitive line is Alakazam:

- `agents/alakazam741_v1`
- `agents/alakazam741_v2`

Retired deck lines live under `archive/agents/` and are ignored by Git. Keep them for local reference and regression comparisons, but do not submit from there by default.

## New Candidate Flow

```powershell
.\.venv\Scripts\python.exe .\scripts\new_agent.py alakazam741_v2 alakazam741_v3
git switch -c feature/alakazam741-v3
```

Before comparing on ladder, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_agent.py --agent alakazam741_v3
.\.venv\Scripts\python.exe .\scripts\local_arena.py alakazam741_v3 alakazam741_v2 --games 40
.\.venv\Scripts\python.exe .\scripts\local_arena.py alakazam741_v3 alakazam741_v1 --games 40
.\.venv\Scripts\python.exe -m pytest
```

## Log Storage

Fetch ladder logs into `data/runs/` with a run name and deck snapshot:

```powershell
.\.venv\Scripts\python.exe .\scripts\fetch_submission_logs.py `
  --submission <SUBMISSION_ID> `
  --run-name alakazam741_v3 `
  --deck-name "Alakazam v3" `
  --deck-dir .\agents\alakazam741_v3 `
  --sleep 0 `
  --zip
```

`data/runs/`, `data/submissions/`, and generated summaries are local-only and ignored by Git.
