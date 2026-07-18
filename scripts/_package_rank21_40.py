"""Package the collected top21-40 Alakazam battle logs into per-rank ZIPs.

Reuses scripts.package_submission_replays.package_submission (layout="run")
to build the same internal structure as the 20260717_kaggle_top20 archives,
then merges every matched submission of a given rank into a single
`rankNN_<team>_full.zip` file.
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.package_submission_replays import package_submission

ROOT = REPO_ROOT / "data" / "runs" / "20260718_kaggle_top21_40_alakazam"


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def matched_submissions() -> dict[int, list[tuple[int, str]]]:
    groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for d in sorted((ROOT / "submissions").glob("*")):
        if not d.is_dir():
            continue
        ridx = d / "replays" / "index.json"
        if not ridx.exists():
            continue
        idx = json.loads(ridx.read_text(encoding="utf-8"))
        if not any(r.get("download_status") == "success" for r in idx):
            continue
        meta = json.loads((d / "submission.json").read_text(encoding="utf-8"))
        rank = int(meta.get("leaderboard_rank") or 0)
        team = str(meta.get("team_name") or "")
        groups[rank].append((int(d.name), team))
    return dict(sorted(groups.items()))


def main() -> None:
    groups = matched_submissions()
    print(f"ranks with matched Alakazam logs: {list(groups)}")
    grand = {"ranks": 0, "zips": 0, "submissions": 0, "replays": 0, "logs": 0, "bytes": 0}
    for rank, subs in groups.items():
        team = subs[0][1]
        rank_zip = ROOT / f"rank{rank:02d}_{safe_name(team)}_full.zip"
        with zipfile.ZipFile(rank_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
            for sid, tname in subs:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
                    tmp = Path(tf.name)
                info = package_submission(
                    source_root=ROOT.resolve(),
                    submission_id=sid,
                    output_zip=tmp,
                    rank=rank,
                    team_name=tname,
                    limit=None,
                    layout="run",
                )
                with zipfile.ZipFile(tmp) as zin:
                    for name in zin.namelist():
                        out.writestr(name, zin.read(name))
                tmp.unlink()
                grand["submissions"] += 1
                grand["replays"] += info["replay_count"]
                grand["logs"] += info["log_file_count"]
                print(
                    f"  rank{rank:02d} sub{sid}: "
                    f"{info['replay_count']} replays / {info['log_file_count']} log files"
                )
        size = rank_zip.stat().st_size
        grand["ranks"] += 1
        grand["zips"] += 1
        grand["bytes"] += size
        print(f"-> {rank_zip.name}  ({size / 1_048_576:.1f} MiB)")
    print("\n=== summary ===")
    print(json.dumps(grand, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
