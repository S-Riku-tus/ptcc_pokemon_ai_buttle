"""Census of the three field corpora + Grimmsnarl pilot ranking.

Writes a single JSON blob to census.json and prints the tables.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ml.core.replay_io import deck_hash, extract_fast_header_from_file  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family, archetype, wilson  # noqa: E402

OUR = "9714ab5c3996f6cc"
OUT = Path(__file__).resolve().parent

CORPORA = {
    "kaggle_top50_meta": ROOT / "data" / "kaggle_top50_meta",
    "kaggle_top100_current": ROOT / "data" / "kaggle_top100_current",
    "kaggle_grimmsnarl_top50": ROOT / "data" / "kaggle_grimmsnarl_top50",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    corpus_stats = {}
    # episode_id -> record
    episodes: dict[int, dict] = {}
    # submission_id -> {team_id, team_name, rating, rank, corpus}
    subs: dict[str, dict] = {}

    for name, root in CORPORA.items():
        srows = read_csv(root / "indexes" / "submissions.csv")
        erows = read_csv(root / "indexes" / "episodes.csv")
        replay_files = sorted((root / "replays").glob("episode_*.json"))
        # top100_current has no index CSVs; recover from submission.json
        if not srows:
            for d in sorted((root / "submissions").iterdir()):
                f = d / "submission.json"
                if not f.exists():
                    continue
                s = json.loads(f.read_text(encoding="utf-8"))
                srows.append({
                    "leaderboard_rank": s.get("leaderboard_rank", ""),
                    "team_id": s.get("team_id", ""),
                    "team_name": s.get("team_name", ""),
                    "submission_id": str(s.get("submission_id", "")),
                    "submission_score": s.get("leaderboard_score", ""),
                    "submitted_at_utc": (s.get("source_rows") or [{}])[0].get(
                        "representative_submitted_at_utc", ""),
                    "downloaded_at": s.get("collected_at", ""),
                    "episode_count": "0",
                })

        ratings = []
        for r in srows:
            sid = str(r.get("submission_id") or "")
            try:
                sc = float(r.get("submission_score") or "nan")
            except ValueError:
                sc = float("nan")
            if not math.isnan(sc):
                ratings.append(sc)
            if sid:
                prev = subs.get(sid)
                rec = {
                    "submission_id": sid,
                    "team_id": str(r.get("team_id") or ""),
                    "team_name": r.get("team_name") or "",
                    "rating": None if math.isnan(sc) else sc,
                    "rank_at_fetch": r.get("leaderboard_rank") or "",
                    "corpora": sorted({name} | set((prev or {}).get("corpora", []))),
                    "submitted_at_utc": r.get("submitted_at_utc") or "",
                    "downloaded_at": r.get("downloaded_at") or "",
                }
                if prev:
                    rec["rating"] = prev["rating"] if prev["rating"] is not None else rec["rating"]
                subs[sid] = rec

        created = [r["created_at"] for r in erows if r.get("created_at")]
        dl = [r["downloaded_at"] for r in srows if r.get("downloaded_at")]
        ratings.sort()

        def q(p):
            if not ratings:
                return None
            i = min(len(ratings) - 1, int(round(p * (len(ratings) - 1))))
            return ratings[i]

        corpus_stats[name] = {
            "submissions": len(srows),
            "episode_index_rows": len(erows),
            "replay_files_on_disk": len(replay_files),
            "distinct_episodes_in_index": len({r["episode_id"] for r in erows if r.get("episode_id")}),
            "rating_n": len(ratings),
            "rating_min": ratings[0] if ratings else None,
            "rating_p25": q(0.25), "rating_median": q(0.5), "rating_p75": q(0.75),
            "rating_max": ratings[-1] if ratings else None,
            "rating_ge_1050": sum(1 for x in ratings if x >= 1050),
            "rating_ge_1100": sum(1 for x in ratings if x >= 1100),
            "rating_ge_1150": sum(1 for x in ratings if x >= 1150),
            "episode_created_min": min(created) if created else None,
            "episode_created_max": max(created) if created else None,
            "downloaded_min": min(dl) if dl else None,
            "downloaded_max": max(dl) if dl else None,
        }

        # build episode table
        for r in erows:
            if r.get("download_status") != "success":
                continue
            eid = int(r["episode_id"])
            path = root / "replays" / f"episode_{eid}.json"
            if not path.exists():
                continue
            rec = episodes.setdefault(eid, {
                "episode_id": eid, "corpus": name,
                "type": r.get("episode_type"), "created_at": r.get("created_at"),
                "path": str(path),
                "sub": [str(r.get("agent_0_submission_id") or ""),
                        str(r.get("agent_1_submission_id") or "")],
            })
            rec.setdefault("owners", set()).add(str(r.get("submission_id") or ""))

    # parse headers once per episode
    print(f"parsing {len(episodes)} replay headers ...", file=sys.stderr)
    for i, rec in enumerate(episodes.values()):
        try:
            h = extract_fast_header_from_file(rec["path"])
        except Exception:  # noqa: BLE001
            rec["ok"] = False
            continue
        decks = h["decks"]
        if len(decks) < 2 or not decks[0] or not decks[1]:
            rec["ok"] = False
            continue
        rec["ok"] = True
        rec["hash"] = [deck_hash(decks[0]), deck_hash(decks[1])]
        rec["family"] = [family(decks[0]), family(decks[1])]
        rec["arch"] = [archetype(decks[0]), archetype(decks[1])]
        rec["decks"] = decks
        rec["rewards"] = h["rewards"]
        rec["teams"] = h["team_names"]
        if i % 500 == 0:
            print(f"  {i}", file=sys.stderr)

    good = [r for r in episodes.values() if r.get("ok")]
    json.dump({"corpus_stats": corpus_stats}, (OUT / "corpus_stats.json").open("w", encoding="utf-8"), indent=2)

    # ---- deck hash census per submission (from the replays themselves) ----
    sub_hash = defaultdict(Counter)
    sub_family = defaultdict(Counter)
    sub_deck = {}
    for rec in good:
        for seat in (0, 1):
            sid = rec["sub"][seat]
            if not sid:
                continue
            sub_hash[sid][rec["hash"][seat]] += 1
            sub_family[sid][rec["family"][seat]] += 1
            sub_deck.setdefault((sid, rec["hash"][seat]), rec["decks"][seat])

    our_counts = Counter(sub_deck[k] and 0 for k in [])  # noop
    # our reference deck (any replay with our hash)
    our_deck = None
    for (sid, h), d in sub_deck.items():
        if h == OUR:
            our_deck = d
            break

    def dist(deck):
        a, b = Counter(deck), Counter(our_deck)
        return sum((a - b).values())

    pilots = {}
    for sid, hc in sub_hash.items():
        top_hash, n = hc.most_common(1)[0]
        fam = sub_family[sid].most_common(1)[0][0]
        d = sub_deck[(sid, top_hash)]
        pilots[sid] = {
            "submission_id": sid,
            "deck_hash": top_hash,
            "games_seen": sum(hc.values()),
            "family": fam,
            "archetype": archetype(d),
            "cards_diff_from_ours": dist(d),
            "is_our_hash": top_hash == OUR,
            **{k: v for k, v in (subs.get(sid) or {}).items()
               if k in ("team_id", "team_name", "rating", "rank_at_fetch", "corpora")},
        }

    json.dump({"pilots": pilots}, (OUT / "pilots.json").open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- per-pilot record, restricted to seats we can attribute ----
    def outcome(rec, seat):
        rw = rec["rewards"]
        if rw[seat] is None:
            return None
        o = rw[1 - seat]
        return bool(rw[seat] > (o if o is not None else 0))

    per_pilot_games = defaultdict(list)
    for rec in good:
        if rec["type"] != "EPISODE_TYPE_PUBLIC":
            continue
        if rec["sub"][0] == rec["sub"][1]:
            continue
        for seat in (0, 1):
            sid = rec["sub"][seat]
            if not sid or sid not in sub_hash:
                continue
            w = outcome(rec, seat)
            if w is None:
                continue
            per_pilot_games[sid].append({
                "episode_id": rec["episode_id"], "seat": seat, "won": w,
                "own_hash": rec["hash"][seat],
                "opp_family": rec["family"][1 - seat],
                "opp_hash": rec["hash"][1 - seat],
                "opp_sub": rec["sub"][1 - seat],
                "created_at": rec["created_at"], "corpus": rec["corpus"],
                "path": rec["path"],
            })

    json.dump(
        {sid: g for sid, g in per_pilot_games.items()},
        (OUT / "pilot_games.json").open("w", encoding="utf-8"), indent=0,
    )
    print("wrote corpus_stats.json / pilots.json / pilot_games.json")
    print(json.dumps(corpus_stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
