from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    canonical_deck, classify_deck, deck_hash, deck_json, describe_variant, load_config,
    load_deck_evidence, load_team_ratings, match_evidence_team, normalize_team,
    parse_bundle_name, resolve_workspace_path, sha256_bytes,
)
from .replay_io import (
    alignment_counts, bundle_metadata, episode_ids, extract_initial_decks, load_replay,
    replay_health, replay_outcome, replay_refs,
)


MANIFEST_COLUMNS = [
    "episode_id", "team", "rank", "rating", "submission_id", "log_acquired_at",
    "deck_60", "deck_hash", "is_alakazam", "deck_type", "opponent", "go_first",
    "outcome", "normal_end", "timeout", "error", "total_steps", "decision_count",
    "legal_candidate_count", "duplicate", "usable", "exclusion_reason", "source_zip",
    "source_member", "source_sha256", "target_seat", "replay_available", "deck_evidence",
]


def _rating_for(team: str, ratings: dict[str, float]) -> float | None:
    key = normalize_team(team)
    if key in ratings:
        return ratings[key]
    for candidate, rating in ratings.items():
        if candidate in key or key in candidate:
            return rating
    return None


def _archetype(deck: list[int]) -> str:
    cards = set(deck)
    if 743 in cards:
        return "Alakazam"
    known = {
        431: "Team Rocket Spidops", 434: "Team Rocket Spidops", 400: "Team Rocket Spidops",
        666: "Cinderace", 1031: "Mega Starmie", 1030: "Mega Starmie",
    }
    for card, label in known.items():
        if card in cards:
            return label
    return f"deck_{deck_hash(deck)[:10]}" if deck else "unknown"


def build_manifest(config: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    replay_glob = resolve_workspace_path(config["replay_glob"])
    zip_paths = sorted(replay_glob.parent.glob(replay_glob.name))
    deck_evidence = load_deck_evidence(resolve_workspace_path(config["deck_evidence"]))
    ratings = load_team_ratings(resolve_workspace_path(config["team_evidence"]))
    majkel_key = match_evidence_team("Majkel1337", deck_evidence)
    majkel_deck = deck_evidence.get(majkel_key or "", [])
    rows: list[dict[str, Any]] = []
    alignment = {"same": Counter(), "next": Counter()}

    for zip_path in zip_paths:
        metadata = bundle_metadata(zip_path)
        parsed_rank, parsed_team, parsed_submission = parse_bundle_name(zip_path)
        rank = int(metadata.get("rank") or parsed_rank or 0) or None
        team = str(metadata.get("team_name") or parsed_team)
        submission = int(metadata.get("submission_id") or parsed_submission or 0) or None
        acquired = str(metadata.get("fetched_at_local") or zip_path.stat().st_mtime)
        source_hash = sha256_bytes(zip_path.read_bytes())
        evidence_key = match_evidence_team(team, deck_evidence)
        evidence_deck = deck_evidence.get(evidence_key or "", [])
        refs = replay_refs(zip_path)
        ref_by_episode = {ref.episode_id: ref for ref in refs}
        for episode_id in episode_ids(zip_path):
            ref = ref_by_episode.get(episode_id)
            base = {column: "" for column in MANIFEST_COLUMNS}
            base.update({
                "episode_id": episode_id,
                "team": team,
                "rank": rank,
                "rating": _rating_for(team, ratings),
                "submission_id": submission,
                "log_acquired_at": acquired,
                "source_zip": str(zip_path.relative_to(resolve_workspace_path("."))),
                "source_sha256": source_hash,
                "replay_available": bool(ref),
            })
            if ref is None:
                is_alakazam, deck_type = classify_deck(evidence_deck)
                base.update({
                    "deck_60": deck_json(evidence_deck) if len(evidence_deck) == 60 else "",
                    "deck_hash": deck_hash(evidence_deck) if len(evidence_deck) == 60 else "",
                    "is_alakazam": is_alakazam,
                    "deck_type": describe_variant(evidence_deck, majkel_deck) if evidence_deck else "unknown",
                    "usable": False,
                    "exclusion_reason": "replay_missing_observation_action",
                    "deck_evidence": f"aggregate:{evidence_key}" if evidence_key else "none",
                })
                rows.append(base)
                continue

            replay = load_replay(ref)
            decks = extract_initial_decks(replay)
            seat = ref.target_seat
            if seat not in (0, 1):
                alakazam_seats = [index for index, deck in enumerate(decks) if classify_deck(deck)[0]]
                seat = alakazam_seats[0] if len(alakazam_seats) == 1 else None
            deck = decks[seat] if seat in (0, 1) else evidence_deck
            opponent_deck = decks[1 - seat] if seat in (0, 1) else []
            is_alakazam, deck_type = classify_deck(deck)
            normal, timeout, error = replay_health(replay)
            same = alignment_counts(replay, seat, 0) if seat in (0, 1) else Counter()
            next_ = alignment_counts(replay, seat, 1) if seat in (0, 1) else Counter()
            alignment["same"].update(same)
            alignment["next"].update(next_)
            decisions = next_.get("legal", 0)
            legal_candidates = 0
            if seat in (0, 1):
                for index in range(max(0, len(replay.get("steps") or []) - 1)):
                    try:
                        select = replay["steps"][index][seat]["observation"].get("select") or {}
                        action = replay["steps"][index + 1][seat].get("action")
                    except (KeyError, IndexError, TypeError, AttributeError):
                        continue
                    if select and isinstance(action, list):
                        legal_candidates += len(select.get("option") or [])
            exclusion = ""
            usable = True
            if seat not in (0, 1):
                usable, exclusion = False, "target_seat_unknown"
            elif not is_alakazam:
                usable, exclusion = False, "target_deck_not_alakazam"
            elif not normal:
                usable, exclusion = False, "abnormal_end"
            elif not decisions:
                usable, exclusion = False, "no_aligned_decisions"
            current0 = {}
            try:
                current0 = replay["steps"][1][seat]["observation"].get("current") or {}
            except (KeyError, IndexError, TypeError, AttributeError):
                pass
            base.update({
                "log_acquired_at": ref.created_at or acquired,
                "deck_60": deck_json(deck) if deck else "",
                "deck_hash": deck_hash(deck) if deck else "",
                "is_alakazam": is_alakazam,
                "deck_type": describe_variant(deck, majkel_deck),
                "opponent": _archetype(opponent_deck),
                "go_first": current0.get("firstPlayer") == seat if current0 else "",
                "outcome": replay_outcome(replay, seat) if seat in (0, 1) else "",
                "normal_end": normal,
                "timeout": timeout,
                "error": error,
                "total_steps": len(replay.get("steps") or []),
                "decision_count": decisions,
                "legal_candidate_count": legal_candidates,
                "usable": usable,
                "exclusion_reason": exclusion,
                "source_member": ref.member,
                "target_seat": seat,
                "deck_evidence": "replay_initial_deck",
            })
            rows.append(base)

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    duplicate_mask = manifest.duplicated(subset=["episode_id"], keep="first")
    manifest.loc[duplicate_mask, "duplicate"] = True
    manifest.loc[~duplicate_mask, "duplicate"] = False
    manifest.loc[duplicate_mask, "usable"] = False
    manifest.loc[duplicate_mask, "exclusion_reason"] = "duplicate_episode"
    manifest = manifest.sort_values(["episode_id", "rank"], kind="stable").reset_index(drop=True)

    cluster_rows: list[dict[str, Any]] = []
    grouped = manifest[manifest["deck_hash"].astype(bool)].groupby("deck_hash", dropna=False)
    for hash_value, group in grouped:
        deck = json.loads(group.iloc[0]["deck_60"])
        counts = Counter(deck)
        cluster_rows.append({
            "deck_hash": hash_value,
            "deck_type": group.iloc[0]["deck_type"],
            "is_alakazam": bool(group.iloc[0]["is_alakazam"]),
            "episode_count": len(group),
            "teams": ";".join(sorted(set(group["team"].astype(str)))),
            "card_count": len(deck),
            "enriching_energy": counts[13],
            "hyper_aroma": counts.get(0, 0),
            "boss": counts[1182],
            "fezandipiti_ex": counts[140],
            "shaymin": counts[343],
            "dunsparce": counts[305],
            "dudunsparce": counts[66],
            "night_stretcher": counts[1097],
            "nighttime_mine": counts[1266],
            "rare_candy": counts[1079],
            "basic_psychic": counts[5],
            "telepath_energy": counts[19],
            "canonical_deck": canonical_deck(deck),
        })
    clusters = pd.DataFrame(cluster_rows)
    stats = {
        "zip_count": len(zip_paths),
        "episode_count": len(manifest),
        "full_replay_count": int(manifest["replay_available"].fillna(False).sum()),
        "alakazam_episode_count": int(manifest["is_alakazam"].fillna(False).sum()),
        "usable_episode_count": int(manifest["usable"].fillna(False).sum()),
        "usable_decision_count": int(pd.to_numeric(manifest.loc[manifest["usable"] == True, "decision_count"], errors="coerce").fillna(0).sum()),
        "alignment": {name: dict(counts) for name, counts in alignment.items()},
        "missing_rank_21_50": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "episode_manifest.csv", index=False, encoding="utf-8")
    clusters.to_csv(output_dir / "deck_clusters.csv", index=False, encoding="utf-8")
    (output_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return manifest, clusters, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "data_processed"))
    args = parser.parse_args()
    _, _, stats = build_manifest(load_config(args.config), Path(args.output))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

