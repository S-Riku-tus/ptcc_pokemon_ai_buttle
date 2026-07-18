from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .deck import classify, has_alakazam_line, major_differences
from .replay_io import (
    ReplayRef,
    deck_hash,
    extract_fast_header,
    replay_bundle_scope,
    replay_refs,
    zip_metadata,
)


def normalize_team_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("&", " and ").replace("_", " ").replace("-", " ")
    return " ".join(re.findall(r"[\w.]+", text))


def _exact_team_seats(team_names: list[Any], target_name: str) -> list[int]:
    target = normalize_team_name(target_name)
    return [i for i, name in enumerate(team_names[:2]) if normalize_team_name(name) == target]


def _resolved_target_team(target_name: str, team_names: list[Any], seat: int) -> str:
    if target_name:
        return target_name
    if 0 <= seat < len(team_names):
        return str(team_names[seat] or "")
    return ""


def _reference_deck(all_headers: list[tuple[ReplayRef, dict[str, Any], dict[str, Any]]]) -> list[int]:
    # Prefer latest/rank-1 Majkel, then any Majkel exact-name seat.
    candidates: list[list[int]] = []
    for ref, header, meta in all_headers:
        if normalize_team_name(meta.get("team_name")) != "majkel1337":
            continue
        exact = _exact_team_seats(header["team_names"], "Majkel1337")
        manifest = meta.get("source_manifest", {}).get(ref.episode_id, {})
        if str(manifest.get("detected_submission_agent_index", "")).isdigit():
            exact = [int(manifest["detected_submission_agent_index"])]
        for seat in exact:
            deck = header["decks"][seat]
            if has_alakazam_line(deck):
                candidates.append(deck)
    if not candidates:
        raise RuntimeError("No Majkel Alakazam reference deck could be inferred")
    modal_hash = Counter(deck_hash(deck) for deck in candidates).most_common(1)[0][0]
    return next(deck for deck in candidates if deck_hash(deck) == modal_hash)


def _deduplicate_usable_trajectories(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove repeated copies of the same submission/episode/seat trajectory.

    A leaderboard episode can be present in both an older top-N snapshot and a
    newer full-submission bundle.  Keeping both would silently overweight that
    expert action sequence and leak identical trajectories across holdouts.
    Prefer the copy with the strongest seat evidence and retain excluded audit
    rows separately.
    """
    usable = frame[frame["usable_manifest"] == True].copy()
    excluded = frame[frame["usable_manifest"] != True].copy()
    before = len(usable)
    usable = (
        usable.sort_values(
            ["trajectory_id", "seat_confidence", "zip_name"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["trajectory_id"], keep="first")
    )
    combined = pd.concat([usable, excluded], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["zip_name", "episode_id", "target_seat"], na_position="last"
    ).reset_index(drop=True)
    return combined, before - len(usable)


def build_manifest(zip_paths: Iterable[str | Path], output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any], list[int]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_headers: list[tuple[ReplayRef, dict[str, Any], dict[str, Any]]] = []
    for zip_path in zip_paths:
        refs = replay_refs(zip_path)
        metadata_by_scope = {
            scope: zip_metadata(zip_path, next(ref.member for ref in refs if replay_bundle_scope(ref.member) == scope))
            for scope in {replay_bundle_scope(ref.member) for ref in refs}
        }
        for ref in refs:
            all_headers.append(
                (ref, extract_fast_header(ref), metadata_by_scope[replay_bundle_scope(ref.member)])
            )
    reference = _reference_deck(all_headers)
    reference_hash = deck_hash(reference)

    by_bundle: dict[tuple[Path, str], list[tuple[ReplayRef, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in all_headers:
        ref = item[0]
        by_bundle[(ref.zip_path, replay_bundle_scope(ref.member))].append(item)

    rows: list[dict[str, Any]] = []
    for (zip_path, _bundle_scope), items in sorted(
        by_bundle.items(), key=lambda kv: (kv[0][0].name, kv[0][1])
    ):
        meta = items[0][2]
        target_name = str(meta.get("team_name") or "")
        source_manifest = meta.get("source_manifest", {})
        exact_target_decks: list[list[int]] = []
        for ref, header, _ in items:
            manifest_row = source_manifest.get(ref.episode_id, {})
            manifest_seat = str(manifest_row.get("detected_submission_agent_index", ""))
            seats = [int(manifest_seat)] if manifest_seat.isdigit() else _exact_team_seats(header["team_names"], target_name)
            for seat in seats:
                if 0 <= seat <= 1 and header["decks"][seat]:
                    exact_target_decks.append(header["decks"][seat])
        modal_hash = Counter(deck_hash(deck) for deck in exact_target_decks).most_common(1)[0][0] if exact_target_decks else ""

        for ref, header, _ in items:
            teams, rewards, decks, hashes = header["team_names"], header["rewards"], header["decks"], header["deck_hashes"]
            source_row = source_manifest.get(ref.episode_id, {})
            source_seat = str(source_row.get("detected_submission_agent_index", ""))
            candidate_seats: list[int] = []
            method, confidence, reason = "", 0.0, ""
            if source_seat.isdigit() and int(source_seat) in (0, 1):
                candidate_seats = [int(source_seat)]
                method, confidence = "source_manifest", 1.0
            else:
                exact = _exact_team_seats(teams, target_name)
                if len(exact) == 1:
                    candidate_seats = exact
                    method, confidence = "team_name_exact", 0.995
                elif len(exact) == 2 and hashes[0] and hashes[0] == hashes[1]:
                    candidate_seats = [0, 1]
                    method, confidence = "self_play_exact_same_deck", 0.98
                elif len(exact) > 1:
                    reason = "team_name_matches_both_but_decks_differ"
                else:
                    modal_matches = [seat for seat in (0, 1) if modal_hash and hashes[seat] == modal_hash]
                    if len(modal_matches) == 1:
                        candidate_seats = modal_matches
                        method, confidence = "modal_target_deck_unique", 0.92
                    elif len(modal_matches) == 2:
                        target_root = (normalize_team_name(target_name).split() or [""])[0]
                        alias_matches = [
                            seat for seat in modal_matches
                            if target_root and (normalize_team_name(teams[seat]).split() or [""])[0] == target_root
                        ]
                        if len(alias_matches) == 1:
                            candidate_seats = alias_matches
                            method, confidence = "team_alias_plus_modal_deck", 0.90
                        elif normalize_team_name(teams[0]) == normalize_team_name(teams[1]):
                            candidate_seats = [0, 1]
                            method, confidence = "alias_self_play_same_modal_deck", 0.88
                        else:
                            reason = "both_seats_match_target_deck_without_team_identity"
                    else:
                        reason = "target_seat_not_identifiable"

            if not candidate_seats:
                rows.append({
                    "trajectory_id": f"{meta.get('submission_id','na')}:{ref.episode_id}:excluded",
                    "zip_path": str(zip_path), "zip_name": zip_path.name, "replay_path": ref.member,
                    "path_variant": ref.path_variant, "rank": meta.get("rank"), "submission_id": meta.get("submission_id"),
                    "episode_id": ref.episode_id, "target_seat": pd.NA, "seat_method": method or "excluded",
                    "seat_confidence": confidence, "exclusion_reason": reason, "target_team": target_name,
                    "team_names_json": json.dumps(teams, ensure_ascii=False), "usable_manifest": False,
                })
                continue

            for seat in candidate_seats:
                deck = decks[seat]
                opponent = 1 - seat
                resolved_target_team = _resolved_target_team(target_name, teams, seat)
                deck_type, distance = classify(deck, reference)
                line = has_alakazam_line(deck)
                exclusion = "" if line and len(deck) == 60 else ("not_alakazam" if not line else f"initial_deck_size_{len(deck)}")
                reward = rewards[seat] if seat < len(rewards) else None
                rows.append({
                    "trajectory_id": f"{meta.get('submission_id','na')}:{ref.episode_id}:{seat}",
                    "zip_path": str(zip_path), "zip_name": zip_path.name, "replay_path": ref.member,
                    "path_variant": ref.path_variant, "rank": int(meta.get("rank") or 99),
                    "submission_id": meta.get("submission_id"), "episode_id": ref.episode_id,
                    "target_seat": seat, "seat_method": method, "seat_confidence": confidence,
                    "exclusion_reason": exclusion, "target_team": resolved_target_team, "observed_target_team": teams[seat],
                    "opponent_team": teams[opponent], "team_names_json": json.dumps(teams, ensure_ascii=False), "target_reward": reward,
                    "target_win": bool(reward is not None and float(reward) > 0),
                    "target_loss": bool(reward is not None and float(reward) < 0),
                    "deck_hash": hashes[seat], "opponent_deck_hash": hashes[opponent], "deck_type": deck_type,
                    "majkel_distance": distance if math.isfinite(distance) else pd.NA,
                    "major_card_differences_json": json.dumps(major_differences(deck, reference), ensure_ascii=False),
                    "initial_deck_json": json.dumps(deck), "alakazam_line": line,
                    "deck_size": len(deck), "usable_manifest": not exclusion,
                })

    frame = pd.DataFrame(rows).sort_values(["zip_name", "episode_id", "target_seat"], na_position="last").reset_index(drop=True)
    frame, duplicate_trajectory_rows_removed = _deduplicate_usable_trajectories(frame)
    frame.to_csv(output / "manifest.csv", index=False)
    usable = frame[frame["usable_manifest"] == True]
    zip_variants = frame.groupby("zip_name")["path_variant"].agg(lambda x: sorted(set(x))).to_dict()
    stats = {
        "zip_count": int(len({ref.zip_path for ref, _, _ in all_headers})),
        "replay_file_count": int(len(all_headers)),
        "full_replay_count": int(len(all_headers)),
        "usable_episode_count": int(usable["episode_id"].nunique()),
        "legacy_singular_replay_count": int(sum(ref.path_variant == "replay" for ref, _, _ in all_headers)),
        "plural_replays_recovered": int(sum(ref.path_variant == "replays" for ref, _, _ in all_headers)),
        "manifest_trajectory_count": int(len(frame)),
        "usable_trajectory_count": int(len(usable)),
        "top50_plural_zip_count": int(len({
            ref.zip_path for ref, _, _ in all_headers if ref.path_variant == "replays"
        })),
        "excluded_trajectory_rows": int((frame["usable_manifest"] == False).sum()),
        "duplicate_trajectory_rows_removed": int(duplicate_trajectory_rows_removed),
        "unique_teams": int(usable["target_team"].nunique()),
        "unique_submissions": int(usable["submission_id"].nunique()),
        "unique_decks": int(usable["deck_hash"].nunique()),
        "majkel_reference_hash": reference_hash,
        "seat_methods": usable["seat_method"].value_counts().to_dict(),
        "exclusion_reasons": frame.loc[frame["usable_manifest"] == False, "exclusion_reason"].fillna("unknown").value_counts().to_dict(),
        "path_variants_by_zip": zip_variants,
        "replay_layouts_supported": ["replay", "replays"],
        "missing_rank_49_jack": not any("rank49" in name.lower() for name in frame["zip_name"].unique()),
    }
    (output / "manifest_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame, stats, reference


def build_deck_clusters(manifest: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    usable = manifest[manifest["usable_manifest"] == True].copy()
    groups = []
    for deck_hash_value, group in usable.groupby("deck_hash"):
        first = group.iloc[0]
        groups.append({
            "deck_hash": deck_hash_value,
            "deck_type": first["deck_type"],
            "majkel_distance": first["majkel_distance"],
            "trajectory_count": len(group),
            "episode_count": group["episode_id"].nunique(),
            "team_count": group["target_team"].nunique(),
            "teams": " | ".join(sorted(set(str(x) for x in group["target_team"]))),
            "ranks": " | ".join(str(x) for x in sorted(set(int(x) for x in group["rank"]))),
            "wins": int(group["target_win"].sum()),
            "losses": int(group["target_loss"].sum()),
            "win_rate": float(group["target_win"].mean()),
            "major_card_differences_json": first["major_card_differences_json"],
            "initial_deck_json": first["initial_deck_json"],
        })
    clusters = pd.DataFrame(groups).sort_values(["majkel_distance", "deck_hash"]).reset_index(drop=True)
    clusters.to_csv(output_path, index=False)
    return clusters
