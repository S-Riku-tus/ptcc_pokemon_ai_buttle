from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.core.deck import substitution_distance
from ml.core.replay_io import deck_hash, extract_fast_header, replay_refs, zip_metadata


def _card_counts(deck: Iterable[int]) -> Counter[int]:
    return Counter(int(card_id) for card_id in deck)


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    return len(a & b) / len(a | b) if a or b else 0.0


def _major_cards(deck: Iterable[int], limit: int = 14) -> list[int]:
    counts = _card_counts(deck)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [card_id for card_id, _count in ranked[:limit]]


def _core_line_candidates(deck: Iterable[int]) -> list[int]:
    counts = _card_counts(deck)
    # Without full card metadata, use repeated non-energy-looking IDs as an auditable
    # candidate set. The report is intentionally for human review, not auto-promotion.
    return [
        card_id for card_id, count in sorted(counts.items())
        if count >= 2 and card_id > 20
    ][:12]


def _recommendation(team_count: int, replay_count: int) -> str:
    if team_count >= 5 and replay_count >= 100:
        return "high"
    if team_count >= 3:
        return "medium"
    if team_count >= 2:
        return "low"
    return "single-teacher warning"


def _discover_archives(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return [input_path]
    return sorted(input_path.rglob("*.zip"))


def _assign_cluster(deck: list[int], clusters: list[dict[str, Any]]) -> int:
    for index, cluster in enumerate(clusters):
        representative = cluster["representative_deck"]
        if deck_hash(deck) == deck_hash(representative):
            return index
        if substitution_distance(deck, representative) <= 8:
            return index
        if _jaccard(deck, representative) >= 0.72:
            return index
    clusters.append({"representative_deck": deck})
    return len(clusters) - 1


def analyze(input_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for archive_path in _discover_archives(input_path):
        meta = zip_metadata(archive_path)
        refs = replay_refs(archive_path)
        for ref in refs:
            header = extract_fast_header(ref)
            for seat, deck in enumerate(header["decks"][:2]):
                if len(deck) != 60:
                    continue
                cluster_id = _assign_cluster(deck, clusters)
                grouped[cluster_id].append({
                    "archive": archive_path.name,
                    "episode_id": ref.episode_id,
                    "seat": seat,
                    "team": header["team_names"][seat] or meta.get("team_name") or "",
                    "submission_id": meta.get("submission_id"),
                    "rank": meta.get("rank"),
                    "deck_hash": deck_hash(deck),
                    "deck": deck,
                    "reward": header["rewards"][seat] if seat < len(header["rewards"]) else None,
                })

    rows: list[dict[str, Any]] = []
    for cluster_id, records in sorted(grouped.items()):
        teams = {str(row["team"]) for row in records if row.get("team")}
        submissions = {str(row["submission_id"]) for row in records if row.get("submission_id")}
        hashes = {str(row["deck_hash"]) for row in records}
        ranks = [int(row["rank"]) for row in records if str(row.get("rank") or "").isdigit()]
        wins = sum(1 for row in records if isinstance(row.get("reward"), (int, float)) and float(row["reward"]) > 0)
        losses = sum(1 for row in records if isinstance(row.get("reward"), (int, float)) and float(row["reward"]) < 0)
        representative = clusters[cluster_id]["representative_deck"]
        distances = [
            substitution_distance(row["deck"], representative)
            for row in records
        ]
        row = {
            "cluster_id": f"cluster_{cluster_id:03d}",
            "representative_deck_hash": deck_hash(representative),
            "major_cards": " ".join(map(str, _major_cards(representative))),
            "core_line_candidates": " ".join(map(str, _core_line_candidates(representative))),
            "team_count": len(teams),
            "submission_count": len(submissions),
            "replay_count": len({row["episode_id"] for row in records}),
            "decision_count_estimate": len(records) * 45,
            "wins": wins,
            "losses": losses,
            "rank_min": min(ranks) if ranks else "",
            "rank_max": max(ranks) if ranks else "",
            "rank_distribution": " ".join(map(str, sorted(set(ranks)))) if ranks else "",
            "deck_variant_count": len(hashes),
            "mean_replacement_distance": round(sum(distances) / len(distances), 3) if distances else 0.0,
            "ml_recommendation": _recommendation(len(teams), len({row["episode_id"] for row in records})),
            "teams": " | ".join(sorted(teams)),
        }
        rows.append(row)

    csv_path = output_dir / "deck_archetype_clusters.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    report_path = output_dir / "deck_archetype_report.md"
    lines = [
        "# Deck Archetype Analysis",
        "",
        f"- input: `{input_path}`",
        f"- clusters: `{len(rows)}`",
        "",
        "| cluster | teams | submissions | replays | variants | recommendation | major cards |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['cluster_id']} | {row['team_count']} | {row['submission_count']} | "
            f"{row['replay_count']} | {row['deck_variant_count']} | {row['ml_recommendation']} | "
            f"{row['major_cards']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "deck_archetype_clusters.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = analyze(args.input, args.output)
    print(json.dumps({"clusters": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
