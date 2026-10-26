#!/usr/bin/env python3
"""Rank all 50,000 sequences and select the reference-filtered consensus Top 100."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from aggregate import KEYS, aggregate_scores  # noqa: E402
from flow_dual_apex_submission import generate as frozen  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_aggregate(path: Path, backend: str) -> dict[tuple[str, ...], dict]:
    table = aggregate_scores(pd.read_csv(path, float_precision="round_trip"), backend)
    return {tuple(row[key] for key in KEYS): row for row in table.to_dict(orient="records")}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "data/library_50000.csv")
    parser.add_argument("--aggregate", type=Path, help="Combined legacy/pathogen aggregate CSV; defaults to frozen data")
    parser.add_argument("--legacy-scores", type=Path, help="Legacy scores.csv or aggregate_scores.csv")
    parser.add_argument("--pathogen-scores", type=Path, help="Pathogen scores.csv or aggregate_scores.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/antibacterial.fasta")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paired = args.legacy_scores is not None or args.pathogen_scores is not None
    if paired and (args.aggregate is not None or args.legacy_scores is None or args.pathogen_scores is None):
        parser.error("Use either --aggregate or both --legacy-scores and --pathogen-scores")
    if not paired:
        args.aggregate = args.aggregate or ROOT / "data/dual_apex_scores.csv"
    out = args.output_dir.resolve()
    if "raw_data" in out.parts:
        parser.error("Writing to raw_data is prohibited")
    names = ["library_ranked.csv", "consensus_ranked.csv", "top100_scores.csv", "top.fasta", "ranking_metadata.json"]
    if any((out / name).exists() for name in names):
        parser.error("Ranked output already exists; choose a new output directory")

    library = frozen.read_table(args.library)
    references = frozen.read_reference(args.reference)
    frozen.validate_library(library, references)
    legacy_path = args.aggregate or args.legacy_scores
    pathogen_path = args.aggregate or args.pathogen_scores
    legacy_scores = read_aggregate(legacy_path, "legacy")
    pathogen_scores = read_aggregate(pathogen_path, "pathogen")
    library_keys = {tuple(row[key] for key in KEYS) for row in library}
    if set(legacy_scores) != library_keys or set(pathogen_scores) != library_keys:
        raise ValueError("Both score tables must match all library candidate_id/sequence/hash keys exactly")
    legacy_threshold_path = ROOT / "data/legacy_selected11_thresholds.json"
    pathogen_threshold_path = ROOT / "data/pathogen_thresholds.json"
    legacy_threshold = json.loads(legacy_threshold_path.read_text())
    pathogen_threshold = json.loads(pathogen_threshold_path.read_text())
    scores = []
    for library_row in library:
        key = tuple(library_row[column] for column in KEYS)
        row = {**library_row, **legacy_scores[key], **pathogen_scores[key]}
        row.update({name: str(value) for name, value in frozen.recompute_gates(row, legacy_threshold, pathogen_threshold).items()})
        scores.append(row)
    # Reuse the frozen implementation for validation, all four gates and consensus ordering.
    consensus, counts = frozen.rank_consensus(library, scores, legacy_threshold, pathogen_threshold)
    top, examined = frozen.select_top(consensus, references)
    consensus_rank = {row["candidate_id"]: rank for rank, row in enumerate(consensus, start=1)}
    final_rank = {row["candidate_id"]: row["final_rank"] for row in top}
    # Apply the same ordering to every sequence, including those failing the gates.
    scores.sort(key=lambda row: (
        float(row["apex_pathogen_median_log10_mic"]),
        float(row["legacy_selected11_median_log10_mic"]),
        row["sequence_sha256"],
    ))
    ranked = [{"library_rank": rank,
               "raw_consensus_rank": consensus_rank.get(row["candidate_id"], ""),
               "final_rank": final_rank.get(row["candidate_id"], ""), **row}
              for rank, row in enumerate(scores, start=1)]
    ranked_consensus = [{"raw_consensus_rank": rank, **row}
                        for rank, row in enumerate(consensus, start=1)]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "library_ranked.csv", ranked)
    write_csv(out / "consensus_ranked.csv", ranked_consensus)
    (out / "top100_scores.csv").write_bytes(frozen.score_bytes(top))
    (out / "top.fasta").write_bytes(frozen.fasta_bytes(top))
    inputs = {"library": args.library, "legacy_scores": legacy_path, "pathogen_scores": pathogen_path,
              "reference": args.reference, "legacy_thresholds": legacy_threshold_path,
              "pathogen_thresholds": pathogen_threshold_path}
    metadata = {
        "counts": {"library": len(library), "consensus": len(consensus), "top": len(top),
                   "reference_examined": examined, **counts},
        "ranking_ascending": ["apex_pathogen_median_log10_mic", "legacy_selected11_median_log10_mic", "sequence_sha256"],
        "all_library_rank_definition": "Score order for all candidates, regardless of consensus status",
        "consensus_implementation": "src/flow_dual_apex_submission/generate.py:rank_consensus",
        "top_selection_implementation": "src/flow_dual_apex_submission/generate.py:select_top",
        "inputs": {key: {"file": path.name, "sha256": digest(path)} for key, path in inputs.items()},
        "outputs": {name: digest(out / name) for name in names if name != "ranking_metadata.json"},
        "frozen_top_sequence_order_matches": digest(out / "top.fasta") == frozen.EXPECTED_FASTA_HASHES["top.fasta"],
        "uncertainty": "Median of sample SDs in log10(MIC/uM), over the matched 11 targets",
        "activity": "Median log10 of arithmetic ensemble mean MIC (uM), over the matched 11 targets",
        "reference_rule": "Levenshtein.ratio <= 0.8 against all official reference sequences",
        "limitations": "Model predictions and related-model agreement are not experimental validation.",
    }
    (out / "ranking_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "counts": metadata["counts"],
                      "frozen_top_sequence_order_matches": metadata["frozen_top_sequence_order_matches"]}), flush=True)


if __name__ == "__main__":
    main()
