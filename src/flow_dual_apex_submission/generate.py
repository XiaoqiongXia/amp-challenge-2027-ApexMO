"""Reconstruct the frozen submission, without invoking neural models.

The ``generate`` entry point reads hash-checked, previously generated sequences
and model predictions. It recomputes both activity/uncertainty gates, ranking,
and exact reference similarity. This is frozen-sample reconstruction, not fresh
Flow sampling or APEX inference. The separate inference workflow documents those
steps. Only the frozen selection seed 42 is accepted; it is not a neural sampling
seed. No input files are modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

import Levenshtein

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
LIBRARY_SIZE = 50_000
TOP_SIZE = 100
FROZEN_SEED = 42
MAX_REFERENCE_SIMILARITY = 0.8
REQUIRED_INPUTS = (
    "data/library_50000.csv",
    "data/dual_apex_scores.csv",
    "data/antibacterial.fasta",
    "data/legacy_selected11_thresholds.json",
    "data/pathogen_thresholds.json",
    "scripts/official_verify_submission.py",
)
EXPECTED_FASTA_HASHES = {
    "library.fasta": "007a661ac2fc9387da53d888631c80d02ac4987831c778344742f30335ffea8e",
    "top.fasta": "e6b3d160bc9c6541a36bd78c27f0e074aa3fa14c8081cc72fb4c0f06b6277ee6",
}
NUMERIC_SCORE_COLUMNS = (
    "legacy_selected11_median_log10_mic",
    "legacy_selected11_median_mic_um",
    "legacy_selected11_median_log10_mic_sd",
    "legacy_selected11_max_log10_mic_sd",
    "apex_pathogen_median_log10_mic",
    "apex_pathogen_median_mic_um",
    "apex_pathogen_median_log10_mic_sd",
    "apex_pathogen_max_log10_mic_sd",
)
TOP_SCORE_COLUMNS = (
    "final_rank", "raw_consensus_rank", "candidate_id", "sequence", "length",
    "source_training_seeds", "apex_pathogen_median_log10_mic",
    "apex_pathogen_median_mic_um", "apex_pathogen_median_log10_mic_sd",
    "legacy_selected11_median_log10_mic", "legacy_selected11_median_mic_um",
    "legacy_selected11_median_log10_mic_sd", "pathogen_active_pass",
    "pathogen_uncertainty_pass", "legacy_active_pass", "legacy_uncertainty_pass",
    "dual_apex_consensus_pass", "max_reference_levenshtein_ratio",
    "nearest_reference_id",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_inputs(root: Path) -> dict:
    """Check required file bytes against the versioned integrity manifest."""
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1, "Unsupported manifest schema")
    require(manifest["seed"] == FROZEN_SEED, "Manifest is not the frozen seed-42 submission")
    require(
        manifest["expected_outputs"] == EXPECTED_FASTA_HASHES,
        "Manifest expected outputs differ from the frozen submission",
    )
    for relative_name in REQUIRED_INPUTS:
        expected = manifest["files"].get(relative_name)
        require(expected is not None, f"Missing manifest entry: {relative_name}")
        data = (root / relative_name).read_bytes()
        require(
            len(data) == expected["bytes"] and sha256(data) == expected["sha256"],
            f"Input integrity check failed: {relative_name}",
        )
    return manifest


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_reference(path: Path) -> dict[str, str]:
    """Map sequence to its first reference ID in original FASTA order."""
    references: dict[str, str] = {}
    header = None
    parts: list[str] = []

    def append_record() -> None:
        if header is not None:
            sequence = "".join(parts)
            require(bool(sequence), "Empty reference sequence")
            references.setdefault(sequence, header)

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            append_record()
            require(bool(line[1:].strip()), "Empty reference ID")
            header = line[1:].split()[0]
            parts = []
        else:
            require(header is not None, "Reference sequence before its header")
            parts.append(line.upper())
    append_record()
    require(bool(references), "Reference FASTA is empty")
    return references


def validate_library(rows: list[dict], references: dict[str, str]) -> None:
    require(len(rows) == LIBRARY_SIZE, f"Expected {LIBRARY_SIZE} frozen sequences")
    seen_sequences: set[str] = set()
    seen_ids: set[str] = set()
    previous_key = None
    for row in rows:
        sequence = row["sequence"]
        require(8 <= len(sequence) <= 50, "Sequence length outside 8-50 residues")
        require(set(sequence) <= STANDARD_AMINO_ACIDS, "Noncanonical amino acid")
        require(int(row["length"]) == len(sequence), "Incorrect sequence length field")
        sequence_hash = sha256(sequence.encode("ascii"))
        require(row["sequence_sha256"] == sequence_hash, "Sequence SHA256 mismatch")
        require(row["candidate_id"] == "FLOW_" + sequence_hash[:20], "Candidate ID mismatch")
        selection_key = sha256(f"amp-challenge-library-v1|42|{sequence}".encode("ascii"))
        require(row["selection_key"] == selection_key, "Frozen selection key mismatch")
        key = (selection_key, sequence)
        require(previous_key is None or previous_key < key, "Frozen library ordering changed")
        previous_key = key
        require(sequence not in seen_sequences, "Duplicate library sequence")
        require(row["candidate_id"] not in seen_ids, "Duplicate candidate ID")
        require(sequence not in references, "Library overlaps the official reference")
        seen_sequences.add(sequence)
        seen_ids.add(row["candidate_id"])


def recompute_gates(row: dict, legacy: dict, pathogen: dict) -> dict[str, bool]:
    """Apply strict activity cutoffs and inclusive uncertainty cutoffs."""
    legacy_active = float(row["legacy_selected11_median_log10_mic"]) < float(
        legacy["activity_quantiles_log10_mic"]["0.5"]
    )
    legacy_uncertainty = float(row["legacy_selected11_median_log10_mic_sd"]) <= float(
        legacy["uncertainty_threshold_q95_log10_mic_sd"]
    )
    pathogen_active = float(row["apex_pathogen_median_log10_mic"]) < float(
        pathogen["predicted_active_threshold_median_log10_mic"]
    )
    pathogen_uncertainty = float(row["apex_pathogen_median_log10_mic_sd"]) <= float(
        pathogen["uncertainty_threshold_q95_log10_mic_sd"]
    )
    return {
        "legacy_active_pass": legacy_active,
        "legacy_uncertainty_pass": legacy_uncertainty,
        "legacy_activity_uncertainty_pass": legacy_active and legacy_uncertainty,
        "pathogen_active_pass": pathogen_active,
        "pathogen_uncertainty_pass": pathogen_uncertainty,
        "pathogen_activity_uncertainty_pass": pathogen_active and pathogen_uncertainty,
        "dual_apex_active_pass": legacy_active and pathogen_active,
        "dual_apex_uncertainty_pass": legacy_uncertainty and pathogen_uncertainty,
        "dual_apex_consensus_pass": legacy_active and pathogen_active and legacy_uncertainty and pathogen_uncertainty,
    }


def rank_consensus(library: list[dict], scores: list[dict], legacy: dict, pathogen: dict) -> tuple[list[dict], dict]:
    require(len(scores) == LIBRARY_SIZE, "Expected one score row per library sequence")
    require(legacy["activity_scorer"] == "apex", "Incorrect legacy threshold model")
    require(pathogen["activity_scorer"] == "apex_pathogen", "Incorrect pathogen threshold model")
    by_id = {row["candidate_id"]: row for row in scores}
    require(len(by_id) == LIBRARY_SIZE, "Duplicate score candidate IDs")
    require(set(by_id) == {row["candidate_id"] for row in library}, "Score/library ID mismatch")
    counts: dict[str, int] = {}
    consensus: list[dict] = []
    for library_row in library:
        row = by_id[library_row["candidate_id"]].copy()
        for column in library_row:
            require(row[column] == library_row[column], f"Score/library mismatch: {column}")
        for column in NUMERIC_SCORE_COLUMNS:
            value = float(row[column])
            require(math.isfinite(value), f"Nonfinite score: {column}")
            if column.endswith("mic_um"):
                require(value > 0, f"Nonpositive MIC: {column}")
            elif column.endswith("mic_sd"):
                require(value >= 0, f"Negative uncertainty: {column}")
        gates = recompute_gates(row, legacy, pathogen)
        for column, value in gates.items():
            require(row[column] == str(value), f"Saved gate differs from recomputation: {column}")
            counts[column] = counts.get(column, 0) + int(value)
        row.update(gates)
        if gates["dual_apex_consensus_pass"]:
            consensus.append(row)
    consensus.sort(key=lambda row: (
        float(row["apex_pathogen_median_log10_mic"]),
        float(row["legacy_selected11_median_log10_mic"]),
        row["sequence_sha256"],
    ))
    return consensus, counts


def select_top(consensus: list[dict], references: dict[str, str]) -> tuple[list[dict], int]:
    selected: list[dict] = []
    examined = 0
    for raw_rank, row in enumerate(consensus, start=1):
        maximum = -1.0
        nearest_id = ""
        for reference, identifier in references.items():
            similarity = Levenshtein.ratio(row["sequence"], reference)
            if similarity > maximum:
                maximum = similarity
                nearest_id = identifier
        examined += 1
        if maximum <= MAX_REFERENCE_SIMILARITY:
            selected.append({
                **row,
                "raw_consensus_rank": raw_rank,
                "final_rank": len(selected) + 1,
                "max_reference_levenshtein_ratio": format(maximum, ".17g"),
                "nearest_reference_id": nearest_id,
            })
        if len(selected) == TOP_SIZE:
            return selected, examined
    raise ValueError(f"Only {len(selected)} consensus candidates pass the reference filter")


def fasta_bytes(rows: list[dict]) -> bytes:
    return "".join(f">{row['candidate_id']}\n{row['sequence']}\n" for row in rows).encode("ascii")


def score_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=TOP_SCORE_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def reconstruct(root: Path, output_dir: Path, seed: int = FROZEN_SEED) -> dict:
    require(
        seed == FROZEN_SEED,
        "Frozen reconstruction accepts only --seed 42. It does not sample new sequences; use the separate inference workflow for fresh model runs.",
    )
    root, output_dir = root.resolve(), output_dir.resolve()
    require(output_dir != root / "data", "Output directory must not overwrite frozen inputs")
    manifest = verify_inputs(root)
    require(Levenshtein.__version__ == "0.27.3", "Use the pinned Levenshtein==0.27.3 dependency")
    references = read_reference(root / "data/antibacterial.fasta")
    library = read_table(root / "data/library_50000.csv")
    validate_library(library, references)
    scores = read_table(root / "data/dual_apex_scores.csv")
    legacy = json.loads((root / "data/legacy_selected11_thresholds.json").read_text())
    pathogen = json.loads((root / "data/pathogen_thresholds.json").read_text())
    consensus, counts = rank_consensus(library, scores, legacy, pathogen)
    top, examined = select_top(consensus, references)
    counts.update({
        "library": len(library), "consensus": len(consensus), "top": len(top),
        "reference_unique_sequences": len(references),
        "reference_examined_candidates": examined,
        "reference_rejected_candidates": examined - len(top),
    })
    for name, count in counts.items():
        require(count == manifest["expected_counts"][name], f"Frozen count changed: {name}")
    payloads = {
        "library.fasta": fasta_bytes(library),
        "top.fasta": fasta_bytes(top),
        "top100_scores.csv": score_bytes(top),
    }
    for name, expected in EXPECTED_FASTA_HASHES.items():
        require(sha256(payloads[name]) == expected, f"Frozen output hash mismatch: {name}")
    summary = {
        "mode": "frozen-sample reconstruction; no model inference",
        "seed": seed,
        "status": "passed",
        "counts": counts,
        "reference_maximum_similarity_allowed": MAX_REFERENCE_SIMILARITY,
        "input_manifest_sha256": sha256((root / "data/manifest.json").read_bytes()),
        "outputs_sha256": {name: sha256(data) for name, data in payloads.items()},
    }
    payloads["reconstruction_summary.json"] = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=output_dir) as temporary:
        stage = Path(temporary)
        for name, data in payloads.items():
            (stage / name).write_bytes(data)
        for name in payloads:
            os.replace(stage / name, output_dir / name)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED, help="Frozen selection seed; only 42 is accepted")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "generate")
    args = parser.parse_args(argv)
    try:
        summary = reconstruct(REPOSITORY_ROOT, args.output_dir, args.seed)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Frozen reconstruction failed: {error}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
