"""Integration checks use the actual frozen data, without model inference."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from flow_dual_apex_submission.generate import (
    EXPECTED_FASTA_HASHES,
    REPOSITORY_ROOT,
    read_table,
    recompute_gates,
    reconstruct,
    sha256,
    verify_inputs,
)


class FrozenSubmissionTests(unittest.TestCase):
    def test_exact_submission_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generate"
            summary = reconstruct(REPOSITORY_ROOT, output)
            first = {path.name: path.read_bytes() for path in output.iterdir()}
            reconstruct(REPOSITORY_ROOT, output)
            second = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(first, second)
            for name, expected in EXPECTED_FASTA_HASHES.items():
                self.assertEqual(sha256(first[name]), expected)
            self.assertEqual(summary["counts"]["consensus"], 15_553)
            self.assertEqual(summary["counts"]["reference_examined_candidates"], 132)
            self.assertEqual(summary["counts"]["reference_rejected_candidates"], 32)
            generated_scores = read_table(output / "top100_scores.csv")
            recorded_scores = read_table(REPOSITORY_ROOT / "docs/top100_scores.csv")
            self.assertEqual(generated_scores, recorded_scores)

    def test_changed_frozen_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY_ROOT / "data", root / "data")
            shutil.copytree(REPOSITORY_ROOT / "scripts", root / "scripts")
            with (root / "data/dual_apex_scores.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Input integrity check failed"):
                verify_inputs(root)

    def test_unavailable_seed_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "only --seed 42"):
                reconstruct(REPOSITORY_ROOT, output, seed=123)
            self.assertFalse(output.exists())

    def test_threshold_boundaries_are_strict_activity_inclusive_uncertainty(self):
        legacy = {"activity_quantiles_log10_mic": {"0.5": 2.0}, "uncertainty_threshold_q95_log10_mic_sd": 0.5}
        pathogen = {"predicted_active_threshold_median_log10_mic": 3.0, "uncertainty_threshold_q95_log10_mic_sd": 0.6}
        row = {"legacy_selected11_median_log10_mic": 2.0, "legacy_selected11_median_log10_mic_sd": 0.5,
               "apex_pathogen_median_log10_mic": 3.0, "apex_pathogen_median_log10_mic_sd": 0.6}
        gates = recompute_gates(row, legacy, pathogen)
        self.assertFalse(gates["dual_apex_active_pass"])
        self.assertTrue(gates["dual_apex_uncertainty_pass"])
        self.assertFalse(gates["dual_apex_consensus_pass"])


if __name__ == "__main__":
    unittest.main()
