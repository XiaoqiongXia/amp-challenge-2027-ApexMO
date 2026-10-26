"""Derive the exact matched-11 consensus inputs from full APEX outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apex_runtime.apex import strain_slug
from apex_runtime.apex_pathogen import APEX_PATHOGEN_STRAINS

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["candidate_id", "sequence", "sequence_sha256"]


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "sequence" not in frame or frame.empty:
        raise ValueError("Input must contain at least one sequence")
    if not frame.sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{8,50}").all():
        raise ValueError("Sequences must be canonical uppercase and 8–50 residues")
    if frame.sequence.duplicated().any():
        raise ValueError("Sequences must be unique before scoring/ranking")
    hashes = frame.sequence.map(lambda seq: hashlib.sha256(seq.encode("ascii")).hexdigest())
    if "sequence_sha256" in frame and not frame.sequence_sha256.eq(hashes).all():
        raise ValueError("Sequence hashes do not match sequences")
    frame["sequence_sha256"] = hashes
    if "candidate_id" not in frame:
        frame["candidate_id"] = "seq_" + hashes
    if frame.candidate_id.isna().any() or frame.candidate_id.duplicated().any():
        raise ValueError("Candidate identifiers must be nonempty and unique")
    if frame.candidate_id.astype(str).str.strip().eq("").any():
        raise ValueError("Candidate identifiers must be nonempty")
    return frame


def aggregate_scores(frame: pd.DataFrame, backend: str) -> pd.DataFrame:
    frame = normalize_keys(frame)
    if backend == "legacy":
        targets = json.loads((ROOT / "data/legacy_selected11_thresholds.json").read_text())["activity_targets"]
        mic_prefix, sd_prefix, prefix = "apex_mic__", "apex_log10_mic_sd__", "legacy_selected11"
    else:
        targets = APEX_PATHOGEN_STRAINS
        mic_prefix, sd_prefix, prefix = "apex_pathogen_mic__", "apex_pathogen_log10_mic_sd__", "apex_pathogen"
    if len(targets) != 11 or len(set(targets)) != 11:
        raise ValueError("The consensus panel must contain 11 unique targets")
    columns = [f"{prefix}_{suffix}" for suffix in (
        "median_log10_mic", "median_mic_um", "median_log10_mic_sd", "max_log10_mic_sd"
    )]
    mic_columns = [mic_prefix + strain_slug(target) for target in targets]
    sd_columns = [sd_prefix + strain_slug(target) for target in targets]
    result = frame[KEYS].copy()
    if set(mic_columns + sd_columns).issubset(frame.columns):
        mic = frame[mic_columns].to_numpy(dtype=np.float64)
        sd = frame[sd_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(mic).all() or not np.isfinite(sd).all() or (mic <= 0).any() or (sd < 0).any():
            raise ValueError("MICs must be finite and positive; SDs finite and nonnegative")
        result[columns[0]] = np.median(np.log10(mic), axis=1)
        result[columns[1]] = np.median(mic, axis=1)
        result[columns[2]] = np.median(sd, axis=1)
        result[columns[3]] = np.max(sd, axis=1)
        # A saved aggregate with the same meaning must agree with direct derivation.
        for column in columns:
            if column in frame and not np.allclose(frame[column], result[column], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Stored aggregate disagrees with per-strain predictions: {column}")
    elif set(columns).issubset(frame.columns):
        result[columns] = frame[columns].astype(float)
    else:
        raise ValueError(f"Missing full strain predictions or matched-11 aggregates for {backend}")
    numeric = result[columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 1] <= 0).any() or (numeric[:, 2:] < 0).any():
        raise ValueError("Invalid aggregate activity or uncertainty values")
    return result
