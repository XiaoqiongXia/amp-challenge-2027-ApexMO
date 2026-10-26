#!/usr/bin/env python3
"""Generate fresh sequences from exported frozen Flow EMA checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flow_inference.model import AMPDDiT, AMPDDiTConfig
from flow_inference.sampling import generate_token_ids
from flow_inference.tokenizer import PeptideTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path, device: torch.device) -> tuple[AMPDDiT, dict]:
    """Load tensors and plain metadata, without training checkpoint unpickling."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "amp_flow_ema_inference_v1":
        raise ValueError("Expected an exported amp_flow_ema_inference_v1 checkpoint")
    model = AMPDDiT(AMPDDiTConfig.from_mapping(checkpoint["config"]["model"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, choices=[42, 123, 2025], default=42)
    parser.add_argument("--checkpoint", type=Path, help="Optional exported checkpoint override")
    parser.add_argument("--length", type=int, default=20, help="Requested length, 8–50 residues")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=52001, help="Sampling RNG seed")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/fresh_samples.csv"))
    args = parser.parse_args()
    if not 8 <= args.length <= 50:
        parser.error("--length must be between 8 and 50 residues, inclusive")
    if min(args.count, args.batch_size, args.cpu_threads) < 1 or args.steps < 2:
        parser.error("count, batch-size and cpu-threads must be positive; steps must be >=2")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32)")

    output = args.output.resolve()
    sidecar = output.with_suffix(".metadata.json")
    if "raw_data" in output.parts:
        parser.error("Writing to raw_data is not allowed")
    if output.exists() or sidecar.exists():
        parser.error("Output or metadata already exists; choose a new --output path")
    checkpoint_path = args.checkpoint or (
        REPOSITORY_ROOT / "checkpoint" / f"flow_seed{args.training_seed}_step3499_ema.pt"
    )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or --device auto")
    device = torch.device(device_name)
    torch.set_num_threads(args.cpu_threads)
    model, checkpoint = load_model(checkpoint_path, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = PeptideTokenizer()
    rows = []
    for offset in range(0, args.count, args.batch_size):
        batch_count = min(args.batch_size, args.count - offset)
        tokens = generate_token_ids(
            model,
            lengths=[args.length] * batch_count,
            steps=args.steps,
            device=device,
            path_exponent=float(checkpoint["config"]["flow"]["exponent"]),
            time_epsilon=float(checkpoint["config"]["generation"]["time_epsilon"]),
            max_length=int(checkpoint["config"]["model"]["max_length"]),
        )
        for token_ids in tokens:
            sequence = tokenizer.decode(token_ids)
            if len(sequence) != args.length:
                raise RuntimeError("Generated sequence does not match requested length")
            rows.append({
                "sequence": sequence,
                "length": len(sequence),
                "training_seed": checkpoint["config"]["seed"],
                "sampling_seed": args.seed,
                "sampling_steps": args.steps,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "inference/generate.py",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_source_sha256": checkpoint["source_sha256"],
        "output": output.name,
        "output_sha256": sha256(output),
        "count": args.count,
        "length": args.length,
        "training_seed": checkpoint["config"]["seed"],
        "sampling_seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "device": str(device),
        "python_version": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "flow-matching")
        },
        "path_exponent": checkpoint["config"]["flow"]["exponent"],
        "time_epsilon": checkpoint["config"]["generation"]["time_epsilon"],
        "categorical_dtype": "torch.float64",
        "model_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "fresh_samples_only": True,
        "note": "No activity prediction, deduplication, or reference exclusion is applied.",
    }
    with sidecar.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(rows)} fresh samples to {output} using {device}")


if __name__ == "__main__":
    main()
