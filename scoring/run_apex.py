#!/usr/bin/env python3
"""Run the frozen legacy or pathogen APEX ensemble on a CSV of sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["legacy", "pathogen"])
    parser.add_argument("--input", type=Path, default=ROOT / "data/library_50000.csv")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or not 0 <= args.seed < 2**32:
        parser.error("batch-size/threads must be positive, seed in [0, 2**32)")
    out = args.output_dir.resolve()
    if "raw_data" in out.parts:
        parser.error("Writing to raw_data is prohibited")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    import numpy as np
    import pandas as pd
    import torch
    from aggregate import aggregate_scores, normalize_keys
    from apex_runtime import apex, apex_pathogen

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable; select --device auto or cpu")
        torch.cuda.manual_seed_all(args.seed)
    model_root = args.model_root.resolve()
    expected = json.loads((ROOT / "scoring/model_provenance.json").read_text())[args.backend]
    revision = subprocess.run(
        ["git", "-C", str(model_root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if revision != expected["revision"]:
        raise ValueError(f"Model repository revision differs; expected {expected['revision']}")
    code = model_root / expected["upstream_code_file"]
    if digest(code) != expected["upstream_code_sha256"]:
        raise ValueError("Upstream model implementation differs from frozen source")
    models = apex.discover_models(model_root) if args.backend == "legacy" else apex_pathogen.discover_pathogen_models(model_root)
    if {path.name: digest(path) for path in models} != expected["model_sha256"]:
        raise ValueError("Model checkpoint set/hashes differ from the frozen submission")
    frame = normalize_keys(pd.read_csv(args.input, float_precision="round_trip"))
    module = apex if args.backend == "legacy" else apex_pathogen
    runtime = {
        "backend": args.backend, "input_sha256": digest(args.input),
        "wrapper_sha256": digest(Path(__file__)), "scorer_sha256": digest(Path(module.__file__)),
        "model_root": str(model_root), "repository_commit": revision,
        "seed": args.seed, "batch_size": args.batch_size, "cpu_threads": args.threads,
        "device": device, "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "deterministic_algorithms": True, "allow_tf32": False, "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
    }
    out.mkdir(parents=True, exist_ok=True)
    runtime_path = out / "runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError("Runtime changed; choose a fresh output directory")
    elif not runtime_path.exists() and any(out.iterdir()):
        raise ValueError("Output directory has unrelated files; choose a fresh directory")
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    normalized_input = out / "input_sequences.csv"
    frame.to_csv(normalized_input, index=False, float_format="%.17g")
    call_args = [
        "--input", str(normalized_input), "--output", str(out / "scores.csv"),
        "--summary", str(out / "summary.json"), "--manifest", str(out / "manifest.json"),
        "--state", str(out / "ensemble_state.npz"), "--device", device,
        "--batch-size", str(args.batch_size),
    ]
    if args.backend == "legacy":
        call_args += ["--apex-root", str(model_root), "--strain-groups", str(ROOT / "scoring/configs/apex_strain_groups.yaml")]
    else:
        call_args += ["--apex-pathogen-root", str(model_root)]
    module.main(call_args)
    aggregate = aggregate_scores(pd.read_csv(out / "scores.csv", float_precision="round_trip"), args.backend)
    aggregate.to_csv(out / "aggregate_scores.csv", index=False, float_format="%.17g")
    print(json.dumps({"backend": args.backend, "n_sequences": len(frame), "output_dir": str(out),
                      "scores_sha256": digest(out / "scores.csv"), "aggregate_sha256": digest(out / "aggregate_scores.csv")}), flush=True)


if __name__ == "__main__":
    main()
