#!/usr/bin/env python3
"""Export exact EMA tensors from trusted local training checkpoints.

This maintenance utility requires the original research checkout and its
amp_design dependencies. Inference itself needs only generate.py and the
already-exported checkpoint files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from flow_inference.model import AMPDDiT, AMPDDiTConfig


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if "raw_data" in output_root.parts:
        parser.error("Output root cannot be inside raw_data")
    sys.path.insert(0, str(source_root / "src"))
    from amp_design.generate import load_checkpoint_model

    torch.set_num_threads(4)
    target_paths = [
        output_root / "checkpoint" / f"flow_seed{seed}_step3499_ema.pt"
        for seed in (42, 123, 2025)
    ]
    manifest_path = output_root / "metadata/flow_model_manifest.json"
    if any(p.exists() for p in [*target_paths, manifest_path]):
        parser.error("An export or manifest already exists; choose a new output root")
    (output_root / "checkpoint").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "amp_flow_ema_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "flow-matching")
        },
        "export_script": "inference/export_checkpoints.py",
        "description": "Exact EMA inference state; optimizer, RNG and raw training state excluded.",
        "source_files": {
            f"src/amp_design/{name}": file_sha256(source_root / "src/amp_design" / name)
            for name in ("model.py", "tokenizer.py", "flow_matching.py", "generate.py")
        },
        "models": [],
    }
    for seed, target in zip((42, 123, 2025), target_paths, strict=True):
        archive = source_root / f"outputs/test_frozen/seed_{seed}/distribution_50000.csv"
        with archive.open(newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle))
        original_path = Path(first["checkpoint"])
        # Resolve relocated research checkouts using the stable outputs-relative path.
        marker = "/outputs/"
        if marker in str(original_path):
            original_path = source_root / "outputs" / str(original_path).split(marker, 1)[1]
        original_digest = file_sha256(original_path)
        original_model, checkpoint = load_checkpoint_model(
            original_path, device=torch.device("cpu"), use_ema=True
        )
        if checkpoint["step"] != 3499 or checkpoint["config"]["seed"] != seed:
            raise ValueError("Unexpected source checkpoint identity")
        state = {key: value.detach().cpu().clone() for key, value in original_model.state_dict().items()}
        for key, value in checkpoint["ema"]["shadow"].items():
            assert torch.equal(state[key], value.cpu()), f"EMA mismatch for {key}"
        exported = {
            "format": "amp_flow_ema_inference_v1",
            "model": state,
            "config": checkpoint["config"],
            "step": int(checkpoint["step"]),
            "validation_loss": float(checkpoint["validation_loss"]),
            "ema_decay": float(checkpoint["ema"]["decay"]),
            "source_sha256": original_digest,
        }
        torch.save(exported, target)
        if target.stat().st_size >= 100 * 1024 * 1024:
            raise RuntimeError("Export exceeds the 100 MiB file limit")
        reloaded = torch.load(target, map_location="cpu", weights_only=True)
        portable = AMPDDiT(AMPDDiTConfig.from_mapping(reloaded["config"]["model"]))
        portable.load_state_dict(reloaded["model"], strict=True)
        portable.eval()
        tensor_records = []
        for key, original_tensor in state.items():
            saved = reloaded["model"][key]
            assert original_tensor.dtype == saved.dtype
            assert torch.equal(original_tensor, saved), key
            tensor_records.append({
                "name": key,
                "shape": list(saved.shape),
                "dtype": str(saved.dtype),
                "sha256": tensor_sha256(saved),
            })
        # Exercise masking, all-MASK and mixed-token inputs at multiple lengths/times.
        lengths = torch.tensor([8, 20, 50], dtype=torch.long)
        mask = torch.arange(50)[None, :] < lengths[:, None]
        tokens = torch.arange(150).reshape(3, 50).remainder(21)
        tokens[0, :8] = 20
        tokens = torch.where(mask, tokens, torch.full_like(tokens, 21))
        times = torch.tensor([0.0, 0.5, 0.999], dtype=torch.float32)
        with torch.no_grad():
            original_logits = original_model(tokens, times, lengths, mask)
            portable_logits = portable(tokens, times, lengths, mask)
        assert torch.isfinite(original_logits).all()
        assert torch.equal(original_logits, portable_logits), "Forward outputs differ"
        assert file_sha256(original_path) == original_digest, "Source checkpoint changed"
        record = {
            "training_seed": seed,
            "source_checkpoint": str(original_path.relative_to(source_root)),
            "source_checkpoint_bytes": original_path.stat().st_size,
            "source_sha256": original_digest,
            "source_generation_archive": str(archive.relative_to(source_root)),
            "source_archive_sha256": file_sha256(archive),
            "source_sampling_seed": int(first["seed"]),
            "source_sampling_steps": int(first["sampling_steps"]),
            "export": str(target.relative_to(output_root)),
            "export_bytes": target.stat().st_size,
            "export_sha256": file_sha256(target),
            "config": checkpoint["config"],
            "step": int(checkpoint["step"]),
            "validation_loss": float(checkpoint["validation_loss"]),
            "ema_decay": float(checkpoint["ema"]["decay"]),
            "tensor_count": len(state),
            "parameter_count": sum(p.numel() for p in portable.parameters()),
            "tensor_equality": "exact, shape/dtype/value preserved after reload",
            "original_checkpoint_unchanged": True,
            "cpu_forward_check": {
                "equal": True,
                "max_absolute_difference": float((original_logits - portable_logits).abs().max()),
                "input_shape": list(tokens.shape),
                "lengths": lengths.tolist(),
                "times": times.tolist(),
                "logits_sha256": tensor_sha256(portable_logits),
            },
            "tensors": tensor_records,
        }
        manifest["models"].append(record)
        print(f"Exported seed {seed}: {target.stat().st_size} bytes; exact tensors and CPU logits")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
