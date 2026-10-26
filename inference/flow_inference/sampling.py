"""Discrete Flow Matching utilities for length-conditioned AMP generation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from torch import Tensor, nn

from .tokenizer import FLOW_VOCAB_SIZE, MASK_TOKEN_ID


def build_probability_path(exponent: float = 1.0) -> MixtureDiscreteProbPath:
    return MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=exponent))


class AminoAcidPosteriorWrapper(ModelWrapper):
    """Expose a 21-state posterior while assigning exactly zero mass to MASK."""

    def forward(self, x: Tensor, t: Tensor, **extras: Tensor) -> Tensor:
        lengths = extras.get("lengths")
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=torch.long)
        attention_mask = extras.get("attention_mask")
        if attention_mask is None:
            positions = torch.arange(x.shape[1], device=x.device)[None]
            attention_mask = positions < lengths[:, None]
        logits = self.model(
            x_t=x,
            time=t,
            lengths=lengths,
            attention_mask=attention_mask,
        )
        amino_acid_probability = torch.softmax(logits.float(), dim=-1)
        mask_probability = torch.zeros_like(amino_acid_probability[..., :1])
        return torch.cat((amino_acid_probability, mask_probability), dim=-1)


@torch.no_grad()
def generate_token_ids(
    model: nn.Module,
    *,
    lengths: Sequence[int] | Tensor,
    steps: int,
    device: torch.device | str,
    time_epsilon: float = 1e-3,
    path_exponent: float = 1.0,
    max_length: int | None = None,
    dtype_categorical: torch.dtype = torch.float64,
) -> list[Tensor]:
    """Generate batches grouped by exact requested length."""

    if steps < 2:
        raise ValueError("steps must be at least 2")
    length_tensor = torch.as_tensor(lengths, dtype=torch.long)
    if length_tensor.ndim != 1 or length_tensor.numel() == 0:
        raise ValueError("lengths must be a non-empty one-dimensional sequence")
    if max_length is None:
        model_config = getattr(model, "config", None)
        max_length = int(getattr(model_config, "max_length", 64))
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if torch.any(length_tensor < 1) or torch.any(length_tensor > max_length):
        raise ValueError(f"requested lengths must be within [1, {max_length}]")

    path = build_probability_path(exponent=path_exponent)
    solver = MixtureDiscreteEulerSolver(
        model=AminoAcidPosteriorWrapper(model),
        path=path,
        vocabulary_size=FLOW_VOCAB_SIZE,
    )
    device = torch.device(device)
    outputs: list[Tensor | None] = [None] * len(length_tensor)
    for length in sorted(set(length_tensor.tolist())):
        positions = torch.nonzero(length_tensor == length, as_tuple=False).flatten()
        count = len(positions)
        group_lengths = torch.full((count,), length, device=device, dtype=torch.long)
        x_init = torch.full((count, length), MASK_TOKEN_ID, device=device, dtype=torch.long)
        sample = solver.sample(
            x_init=x_init,
            step_size=1.0 / steps,
            time_grid=torch.tensor([0.0, 1.0 - time_epsilon], device=device),
            dtype_categorical=dtype_categorical,
            lengths=group_lengths,
        )
        for local_index, original_index in enumerate(positions.tolist()):
            outputs[original_index] = sample[local_index].detach().cpu()
    if any(output is None for output in outputs):
        raise RuntimeError("Generation failed to populate all requested lengths")
    return [output for output in outputs if output is not None]
