# Generate new peptides with Flow

The three trained Flow models are included in `checkpoint/`. These instructions
sample new peptides from them. To rebuild the submitted 50K library and Top 100
from the saved sequences and scores, use `uv run --locked generate` instead.

## Run a model

Run this from the repository root. `uv` installs the model dependencies recorded
in the lockfile:

```bash
uv run --locked --extra inference python inference/generate.py \
  --training-seed 42 \
  --length 20 \
  --count 10 \
  --steps 256 \
  --seed 52001 \
  --batch-size 512 \
  --device auto \
  --output outputs/fresh_samples.csv
```

This generates ten 20-residue peptides. Change `--length` to any integer from
8 to 50, and `--count` to the number of samples you need. `--device auto` uses
CUDA when available and otherwise CPU; `--device cpu` always runs on CPU.

Choose the model with `--training-seed`:

| Training seed | Checkpoint |
| --- | --- |
| 42 | `checkpoint/flow_seed42_step3499_ema.pt` |
| 123 | `checkpoint/flow_seed123_step3499_ema.pt` |
| 2025 | `checkpoint/flow_seed2025_step3499_ema.pt` |

All three contain the EMA weights from training step 3499. Each is about 29.7 MB
and keeps the original float32 precision. The full configuration and checkpoint
hashes are in the [model manifest](../metadata/flow_model_manifest.json).

`--seed` controls sampling and is separate from the training seed. The archived
runs used sampling seed 52001 and 256 steps for all three models.

## Use the output

The command writes a CSV and a `.metadata.json` file with the checkpoint, seed,
settings and package versions. Choose a new output filename for each run;
existing files are not overwritten.

Samples can repeat and have not yet been scored or filtered against the official
reference. Deduplicate before following the [APEX scoring instructions](apex_scoring.md).
The [method summary](method_summary.md) describes how we assembled the submitted
library from the original samples.

New samples can change with hardware, package versions or batching, even with the
same seed. For the exact submitted sequences and order, use the saved-submission
command at the top of this page.

The sampler uses Meta's [flow-matching](https://github.com/facebookresearch/flow_matching)
library under [CC BY-NC 4.0](https://github.com/facebookresearch/flow_matching/blob/main/LICENSE).
The [checkpoint export helper](../inference/export_checkpoints.py) is provided
for maintainers with the original training checkout; it is not needed to run
the included models.
