# Frozen Flow checkpoints and fresh model inference

The repository includes the exact exponential moving average (EMA) inference
weights from the three Flow training runs used to generate the archived source
pool. Each file is approximately 29.7 MB and retains the original float32
precision. Optimizer state, training RNG state, and non-EMA model weights are
excluded. No training sequences or APEX/APEX-pathogen model weights are included.

| Training seed | Frozen step | EMA checkpoint |
| --- | --- | --- |
| 42 | 3499 | `checkpoint/flow_seed42_step3499_ema.pt` |
| 123 | 3499 | `checkpoint/flow_seed123_step3499_ema.pt` |
| 2025 | 3499 | `checkpoint/flow_seed2025_step3499_ema.pt` |

The frozen final submission and model inference serve different reproducibility
purposes. The root `generate` command reconstructs the submitted library from
archived data and scores. The CLI below actually loads the model and generates
fresh samples. Fresh samples are not claimed to reproduce the archived library
byte for byte: changes in device, package versions, requested length ordering,
batch composition, or RNG consumption can change stochastic outputs. This CLI
does not modify or replace the submitted library or Top 100.

## Run fresh inference

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -r inference/requirements.txt
python inference/generate.py \
  --training-seed 42 \
  --length 20 \
  --count 10 \
  --steps 256 \
  --seed 52001 \
  --batch-size 512 \
  --device auto \
  --output outputs/fresh_samples.csv
```

`--device auto` selects CUDA when available and otherwise CPU. Use `--device cpu`
to reproduce the CPU checks. The sampling seed is independent of the training
seed; the archived 50,000-record runs used sampling seed 52001 for all three
models. Select training seed 123 or 2025 to load the other frozen models.

The CLI accepts lengths 8–50 inclusive, defaults to 256 solver steps, and writes
a CSV plus a JSON metadata sidecar. It refuses to overwrite either output file.
The metadata records the checkpoint and output SHA256 hashes, seeds, device,
package versions, batch size, and solver parameters. It samples the requested
number of sequences; sequences can repeat, and no activity prediction, reference
exclusion, or ranking is performed by this inference command. Challenge filters
must be applied separately before treating fresh samples as a new submission.

For a fast execution check only, use `--count 3 --steps 4 --device cpu`. Four steps
are not the archived generation setting and are not used for the submitted
sequences. All generation inputs are exact-length masks; no prompt sequences or
training-data files are required.

## Export provenance and verification

`metadata/flow_model_manifest.json` records source checkpoint and source CSV
hashes, exported file hashes, all tensor names/shapes/dtypes/hashes, the full
original training configuration, and original source-code hashes. The original
model configuration is preserved in every exported checkpoint. The portable
loader uses `torch.load(..., weights_only=True)`.

During export, the original repository's `load_checkpoint_model(..., use_ema=True)`
was used to materialize the reference model. Every exported tensor was checked
against that model and the EMA source, reloaded, and compared exactly. All three
models gave bitwise-identical CPU outputs between the original and portable
implementations on synthetic inputs at lengths 8, 20, and 50 with mixed tokens,
mask tokens, padding, and times 0, 0.5, and 0.999. The maximum absolute logit
difference was zero. Original checkpoint hashes were rechecked after export.

For maintainers with the original research checkout, the same export can be
performed into a new destination without changing the original files:

```bash
python inference/export_checkpoints.py \
  --source-root /path/to/original/amp_design \
  --output-root /path/to/new/export
```

This export utility requires the original `amp_design` environment. It reads
trusted original training checkpoints with the original loader, including its
full checkpoint format. It is not needed for normal inference. All tensors retain
their precision; no quantization or numerical transformation is used.

## Source ownership and dependencies

The portable `inference/flow_inference/model.py` and `tokenizer.py` are exact
copies of the corresponding project source files. `sampling.py` contains the
inference functions from the project's `flow_matching.py`; unused training-loss
functions and their imports were removed. The inference CLI and export utility
are submission-specific wrappers. No overall project license has been assigned
by these export scripts.

The sampler imports, but does not vendor, Meta's
[`flow-matching` library](https://github.com/facebookresearch/flow_matching),
version 1.0.10. Its copyright and
[`CC BY-NC 4.0` license](https://github.com/facebookresearch/flow_matching/blob/main/LICENSE)
remain with the upstream owners; that license restricts its use to noncommercial
purposes. PyTorch and NumPy likewise retain their respective upstream licenses.
No dependency's source code or weights are relicensed by this repository.

The exported weights and submission code do not grant rights to any original
training dataset or external predictor. This private academic submission does
not set a public redistribution license for the project or its checkpoints.
