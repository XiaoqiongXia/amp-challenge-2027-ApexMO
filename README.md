# Flow library with dual-APEX ranking — AMP Challenge 2027

This private benchmark submission contains 50,000 unique Flow-generated peptide
sequences and a ranked Top 100 selected by agreement between original APEX and
APEX-pathogen. It includes the three frozen Flow EMA checkpoints, runnable model
inference, scoring and ranking code, and the archived predictions needed to
reconstruct the exact submitted files.

## Submission files

| File | Contents |
| --- | --- |
| [`generate/library.fasta`](generate/library.fasta) | 50,000 unique sequences, 8–50 residues |
| [`generate/top.fasta`](generate/top.fasta) | Top 100, in priority order; lengths 12–27 |
| [`generate/top100_scores.csv`](generate/top100_scores.csv) | Final ranks, both model scores, gate decisions and reference similarity |
| [`data/dual_apex_scores.csv`](data/dual_apex_scores.csv) | Aggregate predictions and gate decisions for all 50,000 sequences |
| [`rankings/library_ranked.csv`](rankings/library_ranked.csv) | All 50,000 in score order, with consensus flags and Top 100 membership |
| [`rankings/consensus_ranked.csv`](rankings/consensus_ranked.csv) | All 15,553 candidates passing both activity and uncertainty gates, in priority order |
| [`docs/top100_per_strain_predictions.csv`](docs/top100_per_strain_predictions.csv) | Per-strain predictions for the Top 100 |
| [`checkpoint/`](checkpoint/) | Three frozen EMA checkpoints, training seeds 42, 123 and 2025 |

All sequences are specified as unmodified linear peptides with free N and C
termini. The 50K library has no exact matches in the pinned official antibacterial
reference. Every Top 100 candidate belongs to the library and has a maximum
Levenshtein ratio of at most 0.8 against that reference.

## Reconstruct the exact submission

Use Python 3.10 and [uv](https://docs.astral.sh/uv/). From this checkout:

```bash
uv run --locked generate
```

The default command validates the archived inputs, recomputes the two-model
gates, sorts the candidates, checks reference similarity, and writes the FASTA
files and ranking table to `generate/`. The fixed default seed is 42. Repeated
runs produce identical FASTA files.

This entry point reconstructs the frozen submission from saved sequences and
predictions; it does not run neural inference. Fresh Flow sampling and fresh
APEX scoring are separate commands below. Fresh samples are not expected to
match the archived 50K byte for byte.

## Run the models

```bash
uv run --extra inference python inference/generate.py \
  --training-seed 42 --length 20 --count 10 --device auto \
  --output outputs/fresh_samples.csv
```

The actual Flow inference CLI loads the included EMA checkpoint. It uses CUDA
when available and otherwise CPU. See [`docs/flow_inference.md`](docs/flow_inference.md)
for the frozen settings, other checkpoints and reproducibility checks.

APEX model checkpoints remain in their original upstream repositories. The
scoring instructions pin their revisions and explain how to score with both
ensembles and repeat the ranking: [`docs/apex_scoring.md`](docs/apex_scoring.md).

To recalculate the complete ranking from the archived predictions:

```bash
uv run --extra inference python scoring/rank.py --output-dir outputs/ranked
```

`rankings/` contains the archived results of this ranking command. The canonical
submission files remain in `generate/`. Numerical values and Top 100 order agree;
CSV floating-point display formatting may differ between the two workflows.

## Method and training-data summary

The generator is a length-conditioned discrete Flow Matching model. The source
collection contained 16,860 unique AMP sequences from DBAASP and in-house
measurements, split by sequence-similarity clusters into 13,478 training, 1,693
validation and 1,689 test sequences. Three training seeds (42, 123, 2025) supplied
150,000 archived generation records. Training sequences are not distributed in
this repository.

After restricting lengths to 8–50, checking the canonical alphabet,
deduplicating and excluding exact official-reference matches, 135,001 candidates
remained. A fixed SHA256 ordering selected the 50K library independently of
activity predictions. Both complete APEX ensembles scored all 50,000 candidates.
The consensus uses the same 11-target panel for each model.

| Predictor | Activity: median log10(MIC/µM) < | Uncertainty: median ensemble SD ≤ |
| --- | ---: | ---: |
| Original APEX, matched 11 targets | 1.8882911493880563 | 0.2771194710771432 |
| APEX-pathogen | 2.4281704039353142 | 0.35883064907437795 |

The activity intersection contains 19,508 candidates. Applying both frozen
uncertainty gates leaves 15,553. Candidates are ranked by ascending APEX-pathogen
activity score, then original APEX activity score, then sequence SHA256. Of the
first 132, 32 failed the stricter reference-similarity rule; the remaining 100
form the submission. Full definitions and provenance are in
[`docs/method_summary.md`](docs/method_summary.md).

Model agreement is computational prioritization, not experimental confirmation.
The entire 50K library is not labeled antibacterial. No experimental MIC,
hemolysis or mammalian-toxicity validation is claimed.

## Validation and scope

Rules, reference and the copied official validator are pinned to
[`szczurek-lab/amp-challenge-2027` at `5c8a5d8`](https://github.com/szczurek-lab/amp-challenge-2027/tree/5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e).
Input/output checksums, threshold provenance and model tensor checks are retained
in `data/manifest.json`, `generate/reconstruction_summary.json` and `metadata/`.

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/official_verify_submission.py \
  https://github.com/XiaoqiongXia/amp-challenge-2027-flow-dual-apex \
  --dir outputs/remote_validation
```

The second command requires permission to clone this private repository.
This package supports the benchmark submission; it does not claim to meet the
separate public-repository, permissive-license and full-training-data-disclosure
requirements for co-authorship eligibility. Third-party rights are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
