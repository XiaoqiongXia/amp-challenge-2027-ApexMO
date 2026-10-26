# Flow — AMP Challenge 2027

A library of 50,000 peptide sequences prepared for the
[AMP Challenge](https://github.com/szczurek-lab/amp-challenge-2027), with a ranked
Top 100 for experimental consideration. The sequences come from three
length-conditioned Flow Matching models. Original APEX and APEX-pathogen were
used to score the library and select the Top 100.

## Files to review

| File | Contents |
| --- | --- |
| [library.fasta](generate/library.fasta) | The 50,000-sequence library |
| [top.fasta](generate/top.fasta) | The Top 100, in priority order |
| [top100_scores.csv](generate/top100_scores.csv) | Both models' scores, final ranks and reference similarity |
| [library_ranked.csv](rankings/library_ranked.csv) | Scores and ranks for all 50,000 sequences |

The library contains unique sequences of 8–50 amino acids. All designs are
specified as linear peptides with free termini. The Top 100 are a subset of
this library and are 12–27 amino acids long.

## Reproduce the submitted files

Use Python 3.10 and [uv](https://docs.astral.sh/uv/). Run these commands from the
repository root:

```bash
uv sync --locked
uv run --locked generate
```

This rebuilds the two FASTA files and the Top 100 score table in `generate/`
from the saved sequences and predictions. It recomputes the filtering and
ranking with a fixed selection seed of 42. Repeated runs produce identical
files. **This command does not sample new sequences or rerun APEX.**

## How the Top 100 were selected

Both APEX versions scored every sequence. For each model, we summarized predicted
MIC across the same 11 bacterial targets using the median. Candidates had to
pass both models' activity and uncertainty cutoffs.

We ranked the passing candidates by APEX-pathogen score, with lower predicted
MIC first. Original APEX breaks ties, followed by the sequence hash. We then
excluded candidates with a Levenshtein ratio above 0.8 to any sequence in the
official reference set and took the first 100 that remained.

The [method description](docs/method_summary.md) gives the training-data summary,
score definitions, thresholds and selection counts. These are predicted
candidates; model agreement is not experimental confirmation.

## Generate new sequences or repeat the scoring

The three Flow checkpoints are included in [checkpoint/](checkpoint/).
For example, to generate ten new sequences of length 20:

```bash
uv run --locked --extra inference python inference/generate.py \
  --training-seed 42 --length 20 --count 10 \
  --output outputs/fresh_samples.csv
```

The script uses a GPU when available and falls back to CPU. See the
[Flow instructions](docs/flow_inference.md) for other lengths and checkpoints.
The [APEX instructions](docs/apex_scoring.md) explain how to obtain the fixed
predictor versions, run both scorers and reproduce the full ranking.

For individual strain predictions, see the
[Top 100 per-strain table](docs/top100_per_strain_predictions.csv).
Dependency and model licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
