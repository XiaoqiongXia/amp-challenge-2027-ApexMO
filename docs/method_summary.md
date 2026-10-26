# Method

## Abstract

We prepared a 50,000-sequence peptide library using a length-conditioned discrete
Flow Matching model trained on AMP sequences from DBAASP and in-house
measurements. Samples from three training runs were pooled, deduplicated and
filtered to meet the challenge's sequence requirements. The library was selected
without using activity scores. We then evaluated every sequence with original
APEX and APEX-pathogen. The Top 100 were selected from candidates that passed
both models' activity and uncertainty thresholds, ranked primarily by
APEX-pathogen, and filtered against the official antibacterial reference.
The repository provides the sequences, model weights and scripts for sampling,
scoring and ranking.

## Training and generation

The source collection contained 16,860 unique AMP sequences. A split based on
sequence-similarity clusters assigned 13,478 sequences to training, 1,693 to
validation and 1,689 to testing. The training sequences are not included in this
repository.

Three models were trained with seeds 42, 123 and 2025. Each checkpoint was chosen
by validation cross-entropy before test evaluation. Each model generated 50,000
sequences using exponential moving average (EMA) weights, 256 Euler steps and
sampling seed 52001. Requested lengths followed the held-out test length
distribution. The [three EMA checkpoints](../checkpoint/) are included, with
sampling instructions in [flow_inference.md](flow_inference.md).

## Building the 50K library

We used the existing 150,000 generation records, without additional sampling or
sequence editing. Keeping only the 20 standard amino acids and lengths of 8–50,
then removing duplicates, left 135,156 unique sequences. Removing 155 exact
matches to the [official reference](../data/antibacterial.fasta) left 135,001.

To choose the 50,000-member library, we sorted candidates by
`SHA256('amp-challenge-library-v1|42|' + sequence)` and took the first 50,000,
using the sequence itself to break any tie. This selection is independent of
APEX scores. All designs are specified as unmodified linear peptides with free
N and C termini.

## APEX scores

Every sequence was evaluated with both complete pretrained ensembles:

| Predictor | Ensemble members | Targets used for selection |
| --- | ---: | --- |
| Original APEX | 40 | The 11 targets matching the APEX-pathogen panel, out of 34 outputs |
| APEX-pathogen | 8 | All 11 outputs |

The [scoring instructions](apex_scoring.md) give the exact upstream versions and
commands. Checkpoint hashes are recorded in
[model_provenance.json](../scoring/model_provenance.json).

For each strain, predicted MIC is the arithmetic mean of the ensemble members'
MIC predictions, in µM. A model's activity score is the median of
`log10(MIC / µM)` across the 11 selected strains; lower is better. Its uncertainty
score is the median across those same strains of the sample standard deviation
of the members' log10 MIC predictions.

We applied cutoffs fixed from the known-AMP reference distributions. Activity
cutoffs are the median reference scores; uncertainty cutoffs are the 95th
percentiles. Both models must pass both tests:

| Model | Activity score < | Uncertainty score ≤ |
| --- | ---: | ---: |
| Original APEX, matched 11 targets | 1.8882911493880563 | 0.2771194710771432 |
| APEX-pathogen | 2.4281704039353142 | 0.35883064907437795 |

The scripts read these values from the [original APEX](../data/legacy_selected11_thresholds.json)
and [APEX-pathogen](../data/pathogen_thresholds.json) threshold files.
These are screening thresholds for the panel median, rather than probabilities
of activity or a requirement to be active against every strain.

## Selecting the Top 100

Of the 50,000 sequences, 19,508 passed both activity thresholds. Applying both
uncertainty thresholds left 15,553. We sorted these candidates in ascending
order by APEX-pathogen activity score, original APEX activity score and sequence
SHA256, in that order.

Starting at the top of this list, we compared each candidate against all 39,448
official reference sequences using `Levenshtein.ratio`. Candidates with a maximum
ratio above 0.8 were excluded; exactly 0.8 was allowed. We examined 132 candidates
and excluded 32 to obtain the final 100.

[Top 100 FASTA](../generate/top.fasta) record order and `final_rank` in the
[score table](../generate/top100_scores.csv) give the same priority order. Every
Top 100 sequence belongs to the [50K library](../generate/library.fasta).
The [per-strain table](top100_per_strain_predictions.csv) retains all 34 original
APEX outputs and all 11 APEX-pathogen outputs for these candidates.
