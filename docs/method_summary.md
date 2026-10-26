# Flow-generated library with dual-APEX prioritization

## Library generation and training data

The submission uses existing outputs of a length-conditioned discrete Flow Matching generator. Three sequence-balanced models were trained with training seeds 42, 123, and 2025. Each model's checkpoint was selected using validation cross-entropy before frozen-test evaluation. Each selected checkpoint generated 50,000 sequences using EMA weights, 256 Euler steps, generation seed 52001, and the held-out test length distribution. No additional Flow generation or sequence editing was performed for this submission.

The source AMP collection contained 16,860 unique sequences from DBAASP and in-house AMP measurements, split by sequence-similarity clusters into 13,478 training, 1,693 validation, and 1,689 test sequences. The test split supplied the evaluation length distribution, not the training sequences. In-house data and full training sequences are not distributed. Exact EMA inference weights for all three Flow models are included in checkpoint/. Original APEX and APEX-pathogen weights are obtained separately from their pinned upstream repositories.

The 150,000 generation records were filtered to 8-50 residues and the 20 standard amino acids, globally deduplicated, and checked against the official antibacterial reference. This left 135,156 structurally valid unique sequences; 155 exact reference matches were removed, leaving 135,001. We selected 50,000 by sorting SHA256('amp-challenge-library-v1|42|' + sequence) ascending, with sequence as a tie-break. The library selection is independent of activity scores. All designs are specified as unmodified linear peptides with free N and C termini.

## Frozen activity predictors

Every one of the 50,000 submitted sequences was scored with both complete, pretrained ensembles:

- APEX-pathogen: 8 models, 11 outputs, revision 417a4441a1e6ef8b10d2352e1c059622d5259f3a.
- Original APEX: 40 models, 34 outputs, revision 203055125196eee0b0f3c456335d3e9aede0e942. Its 11 targets matched to the pathogen panel were used for consensus; all 34 outputs for the Top 100 are retained in docs/top100_per_strain_predictions.csv; the scorer exports all 34 for fresh runs.

Checkpoint SHA256 hashes match the corresponding frozen calibration manifests. Predicted MIC values are expressed in micromolar (uM). For each strain, MIC is the arithmetic mean of the ensemble members' MIC predictions. Activity is the median across 11 strains of log10(MIC/uM); lower is better. Uncertainty is the median across those same strains of the sample SD of ensemble-member log10(MIC/uM) predictions.

## Consensus gates and ranking

Both models must satisfy their own previously frozen reference-based activity and uncertainty cutoffs:

| Model | Activity: median log10(MIC/uM), strictly below | Uncertainty: median SD, at or below |
| --- | ---: | ---: |
| APEX-pathogen | 2.4281704039353142 | 0.35883064907437795 |
| Original APEX, matched 11 targets | 1.8882911493880563 | 0.27711947107714319 |

Activity cutoffs are the respective known-AMP median scores; uncertainty cutoffs are the corresponding 95th percentiles. These are operational computational thresholds, not calibrated probabilities of antibacterial activity. They describe a panel median and do not require activity against every individual strain.

Both activity gates passed for 19,508 library sequences. Requiring both uncertainty gates left 15,553. These candidates were ordered by APEX-pathogen median log10(MIC), then original APEX matched-11 median log10(MIC), then sequence SHA256; all ascending.

Starting at the highest-ranked consensus candidate, the exact maximum Levenshtein.ratio against all 39,448 official reference sequences was computed. Candidates with a maximum greater than 0.8 were rejected; exactly 0.8 is permitted. The first 132 ranked candidates were examined, 32 were rejected, and the first 100 passing candidates form top.fasta. FASTA record order is priority order; final_rank in top100_scores.csv is the same order. Every Top 100 sequence is present in library.fasta.
