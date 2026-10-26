# Run both APEX models and reproduce the sequence ranking

The portable scoring commands run actual neural inference using the frozen
upstream APEX checkpoints. They preserve the original project scoring wrappers,
verify upstream revisions and every checkpoint hash, and calculate the same
matched-11 activity and uncertainty aggregates used for the submitted ranking.
External predictor weights are downloaded separately and are not bundled here.

## Install and obtain the exact upstream models

Use Python 3.10 or later, from this submission repository:

```bash
python -m pip install -r scoring/requirements.txt
mkdir -p external
git clone https://gitlab.com/machine-biology-group-public/apex.git external/apex
git -C external/apex checkout 203055125196eee0b0f3c456335d3e9aede0e942
git clone https://gitlab.com/machine-biology-group-public/apex-pathogen.git external/apex-pathogen
git -C external/apex-pathogen checkout 417a4441a1e6ef8b10d2352e1c059622d5259f3a
```

The legacy repository must contain `AMP_DL_model_twohead.py`, `best_key_list`,
and all 40 files in `trained_models/`. The pathogen repository must contain
`APEX_models.py` and all eight files in `APEX_pathogen_models/`. Expected names
and SHA256 hashes are recorded in `scoring/model_provenance.json`. Repository
revision, model-definition code hash, and every model-file hash are checked
before loading. These upstream checkpoint files contain serialized model
objects and are loaded using the same original project mechanism; use the
verified upstream files.

The legacy APEX license is restricted to **non-profit research purposes**;
see the upstream [license at the frozen revision](https://gitlab.com/machine-biology-group-public/apex/-/blob/203055125196eee0b0f3c456335d3e9aede0e942/LICENSE).
APEX-pathogen uses the [MIT license](https://gitlab.com/machine-biology-group-public/apex-pathogen/-/blob/417a4441a1e6ef8b10d2352e1c059622d5259f3a/LICENSE).
The scoring directory contains project-owned wrappers, not redistributed
upstream model code or checkpoints, and does not relicense upstream materials.

## Score the frozen 50,000-sequence library

```bash
python scoring/run_apex.py legacy \
  --model-root external/apex \
  --output-dir outputs/apex_legacy \
  --device auto

python scoring/run_apex.py pathogen \
  --model-root external/apex-pathogen \
  --output-dir outputs/apex_pathogen \
  --device auto
```

The default input is `data/library_50000.csv`. Use `--input path/to/sequences.csv`
to score another unique, canonical 8–50-residue sequence list. A `sequence`
column is required. Missing candidate identifiers and SHA256 sequence hashes
are generated deterministically. `--device auto` selects CUDA when available
and CPU otherwise; `--device cpu` explicitly selects the fallback. Defaults
match the original submission: batch size 3000, seed 42, four CPU threads,
deterministic PyTorch algorithms, TF32 disabled, and cuDNN benchmarking disabled.

Each backend writes:

- `scores.csv`: all strain-specific MICs, sample SDs of log10 MIC, and the
  original backend aggregates.
- `aggregate_scores.csv`: candidate keys plus the four matched-11 aggregates
  needed for the consensus calculation.
- `input_sequences.csv`, `runtime.json`, `manifest.json`, `summary.json`, and
  `ensemble_state.npz`: normalized inputs and resumable inference provenance.

Rerunning with identical inputs and runtime settings resumes after the last
completed ensemble member. A changed input, model set, device, or runtime
contract requires a new output directory. Full 50K inference is substantially
faster on a GPU; CPU is supported for portability and small checks.

## Rank all 50K and select the consensus Top 100

To reproduce ranking from the supplied frozen aggregate scores, without
rerunning the neural models:

```bash
python scoring/rank.py --output-dir outputs/ranked_frozen
```

To rank newly computed complete score tables instead:

```bash
python scoring/rank.py \
  --legacy-scores outputs/apex_legacy/scores.csv \
  --pathogen-scores outputs/apex_pathogen/scores.csv \
  --output-dir outputs/ranked_rescored
```

The two `aggregate_scores.csv` files can also be supplied to these same options.
Alternatively, `--aggregate path/to/dual_apex_scores.csv` accepts a combined
aggregate table. Input keys must match the 50,000-row library exactly.
`--library` can point to a relocated library file, but that file must satisfy
the frozen generator's complete schema, identifiers, ordering, and seed-42
selection-key constraints; this ranking CLI does not accept arbitrary generated
sequence lists. All library uniqueness, length, residue, hash, selection-order,
and reference-exclusion checks are reused from the frozen generator. The
separate `run_apex.py` scoring command does accept other canonical unique
sequence lists.

Outputs are:

- `library_ranked.csv`: all 50,000 sequences sorted by APEX-pathogen activity,
  then legacy matched-11 activity, then sequence SHA256, all ascending.
  `library_rank` is the full-library rank. Gate flags are retained;
  `raw_consensus_rank` and `final_rank` are populated only where applicable.
- `consensus_ranked.csv`: the sequences passing both models' activity and
  uncertainty gates, with their consensus ranks.
- `top100_scores.csv` and `top.fasta`: the first 100 consensus candidates that
  also satisfy the official reference-similarity limit.
- `ranking_metadata.json`: counts, ranking semantics, input and output hashes,
  and whether the selected FASTA matches the frozen submission exactly.

The complete-library ranking does not imply that every sequence passes the
activity gates. Filtering and Top 100 selection are separate from assigning
all-library ranks. The script reuses `rank_consensus` and `select_top` from
`src/flow_dual_apex_submission/generate.py` and refuses to overwrite existing
ranked outputs.

## Exact score and selection definitions

Both models first predict each strain's MIC for every ensemble member. The
strain-specific MIC is the **arithmetic ensemble mean in micromolar units**.
Activity is the median of log10 of these means across 11 strains. Uncertainty
is the median of the **sample SDs** of member-level log10 MIC across those same
11 strains. Lower activity scores correspond to lower predicted MIC.

For legacy APEX, the exact 11 target names are read from
`data/legacy_selected11_thresholds.json`. The legacy scorer also retains its
original 15-pathogen and 34-strain summary columns in `scores.csv`; those broader
summaries are not used for dual-model consensus. For APEX-pathogen, all its 11
outputs are used. Original upstream spellings (including AIG/AIC and PAO1/PA01)
are retained, and per-backend target names are resolved explicitly.

The frozen gates are:

| Model | Activity, strictly below | Median log10-MIC SD, at most |
| --- | ---: | ---: |
| Legacy APEX, matched 11 | 1.8882911493880563 | 0.2771194710771432 |
| APEX-pathogen | 2.4281704039353142 | 0.35883064907437795 |

These are frozen reference-distribution screening cutoffs, not calibrated
probabilities of antibacterial activity. Candidates must pass all four gates.
The consensus is sorted by APEX-pathogen activity, legacy matched-11 activity,
and sequence SHA256. Candidates are then examined in order against every
official reference sequence; only those with maximum `Levenshtein.ratio <= 0.8`
are accepted until 100 remain. The supplied frozen scores yield 15,553 consensus
candidates, and 132 are examined to obtain the submitted Top 100.

## Verification and numerical reproducibility

Both portable wrappers were run on CPU for two archived candidates through all
40 legacy and eight pathogen models. Every strain-specific and aggregate
prediction agreed with the archived CUDA scores at `rtol=1e-5, atol=1e-6`.
Recomputing ranking from all frozen aggregates reproduced the original Top 100
FASTA exactly. Small numerical differences can occur across CPU/GPU, PyTorch
versions, and batch sizes; fresh rescoring is not promised to produce bytewise
identical floating-point tables or unchanged decisions for candidates extremely
close to a cutoff. The default frozen generator remains the route for exact
reconstruction of the submitted sequences and order.

Related-model agreement does not constitute independent experimental validation.
