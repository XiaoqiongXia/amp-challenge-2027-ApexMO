# Score sequences with APEX and reproduce the ranking

Run these commands from the repository root. `uv` installs the locked model
dependencies automatically.

## Recalculate the ranking

Using the saved predictions:

```bash
uv run --locked --extra inference python scoring/rank.py \
  --output-dir outputs/ranked_frozen
```

This writes:

| File | Contents |
| --- | --- |
| `library_ranked.csv` | All 50,000 sequences in score order, with pass/fail flags |
| `consensus_ranked.csv` | The 15,553 sequences passing both models' activity and uncertainty cutoffs |
| `top100_scores.csv` and `top.fasta` | The first 100 consensus candidates that pass the reference-similarity check |
| `ranking_metadata.json` | Counts, input/output hashes and comparison with the submitted Top 100 |

The rank columns cover all 50K (`library_rank`), the consensus
(`raw_consensus_rank`) and the Top 100 (`final_rank`). Outputs are not overwritten.

## Download the models

The APEX weights are kept in their original repositories. These are the exact
revisions used for this submission:

```bash
mkdir -p external
git clone https://gitlab.com/machine-biology-group-public/apex.git external/apex
git -C external/apex checkout 203055125196eee0b0f3c456335d3e9aede0e942
git clone https://gitlab.com/machine-biology-group-public/apex-pathogen.git external/apex-pathogen
git -C external/apex-pathogen checkout 417a4441a1e6ef8b10d2352e1c059622d5259f3a
```

Legacy APEX uses 40 checkpoints; APEX-pathogen uses eight. The script checks
repository revisions and the SHA256 hashes of model code and every checkpoint in
[model_provenance.json](../scoring/model_provenance.json).

Legacy APEX is licensed for
[non-profit research](https://gitlab.com/machine-biology-group-public/apex/-/blob/203055125196eee0b0f3c456335d3e9aede0e942/LICENSE).
APEX-pathogen uses the
[MIT license](https://gitlab.com/machine-biology-group-public/apex-pathogen/-/blob/417a4441a1e6ef8b10d2352e1c059622d5259f3a/LICENSE).

## Run both models

```bash
uv run --locked --extra inference python scoring/run_apex.py legacy \
  --model-root external/apex \
  --output-dir outputs/apex_legacy \
  --device auto

uv run --locked --extra inference python scoring/run_apex.py pathogen \
  --model-root external/apex-pathogen \
  --output-dir outputs/apex_pathogen \
  --device auto
```

The default input is `data/library_50000.csv`. To score another list, add
`--input path/to/sequences.csv`. It must have a `sequence` column containing
unique, uppercase sequences of 8–50 standard amino acids. Missing IDs and hashes
are filled in automatically.

`--device auto` selects CUDA or falls back to CPU; `--device cpu` forces CPU.
Defaults match the original run: batch size 3000, seed 42 and deterministic
inference settings.

Each run saves per-strain predictions in `scores.csv`, matched-11 summaries in
`aggregate_scores.csv`, and settings and progress files. Repeat the same command
to resume. Changed inputs or settings need a new output directory.

## Rank the new predictions

For predictions recomputed on the submitted 50K library:

```bash
uv run --locked --extra inference python scoring/rank.py \
  --legacy-scores outputs/apex_legacy/scores.csv \
  --pathogen-scores outputs/apex_pathogen/scores.csv \
  --output-dir outputs/ranked_rescored
```

The same options accept `aggregate_scores.csv`; `--aggregate` accepts a combined
table. Scores must match the library's IDs, sequences and hashes exactly.

The ranking command expects the submitted 50K library's schema, identifiers,
ordering and seed-42 selection keys. `--library` lets you change that file's
location; it does not make the ranking command accept arbitrary new samples.
The scoring command above can score other sequence lists.

## How the order is chosen

Both models use the same 11-target panel. Activity is the median log10 of each
strain's arithmetic ensemble-mean MIC, in µM; lower is better. Uncertainty is the
median sample SD of member-level log10 MIC across those targets. Legacy APEX's
broader summaries are excluded from consensus.

Candidates must pass both models' activity and uncertainty cutoffs. All ranking
uses APEX-pathogen activity first, legacy matched-11 activity second, and
sequence SHA256 to break ties, all ascending. The Top 100 are taken from the
consensus in that order, accepting only candidates whose maximum
`Levenshtein.ratio` against the official reference is at most 0.8.

The exact target names and cutoffs are in the
[legacy thresholds](../data/legacy_selected11_thresholds.json) and
[pathogen thresholds](../data/pathogen_thresholds.json); see the
[method summary](method_summary.md) for the selection details. Fresh inference
can produce small numerical differences across hardware or package versions.
Use the saved predictions when you need the exact submitted order.
