# Third-party notices

This is a private academic benchmark submission. No overall public redistribution
license is assigned to the project code or Flow checkpoints by this packaging
step. Training data are not distributed here.

- `scripts/official_verify_submission.py` is copied unchanged from the Ewa
  Szczurek lab AMP Challenge repository, commit
  `5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e`. Its BSD-3-Clause license is preserved
  in `licenses/amp-challenge-BSD-3-Clause.txt`. The accompanying
  `data/antibacterial.fasta` is the official benchmark reference at that commit.
- The sampler imports Meta's `flow-matching` 1.0.10 dependency; it does not
  vendor that dependency's source. Its CC BY-NC 4.0 license is copied in
  `licenses/flow-matching.txt` for reference.
- The original APEX and APEX-pathogen checkpoints are not redistributed in
  this repository. Obtain them from their pinned upstream repositories as
  documented in `docs/apex_scoring.md`. Their upstream license texts are
  retained in `licenses/apex.txt` and `licenses/apex-pathogen.txt` respectively.
- PyTorch, NumPy, pandas, PyYAML, Levenshtein and RapidFuzz remain subject to
  their own upstream licenses. The dependency lock does not relicense them.

See `docs/flow_inference.md` for source and checkpoint provenance.
