# Third-party software and models

This repository and its Flow weights do not currently have a project-wide
open-source license. The following third-party licenses apply to the listed
components.

| Component | License and source |
| --- | --- |
| Official challenge validator | [BSD-3-Clause](licenses/amp-challenge-BSD-3-Clause.txt), Ewa Szczurek lab |
| Meta flow-matching 1.0.10 | [CC BY-NC 4.0](licenses/flow-matching.txt) |
| Original APEX | [Non-profit research use](licenses/apex.txt) |
| APEX-pathogen | [MIT](licenses/apex-pathogen.txt) |

The validator is copied unchanged from the
[challenge repository](https://github.com/szczurek-lab/amp-challenge-2027/tree/5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e),
along with its antibacterial reference file. The Flow sampler imports the
flow-matching package as a dependency. APEX model code and checkpoints are
obtained from their upstream repositories; see [the scoring instructions](docs/apex_scoring.md).

PyTorch, NumPy, pandas, PyYAML, Levenshtein and RapidFuzz retain their upstream
licenses. For the source of the included Flow code and weights, see
[the model manifest](metadata/flow_model_manifest.json).
