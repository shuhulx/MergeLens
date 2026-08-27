# Changelog

## 2.0.0

### Correctness

- Restored lazy aligned-tensor iteration and deterministic ordering.
- Added exact tensor/parameter coverage, structural metadata, and score suppression for unsupported comparisons.
- Added explicit reference/candidate identity to every tensor measurement and every inspection region.
- Corrected linear CKA, padding-aware activation pooling, tiny-scale/zero/non-finite semantics, TSV numerical-rank normalization, and SVD resource handling.
- Added dtype validation, immutable remote revisions, complete-shard enforcement, and parameter-aware partial metric coverage.
- Wired metric selection and removed the non-functional cache surface.
- Made unknown MergeKit methods fail closed and used explicit bases in task-vector analysis and generated configs.

### Evidence boundaries

- Reframed the aggregate as a hand-specified, prospectively unvalidated static-risk heuristic.
- Replaced statistical-sounding confidence fields in 0.2.x output with evidence coverage and a non-statistical sensitivity band.
- Added structured computed, skipped, unavailable, numerical-failure, unsupported, and resource-limit statuses.
- Renamed conflict zones to tensor inspection regions and attribution to source-similarity profiles.
- Separated pair-tensor, candidate-set task-vector, and activation-layer evidence; only pair-attributable components enter pair assessments.

### Packaging and public surface

- Removed unfinished capability-audit and unused cache/extras from the public package.
- Embedded Plotly safely in offline reports, neutralized CSV formula cells, and fixed multi-model grouping.
- Pinned the supported MCP major version and hardened CI/publishing artifact validation.
- Consolidated the version at 2.0.0 and expanded reproducibility CI.
- Added limitations, validation status, migration guidance, release notes, and a synthetic worked example.
