# Changelog

## 0.3.0 - release candidate

### Correctness

- Restored lazy aligned-tensor iteration and deterministic ordering.
- Added exact tensor/parameter coverage, structural metadata, and score suppression for unsupported comparisons.
- Added explicit reference/candidate identity to every tensor measurement and every inspection region.
- Corrected linear CKA, padding-aware activation pooling, TSV rank normalization, and SVD resource handling.
- Wired metric selection and removed the non-functional cache surface.
- Made unknown MergeKit methods fail closed and used explicit bases in task-vector analysis and generated configs.

### Evidence boundaries

- Reframed the aggregate as a hand-specified, prospectively unvalidated static-risk heuristic.
- Replaced statistical-sounding confidence fields in serialized v0.3 output with evidence coverage and a non-statistical sensitivity band.
- Added structured computed, skipped, unavailable, numerical-failure, unsupported, and resource-limit statuses.
- Renamed conflict zones to tensor inspection regions and attribution to source-similarity profiles.

### Packaging and public surface

- Removed unfinished capability-audit and unused cache/extras from the public package.
- Embedded Plotly in offline reports and fixed multi-model grouping.
- Consolidated the version at 0.3.0 and expanded reproducibility CI.
- Added limitations, validation status, migration guidance, release notes, and a synthetic worked example.
