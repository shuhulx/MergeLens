# Changelog

## 2.0.2

- Polished the README, limitations, validation notes, and migration guide.

There are no API or behavior changes in this release.

## 2.0.1

- Changed the PyPI development status from Alpha to Beta.
- Simplified the package description and documentation.

There are no API or behavior changes in this release.

## 2.0.0 - Major fixes

### Comparison results

- Fixed tensor iteration, ordering, coverage, and model-pair attribution.
- Fixed NaN, infinity, zero-value, tiny-scale, rank, CKA, TSV, and SVD edge cases.
- Added dtype checks, complete-shard checks, and pinned Hugging Face revisions.
- Added status and reason fields for metrics that are skipped or unavailable.
- Separated pair metrics, candidate-set metrics, and activation metrics.
- Suppressed aggregate scores when a comparison is structurally unsupported.

### MergeKit and reports

- Fixed explicit-base handling and configuration weight parsing.
- Made unknown merge methods return an error.
- Validated generated configurations with the installed MergeKit parser.
- Fixed offline report escaping, CSV formula handling, and multi-model chart grouping.
- Updated MCP compatibility and release checks.

### Breaking changes

- Replaced `layer_metrics` with `tensor_metrics`.
- Replaced `conflict_zones` with `tensor_conflict_regions`.
- Replaced confidence fields with evidence coverage and a sensitivity band.
- Moved sign and TSV results to `candidate_set_metrics`.
- Moved CKA results to `activation_metrics`.
- Removed the unfinished audit command, audit extra, cache, and `--no-cache` option.
- Updated serialized result fields, CLI output, reports, and generated configuration metadata.

See `MIGRATION.md` for the complete field mapping.
