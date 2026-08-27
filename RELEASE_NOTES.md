# MergeLens 2.0.0

## Major fixes

- Fixed tensor coverage, ordering, checkpoint attribution, and incomplete-shard handling.
- Fixed numerical edge cases involving non-finite values, zero tensors, small values, rank calculations, CKA, TSV, and SVD.
- Added dtype checks and pinned Hugging Face revisions for reproducible remote comparisons.
- Added status and reason fields for skipped, unavailable, unsupported, and failed metrics.
- Separated pair, candidate-set, and activation results so each value has a clear source.
- Fixed MergeKit configuration parsing and validation.
- Fixed HTML report escaping, CSV formula handling, and multi-model chart grouping.
- Updated MCP compatibility, package checks, and the release workflow.

## Breaking changes

Version 2.0.0 updates the result schema, CLI output, report fields, and public Python models. It also removes the unfinished audit and cache interfaces. See `MIGRATION.md` for the full list of renamed and removed fields.

## Notes

The aggregate score is an unvalidated static-risk heuristic. It does not replace downstream evaluation of a merged model.
