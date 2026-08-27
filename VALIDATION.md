# Validation status

## What the test suite covers

The automated tests cover:

- tensor matching, ordering, coverage, shape and dtype handling, and pinned remote revisions;
- implicit references, explicit bases, multi-model comparisons, and correct model-pair attribution;
- known-answer cases for cosine, L2, sign, TSV, task-vector energy, numerical rank, scale invariance, zero-rank tensors, scalar tensors, non-finite values, and SVD limits;
- separation of pair, candidate-set, and activation metrics;
- linear CKA against an independent NumPy implementation, including alignment checks, unequal feature widths, degenerate inputs, padding-aware pooling, and high-dimensional warnings;
- selective metric execution and clear reasons for skipped or unavailable results;
- aggregate-score suppression when the available evidence is not valid for scoring;
- MergeKit parsing, explicit-base handling, unknown methods, and the stated limits of configuration diagnosis;
- offline reports, HTML escaping, safe CSV output, and multi-model chart grouping; and
- package building, wheel contents, clean installation, dependency checks, the CLI, and MCP server creation.

These tests show that the software behaves as expected for the covered cases and supported environments.

## What has not been validated

MergeLens has not yet been tested as a predictor of real merge outcomes. The synthetic example only checks deterministic software behaviour.

- The aggregate weights, thresholds, risk tiers, and sensitivity bands have not been calibrated.
- The suggested MergeKit starting rules have not been calibrated.
- There is no held-out evaluation across model families, architectures, scales, or fine-tuning methods.
- False-positive and false-negative rates have not been measured on representative merges.
- There are no external user studies or independent replications yet.
- MergeLens does not predict capability, safety, or instruction-following retention.

For now, treat the raw measurements and coverage as the main output. The aggregate score and strategy suggestions remain labelled `heuristic_unvalidated` and should be checked against the finished model.
