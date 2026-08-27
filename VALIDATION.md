# Validation status

## Verified software properties

The automated suite checks:

- exact-shape rejection, including same-element-count tensors with different shapes;
- deterministic tensor ordering with a full-name tie-breaker;
- explicit missing-name, shape/dtype-mismatch, parameter, architecture, and immutable remote-revision coverage;
- lazy iterator consumption without progress-path materialisation;
- pair identity for implicit-reference, explicit-base, and multi-model comparisons;
- pair-bounded tensor inspection regions and report grouping;
- selective metric execution and structured unavailability reasons;
- known-answer cosine, L2, sign, TSV, task-energy, numerical-rank trimming, scale invariance, zero-rank semantics, scalar tensors, non-finite rejection, and SVD resource-policy behaviour;
- parameter-weighted partial metric availability and aggregate suppression for unsupported numeric evidence;
- exact separation of pair-tensor, candidate-set, and activation-layer evidence;
- linear CKA against an independent NumPy implementation, tiny-scale and orthogonal-transform invariance, unequal feature widths, sample-count/alignment validation, calibration identity, degeneracy rejection, and high-dimensional warnings;
- padding-aware activation pooling;
- fail-closed unknown MergeKit methods and configuration-scope disclosure;
- actual use of the explicit base in task-vector construction;
- generated YAML acceptance by the supported MergeKit parser in the validation job;
- offline report generation with HTML-safe embedded chart specifications and spreadsheet-safe CSV export with status/reason columns;
- package build, artifact inspection, fresh-wheel installation, dependency checks, MCP creation, and CLI smoke tests.

These tests establish software properties within their fixtures and supported environments.

## Scientific validation completed

No prospective scientific validation of downstream merge prediction has been completed. The synthetic example demonstrates deterministic software behaviour only.

## Scientific validation not yet completed

- prospective prediction of behavioural merge outcomes;
- calibration of aggregate weights, score thresholds, risk tiers, or sensitivity bands;
- calibration of merge-method starting rules;
- evaluation across held-out checkpoint families, architectures, scales, and fine-tuning recipes;
- false-positive and false-negative measurement on representative merges;
- external user studies or independent replication;
- prediction of capability, safety, or instruction-following retention.

Until those studies exist, raw measurements and coverage are the primary evidence. The composite and strategy outputs must remain labelled `heuristic_unvalidated` and treated as hypotheses for post-merge testing.
