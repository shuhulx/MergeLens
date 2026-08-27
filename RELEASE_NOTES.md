# MergeLens 2.0.0 release notes

This release focuses on correctness, reproducibility, and inspectable evidence boundaries. It restores genuine lazy tensor iteration, makes comparison coverage and checkpoint-pair identity first-class, separates pair, candidate-set, and activation evidence, corrects numerical-rank and tiny-scale handling, and reports why every unavailable metric is missing.

The aggregate is now explicitly a hand-specified and prospectively unvalidated static-risk heuristic using only pair-attributable cosine, spectral-overlap, and effective-rank components. Structurally or numerically unsupported comparisons suppress it, and partial metric availability reduces evidence weight by parameter coverage. Strategy output is a low-authority rule-based configuration to test and is marked schema-validated only when an installed MergeKit parser accepts it.

The unfinished capability-audit surface, unused cache, and unused extras have been removed. Reports use HTML-safe embedded Plotly, CSV exports neutralize spreadsheet formulas, remote checkpoints are revision-pinned and shard-complete, exported results record their MergeLens version, and the MCP dependency is constrained to its tested compatible range. Package version, CLI version, docs, migration notes, and build metadata are aligned at 2.0.0.

The `2.0.0` major-version boundary reflects the intentionally breaking result-schema, CLI, configuration, and public API changes documented in `MIGRATION.md`. No tag, GitHub release, or package publication is part of this branch update.
