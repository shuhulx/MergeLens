# MergeLens 0.3.0 release-candidate notes

This candidate focuses on correctness, reproducibility, and inspectable evidence boundaries. It restores genuine lazy tensor iteration, makes comparison coverage and checkpoint-pair identity first-class, corrects CKA and task-vector diagnostics, and reports why every unavailable metric is missing.

The aggregate is now explicitly a hand-specified and prospectively unvalidated static-risk heuristic. Structurally unsupported comparisons suppress it while preserving raw measurements. Strategy output is a rule-based configuration to test, and is marked schema-validated only when an installed MergeKit parser accepts it.

The unfinished capability-audit surface, unused cache, and unused extras have been removed. Reports are offline HTML files with embedded Plotly and correct multi-model grouping. Package version, CLI version, docs, migration notes, and build metadata are aligned at 0.3.0.

Proposed local tag after explicit release authorization: `v0.3.0-rc1`. No tag, push, release, or publication is part of this preparation pass.
