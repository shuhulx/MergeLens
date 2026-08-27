# Migrating from 0.2.x to 2.0.0

Version 2.0.0 changed the result schema, CLI output, configuration handling, and public Python API.

| 0.2.x concept | 2.0.0 replacement |
|---|---|
| `layer_metrics` | `tensor_metrics` with reference, candidate, comparison ID, tensor position, and optional transformer block |
| group sign/TSV copied into pair rows | `candidate_set_metrics`, attributable to one explicit base and complete candidate set |
| block-broadcast CKA on tensor rows | `activation_metrics`, attributable to exact activation layers with calibration and feature-width provenance |
| `conflict_zones` | `tensor_conflict_regions` with exact tensor names and heuristic trigger signals |
| compatibility verdict | `risk_tier`, which describes static signals only |
| confidence | `evidence_coverage`, a non-statistical fraction of available heuristic signal weight |
| confidence interval | `heuristic_band_lower` / `heuristic_band_upper`, a non-statistical sensitivity display |
| KL divergence | experimental `weight_distribution_divergence`, excluded from default execution and aggregation |
| attribution map | non-causal `source_similarity_profiles` |
| cache / `--no-cache` | removed; the cache was never completed |
| capability-audit command and extra | removed; the feature was never completed |

Common result fields still accept their old constructor names for now. Serialized 2.0.0 results, CLI output, and reports use only the new names.

Only the metrics you name are calculated:

```python
result = compare_models(models, metrics=["cosine_similarity", "l2_distance"])
```

Activation CKA must be passed as `CKAComparison` objects returned by
`compare_activations_cka`. Add them to the `cka_comparisons` mapping; plain
score dictionaries are no longer supported.

Unknown MergeKit methods now return a validation error instead of falling back to linear merging.
