# Migrating from 0.2.x to 2.0.0

Version 2.0.0 makes evidence boundaries explicit and removes unfinished surface area. The major-version boundary reflects the breaking result-schema, CLI, configuration, and public API changes below.

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
| cache / `--no-cache` | removed; no correctness-preserving cache was implemented |
| capability-audit command and extra | removed; the subsystem was not implemented |

Deprecated Python constructor aliases remain temporarily for common result fields, but 2.0.0 serialization uses only the new names. CLI and report output use the new terminology.

Metric selection now executes only the named signals:

```python
result = compare_models(models, metrics=["cosine_similarity", "l2_distance"])
```

Activation CKA enters checkpoint comparison only through provenance-bearing
`CKAComparison` objects returned by `compare_activations_cka`, passed in the
`cka_comparisons` mapping. Plain score dictionaries are no longer accepted.

Unknown MergeKit methods now raise a validation error instead of being treated as linear merging.
