# MergeLens

[![CI](https://github.com/shuhulx/MergeLens/actions/workflows/ci.yml/badge.svg)](https://github.com/shuhulx/MergeLens/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-176%20passed-brightgreen.svg)](https://github.com/shuhulx/MergeLens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mergelens?cacheSeconds=300)](https://pypi.org/project/mergelens/)
[![GitHub](https://img.shields.io/github/v/tag/shuhulx/MergeLens?label=github&cacheSeconds=300)](https://github.com/shuhulx/MergeLens/tags)
[![Downloads](https://static.pepy.tech/badge/mergelens/month)](https://pepy.tech/project/mergelens)
[![Python](https://img.shields.io/pypi/pyversions/mergelens)](https://pypi.org/project/mergelens/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/shuhulx/mergelens/blob/main/LICENSE)

MergeLens helps you inspect compatible LLM checkpoints before merging them. It checks tensor coverage and calculates weight, spectral, task-vector, and optional activation-similarity metrics. It can also draft a MergeKit configuration.

The metrics can point you toward tensors worth investigating, but they do not predict whether a merged model will perform well. The aggregate score uses hand-set rules that have not been validated against real merge outcomes. Always evaluate the merged model itself.

To try it without downloading a model, run the synthetic example:

```bash
pip install -e .
pip install -e '.[report]'
python examples/synthetic_demo.py --output-dir demo-output
```

The example checks that the software works. It is not evidence that the scores predict merge quality.

## Install

```bash
pip install mergelens
pip install 'mergelens[report]'  # offline HTML reports
pip install 'mergelens[mcp]'     # MCP server
```

## Compare checkpoints

```bash
mergelens compare model_a/ model_b/
mergelens compare finetune_a/ finetune_b/ --base shared_base/
mergelens compare model_a/ model_b/ --metric cosine_similarity --metric l2_distance --json result.json
mergelens compare model_a/ model_b/ --report report.html
```

Each result includes:

- the reference and candidate for every tensor comparison.
- separate candidate-set metrics for sign/TSV and activation metrics for CKA.
- tensor and parameter counts, missing names, shape or dtype mismatches, and comparable coverage.
- architecture details and known structural conflicts.
- a status and reason for metrics that were skipped, unavailable, failed, or limited by resources.
- the raw components used by the aggregate heuristic.
- a machine-readable `validation_status: heuristic_unvalidated`.
- tensor regions that may deserve closer inspection.
- a MergeKit starting configuration, marked as either illustrative or parser-validated.

When model structures are incompatible, MergeLens still reports coverage and raw measurements but does not calculate an aggregate score.

Python API:

```python
from mergelens import compare_models

result = compare_models(["model_a/", "model_b/"])

print(result.coverage[0].parameter_coverage_reference)
print(result.mci.score)  # float or None when suppressed
print(result.mci.risk_tier)
print(result.mci.validation_status)  # heuristic_unvalidated

for row in result.tensor_metrics:
    print(row.reference_model, row.candidate_model, row.tensor_name, row.cosine_similarity)

for row in result.candidate_set_metrics:
    print(row.base_model, row.candidate_models, row.tensor_name, row.sign_disagreement_rate)

for row in result.activation_metrics:
    print(row.comparison_id, row.activation_layer, row.cka_similarity, row.warnings)

for signal in result.metric_availability:
    print(signal.metric, signal.status.value, signal.reason)
```

## Diagnostic signals

MergeLens calculates nine underlying signals. The aggregate heuristic combines some of them and is not counted as a tenth signal.

| Signal | Level | Direct object | Default | Composite |
|---|---|---|---|---|
| Cosine similarity | Weight | Exact-shape flattened tensor alignment | Yes | Yes |
| Normalized L2 distance | Weight | Difference relative to average tensor norm | Yes | No; displayed raw |
| Weight-distribution divergence | Weight, experimental | Directional softmax transform of flattened weights | No; explicit selection only | No |
| Spectral overlap | Weight | Leading left-singular-subspace overlap for matrices | Yes, resource bounded | Yes |
| Effective-rank ratio | Weight | Ratio of entropy-derived effective ranks | Yes, resource bounded | Yes |
| Sign disagreement | Candidate set task vector | Pairwise sign mismatch; zero/nonzero counts as mismatch | Yes when a shared base and at least two candidates exist | No |
| TSV interference | Candidate set task vector | Pairwise numerical-rank right-subspace overlap | Yes when a shared base and at least two candidates exist | No |
| Task-vector energy | Pair tensor task vector | Fraction of spectral energy in retained leading values | Yes with an explicit base | No |
| Linear CKA | Activation layer, optional | Exact activation-layer observations with calibration and feature-width provenance | Only when supplied | No |

MergeLens only runs full SVDs on tensors within its size limits. It keeps numerical-rank directions and treats a subspace that fills the whole ambient space as uninformative. Tensors over the limit are reported as `resource_limit_skipped`; MergeLens does not substitute an approximation.

## MergeKit configuration diagnosis

```bash
mergelens diagnose merge.yaml --json diagnosis.json
```

Diagnosis reads checkpoint references, an explicit task-vector base, and non-negative scalar weights from top-level full-model inputs. It does not model slices, gradients, tokenizer changes, chat templates, or every method-specific option. Unknown merge methods return an error instead of being treated as `linear`.

Generated configurations follow the current MergeKit model and parameter layout. A configuration is marked `schema_validated` only when the installed MergeKit parser accepts it; otherwise it is marked `illustrative`.

## Reports and MCP

`mergelens[report]` produces one HTML file with Plotly JavaScript embedded. Charts group by explicit comparison ID and preserve missing metric values.

The MCP server exposes seven tools: `compare_models`, `diagnose_merge`, `get_conflict_zones`, `suggest_strategy`, `generate_report`, `explain_layer`, and `get_compatibility_score`.

```json
{
  "mcpServers": {
    "mergelens": {
      "command": "mergelens",
      "args": ["serve"]
    }
  }
}
```

## Memory and reproducibility

Safetensors are memory-mapped and read lazily, so MergeLens does not keep every tensor in memory at once. Peak memory still depends on float32 conversions, task vectors, SVD workspaces, activations, result data, reports, and framework overhead.

See [limitations](https://github.com/shuhulx/mergelens/blob/main/LIMITATIONS.md), [validation status](https://github.com/shuhulx/mergelens/blob/main/VALIDATION.md), [migration guidance](https://github.com/shuhulx/mergelens/blob/main/MIGRATION.md), and the [changelog](https://github.com/shuhulx/mergelens/blob/main/CHANGELOG.md) before interpreting results.

## Development

```bash
python -m pip install -e '.[dev,all]'
ruff check .
ruff format --check .
pytest -q
mypy src/mergelens
python -m build
```

Supported Python versions are 3.10, 3.11, and 3.12.

## References

- MergeKit documentation and schema: [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)
- Kornblith et al., “Similarity of Neural Network Representations Revisited”: [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Yadav et al., “TIES-Merging”: [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)
- Gargiulo et al., “Task Singular Vectors”: [arXiv:2412.00081](https://arxiv.org/abs/2412.00081)

## License

Apache-2.0. See the [license](https://github.com/shuhulx/mergelens/blob/main/LICENSE).
