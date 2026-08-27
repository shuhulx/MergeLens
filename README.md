# MergeLens

[![CI](https://github.com/shuhulx/MergeLens/actions/workflows/ci.yml/badge.svg)](https://github.com/shuhulx/MergeLens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mergelens)](https://pypi.org/project/mergelens/)
[![Python](https://img.shields.io/pypi/pyversions/mergelens)](https://pypi.org/project/mergelens/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

MergeLens is an experimental inspection toolkit for homologous LLM checkpoints. It reports exact tensor coverage and weight, spectral, task-vector, and optional activation-similarity signals; highlights tensors worth inspecting; and proposes a rule-based MergeKit starting configuration. It is intended for model-merging researchers and engineers who need inspectable evidence before spending compute on candidate merges.

MergeLens does **not** establish downstream merged-model quality, capability retention, or the best merge method. Its aggregate score and thresholds are hand-specified, unvalidated heuristics. Post-merge behavioural evaluation remains necessary.

Run a local software demonstration with tiny synthetic safetensors:

```bash
pip install -e .
pip install -e '.[report]'
python examples/synthetic_demo.py --output-dir demo-output
```

The example is a software and known-answer demonstration, not scientific validation of merge outcomes.

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

The result exposes:

- the reference and candidate identity for every tensor row;
- total tensors and parameters, missing names, shape mismatches, and exact comparable coverage;
- architecture metadata and known structural incompatibilities;
- a status and reason for every computed, skipped, unavailable, failed, or resource-limited signal;
- raw parameter-weighted heuristic components and the weights renormalized for that run;
- a machine-readable `validation_status: heuristic_unvalidated`;
- pair-bounded tensor inspection regions; and
- an illustrative or parser-validated MergeKit starting configuration.

Structurally unsupported comparisons retain their raw coverage and measurements but suppress aggregate scoring.

Python API:

```python
from mergelens import compare_models

result = compare_models(["model_a/", "model_b/"])

print(result.coverage[0].parameter_coverage_reference)
print(result.mci.score)              # float or None when suppressed
print(result.mci.risk_tier)
print(result.mci.validation_status)  # heuristic_unvalidated

for row in result.tensor_metrics:
    print(row.reference_model, row.candidate_model, row.tensor_name, row.cosine_similarity)

for signal in result.metric_availability:
    print(signal.metric, signal.status.value, signal.reason)
```

## Nine underlying diagnostic signals

The composite heuristic is not counted as a separate diagnostic signal.

| Signal | Level | Direct object | Default | Composite |
|---|---|---|---|---|
| Cosine similarity | Weight | Exact-shape flattened tensor alignment | Yes | Yes |
| Normalized L2 distance | Weight | Difference relative to average tensor norm | Yes | No; displayed raw |
| Weight-distribution divergence | Weight, experimental | Directional softmax transform of flattened weights | No; explicit selection only | No |
| Spectral overlap | Weight | Leading left-singular-subspace overlap for matrices | Yes, resource bounded | Yes |
| Effective-rank ratio | Weight | Ratio of entropy-derived effective ranks | Yes, resource bounded | Yes |
| Sign disagreement | Task vector | Pairwise sign mismatch; zero/nonzero counts as mismatch | Yes when a shared base and at least two candidates exist | Yes |
| TSV interference | Task vector | Pairwise right-singular-subspace overlap | Yes when a shared base and at least two candidates exist | Yes |
| Task-vector energy | Task vector | Fraction of spectral energy in retained leading values | Yes with an explicit base | No; strategy signal only |
| Linear CKA | Activation, optional | Aligned calibration representations with recorded calibration identity | Only when supplied | Yes only when valid |

All SVD-backed signals use a conservative full-decomposition resource policy. A metric skipped by that policy is reported as `resource_limit_skipped`; it is not silently converted into a plausible number.

## MergeKit configuration diagnosis

```bash
mergelens diagnose merge.yaml --json diagnosis.json
```

Diagnosis honours checkpoint references, an explicit task-vector base, and scalar full-model weights. It discloses ignored or unsupported semantics such as slice assembly, gradients, tokenizer remapping, chat templates, and method-specific merge execution. Unknown merge methods fail closed instead of becoming `linear`.

Generated configurations follow current MergeKit model/parameter placement. They are marked `schema_validated` only when the installed MergeKit parser accepted them; otherwise they are marked `illustrative`.

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

Safetensors are memory-mapped and aligned tensor groups are consumed lazily. No exact peak-memory multiplier is claimed: runtime memory also includes float32 conversions, task vectors, bounded SVD workspaces, activation tensors, result rows, report data, and framework overhead.

See [LIMITATIONS.md](LIMITATIONS.md), [VALIDATION.md](VALIDATION.md), [MIGRATION.md](MIGRATION.md), and [CHANGELOG.md](CHANGELOG.md) before interpreting results.

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

Apache-2.0. See [LICENSE](LICENSE).
