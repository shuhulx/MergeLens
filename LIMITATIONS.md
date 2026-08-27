# Limitations

MergeLens is an experimental static-inspection tool, not a merge-outcome predictor.

- Checkpoints must be homologous enough for tensor names and exact shapes to align. Matching element counts with different shapes are rejected.
- Coverage can be partial because names are missing or shapes differ. Raw coverage remains visible, and known structural conflicts suppress the aggregate heuristic.
- Static tensor signals do not establish downstream behaviour, capability retention, safety, instruction following, or generation quality.
- Composite weights, thresholds, risk tiers, and sensitivity bands are hand-specified. They have not been prospectively calibrated against a representative dataset of merge outcomes.
- Linear CKA is optional and experimental in this workflow. It requires the same ordered calibration samples, padding-aware pooling, explicit layer alignment, and a recorded calibration identity.
- Task-vector energy requires an explicit shared base. Sign disagreement and TSV interference additionally require at least two candidate task vectors.
- Configuration diagnosis does not execute MergeKit. It honours checkpoint identities, an explicit base, and scalar full-model weights, but does not model slice assembly, gradients, layer filters, tokenizer or embedding remapping, chat templates, output dtype effects, or complete method-specific semantics.
- Source-similarity profiles are descriptive cosine measurements. They are not causal attribution and do not estimate how much a source contributed.
- Full SVD is deliberately bounded by tensor element count and matrix dimension. Resource-limited signals are reported as unavailable rather than approximated silently.
- Lazy safetensors iteration reduces accumulation but does not imply a fixed peak-memory bound. Float32 conversions, task vectors, SVD workspaces, activations, report structures, and framework overhead also consume memory.
- MergeKit parser acceptance establishes schema compatibility only. It does not establish that a proposed configuration is useful.
- Every candidate merge still requires post-merge behavioural, capability, safety, and operational evaluation on the intended use case.
