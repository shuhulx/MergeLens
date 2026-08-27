# Limitations

MergeLens inspects checkpoints. It does not predict the quality of a finished merge.

- The checkpoints need compatible tensor names, shapes, and floating-point dtypes. Tensors with different shapes or non-floating dtypes are skipped. Supported floating dtypes are compared in float32, and the conversion is recorded.
- Coverage may be partial when tensor names or shapes differ. MergeLens reports the missing coverage and withholds the aggregate score when it finds a known structural conflict.
- Tensor metrics cannot tell you whether the merged model will retain capabilities, follow instructions, behave safely, or generate good output.
- The aggregate weights, thresholds, risk tiers, and sensitivity band are hand-set. They have not been calibrated against a representative collection of real merge outcomes.
- Linear CKA is optional and does not affect the aggregate score. It requires matching, ordered calibration samples and exact layer identity. Results can be misleading when there are far more features than samples, so high-dimensional comparisons need a matched baseline.
- Task-vector energy needs an explicit shared base. Sign disagreement and TSV interference also need at least two candidate models.
- Configuration diagnosis reads a limited part of a MergeKit file. It does not run the merge or fully model slices, gradients, layer filters, tokenizer changes, chat templates, output dtypes, or every method-specific option.
- Source-similarity profiles are cosine measurements, not causal attribution. They do not measure how much a source model contributed.
- Full SVD is limited by tensor size and matrix dimensions. Signals that exceed those limits are reported as unavailable instead of being silently approximated.
- Sign and TSV metrics describe the full candidate set relative to one base. They are not pairwise scores and do not select a merge method on their own.
- Hugging Face inputs are pinned to one commit for a run. Local directories can still change, so archive them separately if you need exact reproduction.
- Lazy tensor loading reduces memory use but does not set a fixed upper bound. Conversions, task vectors, SVD workspaces, activations, reports, and framework overhead also use memory.
- Passing the MergeKit parser only means that the generated configuration matches the schema. It does not mean the configuration will produce a useful model.

Always test a completed merge on the behaviour, capabilities, safety checks, and operating conditions that matter for your use case.
