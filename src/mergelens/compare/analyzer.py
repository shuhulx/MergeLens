"""Evidence-aware, streaming checkpoint comparison orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import torch
from rich.progress import track

from mergelens.__about__ import __version__
from mergelens.activations.cka import CKAComparison
from mergelens.compare.loader import (
    ModelHandle,
    comparison_coverage,
    find_common_tensors,
    iter_aligned_tensors,
    transformer_block_index,
)
from mergelens.compare.metrics import (
    DEFAULT_METRICS,
    DIAGNOSTIC_METRIC_NAMES,
    DegenerateMetricInputError,
    WeightDivergenceResourceLimitError,
    centered_task_vector_energy,
    compute_heuristic_assessment,
    cosine_similarity,
    effective_rank_ratio,
    l2_distance,
    sign_disagreement_rate,
    spectral_subspace_overlap,
    tsv_interference_score,
    weight_distribution_divergence,
)
from mergelens.compare.strategy import recommend_strategy
from mergelens.models import (
    ActivationMetrics,
    CandidateSetTensorMetrics,
    CompareResult,
    ComparisonCoverage,
    LayerType,
    MergeCompatibilityIndex,
    MetricAvailability,
    MetricObservation,
    MetricStatus,
    ModelRole,
    Severity,
    TensorConflictRegion,
    TensorMetrics,
)
from mergelens.utils.tensor_ops import SVDResourceLimitError, compute_task_vector, flatten_to_2d

logger = logging.getLogger(__name__)

_LEGACY_METRIC_NAMES = {
    "kl_divergence": "weight_distribution_divergence",
    "spectral_subspace_overlap": "spectral_overlap",
    "tsv_interference_score": "tsv_interference",
    "centered_task_vector_energy": "task_vector_energy",
}


def compare_models(
    model_paths: list[str],
    base_model: str | None = None,
    device: str = "cpu",
    metrics: list[str] | None = None,
    svd_rank: int = 64,
    show_progress: bool = True,
    include_strategy: bool = True,
    cka_comparisons: dict[str, CKAComparison] | None = None,
) -> CompareResult:
    """Compare two or more checkpoints using exact-shape tensor alignment.

    ``metrics=None`` selects the documented default signals. The experimental
    weight-distribution divergence must be requested explicitly. CKA values
    must come from :func:`compare_activations_cka` and be keyed by candidate
    path or candidate name.
    """

    if len(model_paths) < 2:
        raise ValueError("Need at least 2 models to compare.")
    if svd_rank < 1:
        raise ValueError("svd_rank must be at least 1.")
    requested_metrics = _normalize_metric_selection(metrics)
    if cka_comparisons and not all(
        isinstance(comparison, CKAComparison) for comparison in cka_comparisons.values()
    ):
        raise TypeError("cka_comparisons values must be provenance-bearing CKAComparison objects.")

    handles = [ModelHandle(path, device=device) for path in model_paths]
    if base_model:
        reference = ModelHandle(base_model, device=device)
        candidates = handles
        explicit_shared_base = True
    else:
        reference = handles[0]
        candidates = handles[1:]
        explicit_shared_base = False

    coverages = [
        comparison_coverage(
            reference,
            candidate,
            f"comparison_{index}",
            explicit_shared_base=explicit_shared_base,
        )
        for index, candidate in enumerate(candidates)
    ]

    rows: list[TensorMetrics] = []
    for coverage, candidate in zip(coverages, candidates):
        names = find_common_tensors([reference, candidate])
        iterator: Iterable[tuple[str, LayerType, list[torch.Tensor]]] = iter_aligned_tensors(
            [reference, candidate], names
        )
        if show_progress:
            iterator = track(
                iterator,
                total=len(names),
                description=f"Comparing {candidate.info.name}...",
            )
        for position, (name, tensor_type, tensors) in enumerate(iterator):
            reference_tensor, candidate_tensor = tensors
            block = transformer_block_index(name)
            row = TensorMetrics(
                reference_model=reference.path_or_repo,
                candidate_model=candidate.path_or_repo,
                comparison_id=coverage.comparison_id,
                tensor_name=name,
                tensor_position=position,
                transformer_block=block,
                tensor_type=tensor_type,
                shape=tuple(reference_tensor.shape),
                parameter_count=reference_tensor.numel(),
            )
            finite_weight_inputs = _compute_pair_metrics(
                row,
                reference_tensor,
                candidate_tensor,
                requested_metrics=requested_metrics,
                svd_rank=svd_rank,
                explicit_shared_base=explicit_shared_base,
            )
            if not finite_weight_inputs:
                coverage.scoring_supported = False
                condition = (
                    "Aggregate scoring is suppressed because at least one exact-shape tensor "
                    "contains NaN or infinite values."
                )
                if condition not in coverage.unsupported_conditions:
                    coverage.unsupported_conditions.append(condition)
            rows.append(row)

    if not rows:
        details = "; ".join(
            f"{item.comparison_id}: {', '.join(item.unsupported_conditions) or 'no comparable tensors'}"
            for item in coverages
        )
        raise ValueError(f"No exact-shape-compatible tensor comparisons are available. {details}")

    candidate_set_metrics = _compute_cross_candidate_metrics(
        reference,
        candidates,
        requested_metrics=requested_metrics,
        svd_rank=svd_rank,
        explicit_shared_base=explicit_shared_base,
        show_progress=show_progress,
    )
    activation_metrics = _collect_activation_metrics(
        reference,
        candidates,
        coverages,
        requested_metrics=requested_metrics,
        cka_comparisons=cka_comparisons,
    )

    availability = _summarize_metric_availability(rows)
    if candidate_set_metrics:
        availability = _replace_metric_availability(
            availability,
            _summarize_metric_availability(
                candidate_set_metrics,
                tuple(
                    metric
                    for metric in ("sign_disagreement_rate", "tsv_interference")
                    if metric in requested_metrics
                ),
            ),
        )
    if activation_metrics:
        availability = _replace_metric_availability(
            availability,
            [
                MetricAvailability(
                    metric="cka_similarity",
                    status=MetricStatus.COMPUTED,
                    computed_tensor_count=len(activation_metrics),
                    affected_tensor_count=len(activation_metrics),
                    reason="Computed as activation-layer evidence; not broadcast to tensor rows or used in the aggregate.",
                )
            ],
        )
    pair_assessments: dict[str, MergeCompatibilityIndex] = {}
    for coverage in coverages:
        pair_rows = [row for row in rows if row.comparison_id == coverage.comparison_id]
        pair_availability = _summarize_metric_availability(pair_rows)
        pair_assessments[coverage.comparison_id] = compute_heuristic_assessment(
            pair_rows,
            pair_availability,
            scoring_supported=coverage.scoring_supported,
        )
    mci = _conservative_overall_assessment(pair_assessments, availability, coverages)

    regions: list[TensorConflictRegion] = []
    for coverage in coverages:
        pair_rows = [row for row in rows if row.comparison_id == coverage.comparison_id]
        regions.extend(_detect_tensor_conflict_regions(pair_rows))

    if explicit_shared_base:
        reference_info = reference.info_with_role(ModelRole.EXPLICIT_SHARED_BASE)
        model_infos = [handle.info_with_role(ModelRole.CANDIDATE) for handle in handles]
        explicit_base_info = reference_info
    else:
        reference_info = reference.info_with_role(ModelRole.IMPLICIT_REFERENCE)
        model_infos = [reference_info] + [
            handle.info_with_role(ModelRole.CANDIDATE) for handle in candidates
        ]
        explicit_base_info = None

    result = CompareResult(
        models=model_infos,
        reference_model=reference_info,
        explicit_base=explicit_base_info,
        coverage=coverages,
        tensor_metrics=rows,
        candidate_set_metrics=candidate_set_metrics,
        activation_metrics=activation_metrics,
        tensor_conflict_regions=regions,
        mci=mci,
        pair_assessments=pair_assessments,
        metric_availability=availability,
        metadata={
            "mergelens_version": __version__,
            "device": device,
            "svd_rank": svd_rank,
            "requested_metrics": sorted(requested_metrics),
            "total_diagnostic_signals": len(DIAGNOSTIC_METRIC_NAMES),
            "explicit_shared_base": explicit_shared_base,
            "aggregate_rule": "minimum_pair_score",
            "cka_provenance": {
                key: {
                    "calibration_id": comparison.calibration_id,
                    "sample_count": comparison.sample_count,
                    "pooling_rule": comparison.pooling_rule,
                    "aligned_layers": list(comparison.aligned_layers),
                    "feature_dimensions": comparison.feature_dimensions,
                    "warnings": list(comparison.warnings),
                }
                for key, comparison in (cka_comparisons or {}).items()
            },
            "streaming_note": (
                "Tensor groups are iterated lazily. Peak memory also includes float32 conversions, "
                "task vectors, bounded SVD workspaces, result rows, and framework overhead."
            ),
        },
    )
    if include_strategy and result.mci.score is not None:
        result.strategy = recommend_strategy(result)
    return result


def _normalize_metric_selection(metrics: list[str] | None) -> frozenset[str]:
    if metrics is None:
        return DEFAULT_METRICS
    normalized = {_LEGACY_METRIC_NAMES.get(name, name) for name in metrics}
    unknown = normalized - set(DIAGNOSTIC_METRIC_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Supported metrics: {', '.join(DIAGNOSTIC_METRIC_NAMES)}"
        )
    return frozenset(normalized)


def _set_observation(
    row: TensorMetrics | CandidateSetTensorMetrics,
    metric: str,
    status: MetricStatus,
    reason: str | None = None,
) -> None:
    row.metric_observations[metric] = MetricObservation(status=status, reason=reason)


def _compute_pair_metrics(
    row: TensorMetrics,
    reference_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    *,
    requested_metrics: frozenset[str],
    svd_rank: int,
    explicit_shared_base: bool,
) -> bool:
    for metric in DIAGNOSTIC_METRIC_NAMES:
        if metric not in requested_metrics:
            _set_observation(
                row, metric, MetricStatus.SKIPPED_BY_USER, "Not selected for this run."
            )
    for metric, reason in (
        (
            "sign_disagreement_rate",
            "Candidate-set task-vector signal requiring an explicit shared base and at least two candidates; see candidate_set_metrics.",
        ),
        (
            "tsv_interference",
            "Candidate-set task-vector signal requiring an explicit shared base and at least two candidates; see candidate_set_metrics.",
        ),
        (
            "cka_similarity",
            "Activation-layer signal; no aligned calibration activations are stored on tensor rows; see activation_metrics.",
        ),
    ):
        if metric in requested_metrics:
            _set_observation(row, metric, MetricStatus.STRUCTURALLY_UNAVAILABLE, reason)

    finite_weight_inputs = bool(
        torch.isfinite(reference_tensor).all() and torch.isfinite(candidate_tensor).all()
    )
    if not finite_weight_inputs:
        reason = "Reference or candidate tensor contains NaN or infinite values."
        for metric in requested_metrics - {
            "cka_similarity",
            "sign_disagreement_rate",
            "tsv_interference",
        }:
            _set_observation(row, metric, MetricStatus.UNSUPPORTED_INPUT, reason)

    if finite_weight_inputs and "cosine_similarity" in requested_metrics:
        row.cosine_similarity = cosine_similarity(reference_tensor, candidate_tensor)
        _set_observation(row, "cosine_similarity", MetricStatus.COMPUTED)
    if finite_weight_inputs and "l2_distance" in requested_metrics:
        row.l2_distance = l2_distance(reference_tensor, candidate_tensor)
        _set_observation(row, "l2_distance", MetricStatus.COMPUTED)

    if finite_weight_inputs and "weight_distribution_divergence" in requested_metrics:
        try:
            row.weight_distribution_divergence = weight_distribution_divergence(
                reference_tensor, candidate_tensor
            )
            _set_observation(row, "weight_distribution_divergence", MetricStatus.COMPUTED)
        except WeightDivergenceResourceLimitError as exc:
            _set_observation(
                row,
                "weight_distribution_divergence",
                MetricStatus.RESOURCE_LIMIT_SKIPPED,
                str(exc),
            )

    for metric, function, attribute in (
        ("spectral_overlap", spectral_subspace_overlap, "spectral_overlap"),
        ("effective_rank_ratio", effective_rank_ratio, "effective_rank_ratio"),
    ):
        if metric not in requested_metrics or not finite_weight_inputs:
            continue
        if min(flatten_to_2d(reference_tensor).shape) < 2:
            _set_observation(
                row,
                metric,
                MetricStatus.STRUCTURALLY_UNAVAILABLE,
                "Vectors and one-row matrices do not provide an informative spectral comparison.",
            )
            continue
        try:
            value = (
                function(reference_tensor, candidate_tensor, k=svd_rank)
                if metric == "spectral_overlap"
                else function(reference_tensor, candidate_tensor)
            )
            setattr(row, attribute, value)
            _set_observation(row, metric, MetricStatus.COMPUTED)
        except SVDResourceLimitError as exc:
            _set_observation(row, metric, MetricStatus.RESOURCE_LIMIT_SKIPPED, str(exc))
        except DegenerateMetricInputError as exc:
            _set_observation(row, metric, MetricStatus.STRUCTURALLY_UNAVAILABLE, str(exc))
        except RuntimeError as exc:
            logger.warning("Numerical failure in %s for %s: %s", metric, row.tensor_name, exc)
            _set_observation(row, metric, MetricStatus.FAILED_NUMERICALLY, str(exc))

    if finite_weight_inputs and "task_vector_energy" in requested_metrics:
        if not explicit_shared_base:
            _set_observation(
                row,
                "task_vector_energy",
                MetricStatus.STRUCTURALLY_UNAVAILABLE,
                "An explicit shared base is required to interpret the difference as a task vector.",
            )
        else:
            try:
                task_vector = compute_task_vector(candidate_tensor, reference_tensor)
                row.task_vector_energy = centered_task_vector_energy(task_vector, k=svd_rank)
                _set_observation(row, "task_vector_energy", MetricStatus.COMPUTED)
            except SVDResourceLimitError as exc:
                _set_observation(
                    row, "task_vector_energy", MetricStatus.RESOURCE_LIMIT_SKIPPED, str(exc)
                )
            except DegenerateMetricInputError as exc:
                _set_observation(
                    row, "task_vector_energy", MetricStatus.STRUCTURALLY_UNAVAILABLE, str(exc)
                )
            except RuntimeError as exc:
                logger.warning(
                    "Numerical failure in task_vector_energy for %s: %s", row.tensor_name, exc
                )
                _set_observation(
                    row, "task_vector_energy", MetricStatus.FAILED_NUMERICALLY, str(exc)
                )

    return finite_weight_inputs


def _compute_cross_candidate_metrics(
    reference: ModelHandle,
    candidates: list[ModelHandle],
    *,
    requested_metrics: frozenset[str],
    svd_rank: int,
    explicit_shared_base: bool,
    show_progress: bool,
) -> list[CandidateSetTensorMetrics]:
    selected = requested_metrics & {"sign_disagreement_rate", "tsv_interference"}
    if not selected or not explicit_shared_base or len(candidates) < 2:
        return []

    common_names = find_common_tensors([reference, *candidates])
    iterator: Iterable[tuple[str, LayerType, list[torch.Tensor]]] = iter_aligned_tensors(
        [reference, *candidates], common_names
    )
    if show_progress:
        iterator = track(iterator, total=len(common_names), description="Comparing task vectors...")
    result: list[CandidateSetTensorMetrics] = []
    for position, (name, tensor_type, tensors) in enumerate(iterator):
        base_tensor, candidate_tensors = tensors[0], tensors[1:]
        row = CandidateSetTensorMetrics(
            candidate_set_id="candidate_set_0",
            base_model=reference.path_or_repo,
            candidate_models=[candidate.path_or_repo for candidate in candidates],
            tensor_name=name,
            tensor_position=position,
            transformer_block=transformer_block_index(name),
            tensor_type=tensor_type,
            shape=tuple(base_tensor.shape),
            parameter_count=base_tensor.numel(),
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            reason = (
                "A base or candidate tensor required for the cross-candidate signal contains "
                "NaN or infinite values."
            )
            for metric in selected:
                _set_observation(row, metric, MetricStatus.UNSUPPORTED_INPUT, reason)
            result.append(row)
            continue
        task_vectors = [compute_task_vector(tensor, base_tensor) for tensor in candidate_tensors]
        if "sign_disagreement_rate" in selected:
            row.sign_disagreement_rate = sign_disagreement_rate(task_vectors)
            _set_observation(row, "sign_disagreement_rate", MetricStatus.COMPUTED)
        if "tsv_interference" in selected:
            try:
                row.tsv_interference = tsv_interference_score(task_vectors, k=svd_rank)
                _set_observation(row, "tsv_interference", MetricStatus.COMPUTED)
            except SVDResourceLimitError as exc:
                _set_observation(
                    row, "tsv_interference", MetricStatus.RESOURCE_LIMIT_SKIPPED, str(exc)
                )
            except DegenerateMetricInputError as exc:
                _set_observation(
                    row, "tsv_interference", MetricStatus.STRUCTURALLY_UNAVAILABLE, str(exc)
                )
            except RuntimeError as exc:
                logger.warning("Numerical failure in TSV interference for %s: %s", name, exc)
                _set_observation(row, "tsv_interference", MetricStatus.FAILED_NUMERICALLY, str(exc))
        result.append(row)
    return result


def _collect_activation_metrics(
    reference: ModelHandle,
    candidates: list[ModelHandle],
    coverages: list[ComparisonCoverage],
    *,
    requested_metrics: frozenset[str],
    cka_comparisons: dict[str, CKAComparison] | None,
) -> list[ActivationMetrics]:
    if "cka_similarity" not in requested_metrics or not cka_comparisons:
        return []
    result: list[ActivationMetrics] = []
    for coverage, candidate in zip(coverages, candidates):
        comparison = next(
            (
                cka_comparisons[key]
                for key in (candidate.path_or_repo, candidate.info.name)
                if key in cka_comparisons
            ),
            None,
        )
        if comparison is None:
            continue
        for layer, score in comparison.scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"CKA score for {layer} must be in [0, 1], got {score}.")
            widths = comparison.feature_dimensions[layer]
            result.append(
                ActivationMetrics(
                    comparison_id=coverage.comparison_id,
                    reference_model=reference.path_or_repo,
                    candidate_model=candidate.path_or_repo,
                    activation_layer=layer,
                    cka_similarity=score,
                    calibration_id=comparison.calibration_id,
                    sample_count=comparison.sample_count,
                    pooling_rule=comparison.pooling_rule,
                    reference_feature_count=widths[0],
                    candidate_feature_count=widths[1],
                    warnings=list(comparison.warnings),
                )
            )
    return result


def _summarize_metric_availability(
    rows: Sequence[TensorMetrics | CandidateSetTensorMetrics],
    metric_names: tuple[str, ...] = DIAGNOSTIC_METRIC_NAMES,
) -> list[MetricAvailability]:
    summaries: list[MetricAvailability] = []
    priority = (
        MetricStatus.FAILED_NUMERICALLY,
        MetricStatus.RESOURCE_LIMIT_SKIPPED,
        MetricStatus.UNSUPPORTED_INPUT,
        MetricStatus.STRUCTURALLY_UNAVAILABLE,
        MetricStatus.SKIPPED_BY_USER,
    )
    for metric in metric_names:
        observations = [row.metric_observations[metric] for row in rows]
        computed = sum(item.status == MetricStatus.COMPUTED for item in observations)
        affected_parameters = sum(row.parameter_count for row in rows)
        computed_parameters = sum(
            row.parameter_count
            for row, observation in zip(rows, observations)
            if observation.status == MetricStatus.COMPUTED
        )
        parameter_coverage = (
            computed_parameters / affected_parameters if affected_parameters else None
        )
        if computed:
            unavailable_count = len(observations) - computed
            reason = (
                f"Computed for {computed}/{len(observations)} tensor comparisons; "
                f"{unavailable_count} had a documented unavailable or skipped status."
                if unavailable_count
                else None
            )
            summaries.append(
                MetricAvailability(
                    metric=metric,
                    status=MetricStatus.COMPUTED,
                    reason=reason,
                    computed_tensor_count=computed,
                    affected_tensor_count=len(observations),
                    computed_parameter_count=computed_parameters,
                    affected_parameter_count=affected_parameters,
                    parameter_coverage=parameter_coverage,
                )
            )
            continue
        status = next(
            candidate
            for candidate in priority
            if any(item.status == candidate for item in observations)
        )
        reasons = sorted({item.reason for item in observations if item.reason})
        summaries.append(
            MetricAvailability(
                metric=metric,
                status=status,
                reason="; ".join(reasons) if reasons else None,
                affected_tensor_count=len(observations),
                affected_parameter_count=affected_parameters,
                parameter_coverage=parameter_coverage,
            )
        )
    return summaries


def _replace_metric_availability(
    current: list[MetricAvailability], replacements: list[MetricAvailability]
) -> list[MetricAvailability]:
    by_name = {item.metric: item for item in replacements}
    return [by_name.get(item.metric, item) for item in current]


def _conservative_overall_assessment(
    pair_assessments: dict[str, MergeCompatibilityIndex],
    availability: list[MetricAvailability],
    coverages: list[ComparisonCoverage],
) -> MergeCompatibilityIndex:
    if not all(coverage.scoring_supported for coverage in coverages):
        return MergeCompatibilityIndex(
            score=None,
            risk_tier="insufficient_evidence",
            evidence_coverage=0.0,
            available_metrics=[
                item.metric for item in availability if item.status == MetricStatus.COMPUTED
            ],
            unavailable_metrics=[
                item for item in availability if item.status != MetricStatus.COMPUTED
            ],
            notes=[
                "Aggregate scoring was suppressed because at least one pair has an unsupported structural condition."
            ],
        )
    scored = [
        assessment for assessment in pair_assessments.values() if assessment.score is not None
    ]
    if len(scored) != len(pair_assessments):
        return MergeCompatibilityIndex(
            score=None,
            risk_tier="insufficient_evidence",
            available_metrics=[
                item.metric for item in availability if item.status == MetricStatus.COMPUTED
            ],
            unavailable_metrics=[
                item for item in availability if item.status != MetricStatus.COMPUTED
            ],
        )
    conservative = min(scored, key=lambda assessment: float(assessment.score or 0.0)).model_copy(
        deep=True
    )
    conservative.available_metrics = [
        item.metric for item in availability if item.status == MetricStatus.COMPUTED
    ]
    conservative.unavailable_metrics = [
        item for item in availability if item.status != MetricStatus.COMPUTED
    ]
    conservative.partial_metrics = [
        item
        for item in availability
        if item.status == MetricStatus.COMPUTED and (item.parameter_coverage or 0.0) < 1.0
    ]
    conservative.notes.append(
        "For multi-candidate runs, the aggregate score is the lowest pair score, not an average."
    )
    return conservative


def _detect_tensor_conflict_regions(
    rows: list[TensorMetrics],
    cosine_threshold: float = 0.80,
    minimum_region_size: int = 1,
) -> list[TensorConflictRegion]:
    """Identify ordered tensor ranges triggered by a heuristic cosine threshold."""

    regions: list[TensorConflictRegion] = []
    current: list[TensorMetrics] = []
    for row in rows:
        if row.cosine_similarity is not None and row.cosine_similarity < cosine_threshold:
            current.append(row)
        else:
            if len(current) >= minimum_region_size:
                regions.append(_build_tensor_region(current))
            current = []
    if len(current) >= minimum_region_size:
        regions.append(_build_tensor_region(current))
    return regions


def _build_tensor_region(rows: list[TensorMetrics]) -> TensorConflictRegion:
    cosine_values = [
        float(row.cosine_similarity) for row in rows if row.cosine_similarity is not None
    ]
    average_cosine = sum(cosine_values) / len(cosine_values)
    if average_cosine < 0.5:
        severity = Severity.CRITICAL
    elif average_cosine < 0.7:
        severity = Severity.HIGH
    elif average_cosine < 0.8:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW
    triggers = [f"mean cosine similarity {average_cosine:.4f} below heuristic 0.80 threshold"]
    return TensorConflictRegion(
        comparison_id=rows[0].comparison_id,
        reference_model=rows[0].reference_model,
        candidate_model=rows[0].candidate_model,
        start_tensor_position=rows[0].tensor_position,
        end_tensor_position=rows[-1].tensor_position,
        tensor_names=[row.tensor_name for row in rows],
        severity=severity,
        avg_cosine_similarity=round(average_cosine, 4),
        avg_sign_disagreement=None,
        triggering_signals=triggers,
        heuristic_inspection_note=(
            "Inspection priority only. Review these exact tensors and evaluate any merge-method or "
            "weight changes after merging; this static signal does not establish that a particular "
            "method will improve behavior."
        ),
    )


_detect_conflict_zones = _detect_tensor_conflict_regions
