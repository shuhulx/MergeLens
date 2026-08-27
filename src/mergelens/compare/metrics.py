"""Static comparison metrics with explicit shape and resource policies."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from mergelens.models import (
    MergeCompatibilityIndex,
    MetricAvailability,
    MetricStatus,
    TensorMetrics,
)
from mergelens.utils.tensor_ops import (
    bounded_full_svd,
    bounded_singular_values,
    effective_rank,
    flatten_to_2d,
    grassmann_distance,
)

METRIC_REGISTRY: dict[str, Callable[..., float]] = {}

DIAGNOSTIC_METRIC_NAMES: tuple[str, ...] = (
    "cosine_similarity",
    "l2_distance",
    "weight_distribution_divergence",
    "spectral_overlap",
    "effective_rank_ratio",
    "sign_disagreement_rate",
    "tsv_interference",
    "task_vector_energy",
    "cka_similarity",
)

DEFAULT_METRICS: frozenset[str] = frozenset(
    {
        "cosine_similarity",
        "l2_distance",
        "spectral_overlap",
        "effective_rank_ratio",
        "sign_disagreement_rate",
        "tsv_interference",
        "task_vector_energy",
        "cka_similarity",
    }
)

MAX_ELEMENTS_FOR_WEIGHT_DIVERGENCE = 1_000_000


class WeightDivergenceResourceLimitError(ValueError):
    """Raised when the experimental softmax divergence would allocate too much."""


class NonFiniteTensorError(ValueError):
    """Raised when a diagnostic receives NaN or infinite tensor values."""


class DegenerateMetricInputError(ValueError):
    """Raised when a mathematically undefined metric input is encountered."""


def register_metric(name: str):
    """Register one underlying diagnostic signal."""

    def decorator(func: Callable[..., float]) -> Callable[..., float]:
        METRIC_REGISTRY[name] = func
        return func

    return decorator


def _require_same_shape(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Exact tensor shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


def _require_finite(*tensors: torch.Tensor) -> None:
    for index, tensor in enumerate(tensors):
        if not bool(torch.isfinite(tensor).all()):
            raise NonFiniteTensorError(
                f"Tensor input {index} contains NaN or infinite values; the metric is unavailable."
            )


def _scaled_float(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return a finite float tensor scaled by its maximum absolute value."""

    _require_finite(tensor)
    values = tensor.reshape(-1).float()
    maximum = float(values.abs().max()) if values.numel() else 0.0
    if maximum == 0.0:
        return values, 0.0
    return values / maximum, maximum


@register_metric("cosine_similarity")
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of exact-shape flattened tensors.

    Two all-zero tensors are treated as identical (1.0). A zero tensor and a
    non-zero tensor have undefined angular similarity and are reported as 0.0.
    """

    _require_same_shape(a, b)
    _require_finite(a, b)
    a_flat, scale_a = _scaled_float(a)
    b_flat, scale_b = _scaled_float(b)
    if scale_a == 0.0 and scale_b == 0.0:
        return 1.0
    if scale_a == 0.0 or scale_b == 0.0:
        return 0.0
    norm_a = torch.linalg.vector_norm(a_flat)
    norm_b = torch.linalg.vector_norm(b_flat)
    result = float(torch.dot(a_flat / norm_a, b_flat / norm_b))
    if not math.isfinite(result):
        raise NonFiniteTensorError("Cosine computation produced a non-finite value.")
    return max(-1.0, min(1.0, result))


@register_metric("l2_distance")
def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 distance normalized by the tensors' average L2 norm."""

    _require_same_shape(a, b)
    _require_finite(a, b)
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    scale = max(float(a_flat.abs().max()), float(b_flat.abs().max()))
    if scale == 0.0:
        return 0.0
    a_flat = a_flat / scale
    b_flat = b_flat / scale
    distance = float(torch.linalg.vector_norm(a_flat - b_flat))
    average_norm = float((torch.linalg.vector_norm(a_flat) + torch.linalg.vector_norm(b_flat)) / 2)
    result = distance / average_norm
    if not math.isfinite(result):
        raise NonFiniteTensorError("L2 computation produced a non-finite value.")
    return result


@register_metric("weight_distribution_divergence")
def weight_distribution_divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Experimental directional softmax-weight divergence, reference to candidate.

    This is a descriptive transformation of flattened weights, not a KL
    divergence between model output distributions. It is excluded from default
    execution and from the composite heuristic.
    """

    _require_same_shape(a, b)
    _require_finite(a, b)
    if a.numel() > MAX_ELEMENTS_FOR_WEIGHT_DIVERGENCE:
        raise WeightDivergenceResourceLimitError(
            f"Tensor has {a.numel():,} elements, above the "
            f"{MAX_ELEMENTS_FOR_WEIGHT_DIVERGENCE:,}-element divergence limit."
        )
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    scale = max(float(a_flat.abs().max()), float(b_flat.abs().max()))
    if scale == 0.0:
        return 0.0
    a_flat = a_flat / scale
    b_flat = b_flat / scale
    temperature = max(
        float(a_flat.std(unbiased=False)),
        float(b_flat.std(unbiased=False)),
        1e-6,
    )
    log_p = torch.log_softmax(a_flat / temperature, dim=0)
    log_q = torch.log_softmax(b_flat / temperature, dim=0)
    p = torch.exp(log_p)
    return max(0.0, float(torch.sum(p * (log_p - log_q))))


def kl_divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Deprecated v0.2 alias for :func:`weight_distribution_divergence`."""

    return float(weight_distribution_divergence(a, b))


@register_metric("spectral_overlap")
def spectral_subspace_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 64) -> float:
    """Overlap of bounded leading left-singular subspaces.

    One-row matrices and vectors do not define an informative subspace
    comparison and are rejected as structurally unavailable.
    """

    _require_same_shape(a, b)
    _require_finite(a, b)
    matrix_a = flatten_to_2d(a)
    matrix_b = flatten_to_2d(b)
    if min(matrix_a.shape) < 2 or min(matrix_b.shape) < 2:
        raise ValueError("Spectral overlap requires matrices with at least two rows and columns.")
    u_a, _, _ = bounded_full_svd(matrix_a, k=k)
    u_b, _, _ = bounded_full_svd(matrix_b, k=k)
    retained_rank = min(u_a.shape[1], u_b.shape[1])
    if retained_rank == 0:
        raise DegenerateMetricInputError(
            "Spectral overlap is undefined when either matrix has numerical rank zero."
        )
    if retained_rank >= min(u_a.shape[0], u_b.shape[0]):
        raise DegenerateMetricInputError(
            "Spectral overlap is uninformative when the retained subspace spans the ambient dimension."
        )
    distance = grassmann_distance(u_a[:, :retained_rank], u_b[:, :retained_rank])
    return float(1.0 - distance)


@register_metric("effective_rank_ratio")
def effective_rank_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """Ratio of bounded effective ranks for exact-shape matrices."""

    _require_same_shape(a, b)
    _require_finite(a, b)
    rank_a = effective_rank(a)
    rank_b = effective_rank(b)
    if rank_a == 0.0 and rank_b == 0.0:
        return 1.0
    if rank_a == 0.0 or rank_b == 0.0:
        return 0.0
    return min(rank_a, rank_b) / max(rank_a, rank_b)


@register_metric("sign_disagreement_rate")
def sign_disagreement_rate(task_vectors: list[torch.Tensor]) -> float:
    """Mean pairwise sign mismatch rate across task vectors.

    Zero versus non-zero deliberately counts as disagreement. At least two
    exact-shape task vectors are required.
    """

    if len(task_vectors) < 2:
        raise ValueError("Sign disagreement requires at least two task vectors.")
    _require_finite(*task_vectors)
    shapes = {tuple(vector.shape) for vector in task_vectors}
    if len(shapes) != 1:
        raise ValueError(f"Task-vector shape mismatch: {sorted(shapes)}")
    signs = [torch.sign(vector.reshape(-1).float()) for vector in task_vectors]
    disagreements: list[float] = []
    for index, first in enumerate(signs):
        for second in signs[index + 1 :]:
            disagreements.append(float((first != second).float().mean()))
    return float(np.mean(disagreements))


@register_metric("tsv_interference")
def tsv_interference_score(task_vectors: list[torch.Tensor], k: int = 64) -> float:
    """Mean right-singular-subspace overlap across task vectors.

    Normalization uses the actual retained rank, including when ``k`` exceeds
    matrix rank.
    """

    if len(task_vectors) < 2:
        raise ValueError("TSV interference requires at least two task vectors.")
    _require_finite(*task_vectors)
    shapes = {tuple(vector.shape) for vector in task_vectors}
    if len(shapes) != 1:
        raise ValueError(f"Task-vector shape mismatch: {sorted(shapes)}")
    right_subspaces = [bounded_full_svd(vector, k=k)[2] for vector in task_vectors]
    retained_rank = min(subspace.shape[0] for subspace in right_subspaces)
    if retained_rank == 0:
        raise DegenerateMetricInputError("No nonzero singular directions were retained.")
    if retained_rank >= min(subspace.shape[1] for subspace in right_subspaces):
        raise DegenerateMetricInputError(
            "TSV overlap is uninformative when the retained subspace spans the ambient dimension."
        )
    right_subspaces = [subspace[:retained_rank] for subspace in right_subspaces]
    interferences: list[float] = []
    for index, first in enumerate(right_subspaces):
        for second in right_subspaces[index + 1 :]:
            overlap = first @ second.T
            interferences.append(
                float(torch.linalg.matrix_norm(overlap, ord="fro")) / math.sqrt(retained_rank)
            )
    return max(0.0, min(1.0, float(np.mean(interferences))))


@register_metric("task_vector_energy")
def centered_task_vector_energy(task_vector: torch.Tensor, k: int = 64) -> float:
    """Fraction of bounded task-vector spectral energy in the leading ``k`` values."""

    _require_finite(task_vector)
    singular_values = bounded_singular_values(task_vector)
    maximum = float(singular_values.max()) if singular_values.numel() else 0.0
    if maximum == 0.0:
        return 0.0
    normalized = singular_values / maximum
    total_energy = float(torch.sum(normalized**2))
    retained_rank = min(k, len(singular_values))
    leading_energy = float(torch.sum(normalized[:retained_rank] ** 2))
    return leading_energy / total_energy


@register_metric("cka_similarity")
def cka_similarity(activations_a: torch.Tensor, activations_b: torch.Tensor) -> float:
    """Standard linear CKA for aligned samples and arbitrary feature widths.

    Computes ``||X.T @ Y||_F^2 / (||X.T @ X||_F ||Y.T @ Y||_F)`` after
    feature-wise centering. The sample counts must match exactly.
    """

    if activations_a.ndim != 2 or activations_b.ndim != 2:
        raise ValueError("CKA inputs must both have shape (samples, features).")
    if activations_a.shape[0] != activations_b.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {activations_a.shape[0]} vs {activations_b.shape[0]} samples"
        )
    if activations_a.shape[0] < 2:
        raise ValueError("CKA requires at least two aligned samples.")
    _require_finite(activations_a, activations_b)
    x_scale = float(activations_a.abs().max())
    y_scale = float(activations_b.abs().max())
    if x_scale == 0.0 or y_scale == 0.0:
        raise DegenerateMetricInputError(
            "CKA is undefined when either activation matrix has no variation."
        )
    x = activations_a.float() / x_scale
    y = activations_b.float() / y_scale
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    numerator = float(torch.linalg.matrix_norm(x.T @ y, ord="fro") ** 2)
    x_norm = float(torch.linalg.matrix_norm(x.T @ x, ord="fro"))
    y_norm = float(torch.linalg.matrix_norm(y.T @ y, ord="fro"))
    denominator = x_norm * y_norm
    if denominator == 0.0:
        raise DegenerateMetricInputError(
            "CKA is undefined when either centered activation matrix has no variation."
        )
    result = numerator / denominator
    if not math.isfinite(result):
        raise NonFiniteTensorError("CKA computation produced a non-finite value.")
    return max(0.0, min(1.0, result))


_HEURISTIC_WEIGHTS: dict[str, float] = {
    "cosine_similarity": 0.40,
    "spectral_overlap": 0.20,
    "effective_rank_ratio": 0.10,
}


def _parameter_weighted_mean(rows: list[TensorMetrics], attribute: str) -> float | None:
    present = [row for row in rows if getattr(row, attribute) is not None]
    if not present:
        return None
    weights = np.asarray([max(row.parameter_count, 1) for row in present], dtype=np.float64)
    values = np.asarray([float(getattr(row, attribute)) for row in present], dtype=np.float64)
    return float(np.average(values, weights=weights))


def compute_heuristic_assessment(
    rows: list[TensorMetrics],
    availability: list[MetricAvailability],
    *,
    scoring_supported: bool,
) -> MergeCompatibilityIndex:
    """Compute a hand-specified, explicitly unvalidated static-risk heuristic."""

    available_names = [item.metric for item in availability if item.status == MetricStatus.COMPUTED]
    unavailable = [item for item in availability if item.status != MetricStatus.COMPUTED]
    if not scoring_supported or not rows:
        return MergeCompatibilityIndex(
            score=None,
            risk_tier="insufficient_evidence",
            evidence_coverage=0.0,
            available_metrics=available_names,
            unavailable_metrics=unavailable,
            notes=[
                "Aggregate scoring was suppressed because structural support was not established."
            ],
        )

    components: dict[str, float] = {}
    cosine = _parameter_weighted_mean(rows, "cosine_similarity")
    if cosine is not None:
        components["cosine_similarity"] = max(0.0, min(1.0, cosine))
    spectral = _parameter_weighted_mean(rows, "spectral_overlap")
    if spectral is not None:
        components["spectral_overlap"] = spectral
    rank_ratio = _parameter_weighted_mean(rows, "effective_rank_ratio")
    if rank_ratio is not None:
        components["effective_rank_ratio"] = rank_ratio
    availability_by_metric = {item.metric: item for item in availability}
    component_weights = {}
    for name, weight in _HEURISTIC_WEIGHTS.items():
        if name not in components:
            continue
        metric_coverage = availability_by_metric.get(name)
        coverage_fraction = (
            metric_coverage.parameter_coverage
            if metric_coverage and metric_coverage.parameter_coverage is not None
            else 0.0
        )
        component_weights[name] = weight * coverage_fraction
    available_weight = sum(component_weights.values())
    total_weight = sum(_HEURISTIC_WEIGHTS.values())
    if available_weight == 0:
        return MergeCompatibilityIndex(
            score=None,
            risk_tier="insufficient_evidence",
            evidence_coverage=0.0,
            available_metrics=available_names,
            unavailable_metrics=unavailable,
            components=components,
        )
    normalized = {name: weight / available_weight for name, weight in component_weights.items()}
    score = 100.0 * sum(components[name] * normalized[name] for name in components)
    evidence_coverage = available_weight / total_weight
    margin = 5.0 + (1.0 - evidence_coverage) * 15.0
    if score >= 75:
        risk_tier = "lower_static_conflict"
    elif score >= 55:
        risk_tier = "mixed_static_signals"
    else:
        risk_tier = "elevated_static_conflict"
    return MergeCompatibilityIndex(
        score=round(max(0.0, min(100.0, score)), 1),
        risk_tier=risk_tier,
        evidence_coverage=round(evidence_coverage, 3),
        available_metrics=available_names,
        unavailable_metrics=unavailable,
        partial_metrics=[
            item
            for item in availability
            if item.status == MetricStatus.COMPUTED and (item.parameter_coverage or 0.0) < 1.0
        ],
        heuristic_band_lower=round(max(0.0, score - margin), 1),
        heuristic_band_upper=round(min(100.0, score + margin), 1),
        components=components,
        component_weights=normalized,
        notes=[
            "Weights and thresholds are hand-specified and have not been prospectively calibrated.",
            "Available component weights were renormalized over the signals computed in this run.",
            "Partial metric availability reduces component weight by compared-parameter coverage.",
            "The heuristic band is a sensitivity display, not a statistical confidence interval.",
        ],
    )


def merge_compatibility_index(
    cosine_sims: list[float],
    spectral_overlaps: list[float] | None = None,
    rank_ratios: list[float] | None = None,
    sign_disagreements: list[float] | None = None,
    tsv_scores: list[float] | None = None,
    energy_scores: list[float] | None = None,
    cka_scores: list[float] | None = None,
) -> MergeCompatibilityIndex:
    """Deprecated list-based adapter for the 2.0.0 heuristic result model."""

    del energy_scores, sign_disagreements, tsv_scores, cka_scores
    components: dict[str, float] = {}
    if cosine_sims:
        components["cosine_similarity"] = max(0.0, min(1.0, float(np.mean(cosine_sims))))
    if spectral_overlaps:
        components["spectral_overlap"] = float(np.mean(spectral_overlaps))
    if rank_ratios:
        components["effective_rank_ratio"] = float(np.mean(rank_ratios))
    weights = {name: _HEURISTIC_WEIGHTS[name] for name in components}
    total = sum(weights.values())
    if not total:
        return MergeCompatibilityIndex()
    normalized = {name: weight / total for name, weight in weights.items()}
    score = 100.0 * sum(components[name] * normalized[name] for name in components)
    tier = (
        "lower_static_conflict"
        if score >= 75
        else "mixed_static_signals"
        if score >= 55
        else "elevated_static_conflict"
    )
    return MergeCompatibilityIndex(
        score=round(score, 1),
        risk_tier=tier,
        evidence_coverage=round(total / sum(_HEURISTIC_WEIGHTS.values()), 3),
        heuristic_band_lower=max(0.0, round(score - 10, 1)),
        heuristic_band_upper=min(100.0, round(score + 10, 1)),
        components=components,
        component_weights=normalized,
    )
