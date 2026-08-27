"""Descriptive interference proxies with explicit shared-base handling."""

from __future__ import annotations

import math

import torch

from mergelens.compare.loader import ModelHandle, find_common_tensors
from mergelens.compare.metrics import cosine_similarity
from mergelens.models import InterferenceScore
from mergelens.utils.tensor_ops import compute_task_vector


def compute_interference(
    source_handles: list[ModelHandle],
    *,
    base_handle: ModelHandle | None = None,
    weights: list[float] | None = None,
) -> list[InterferenceScore]:
    """Compute a static proxy from task vectors or equal-checkpoint deviations.

    With ``base_handle``, task vectors are constructed from that exact handle.
    Without one, the score is only a deviation-from-weighted-average proxy.
    """

    if len(source_handles) < 2:
        return []
    normalized_weights = _normalize_weights(weights, len(source_handles))
    handles = [base_handle, *source_handles] if base_handle is not None else source_handles
    common_names = find_common_tensors(handles)
    scores: list[InterferenceScore] = []

    for name in common_names:
        source_tensors = [handle.get_tensor(name) for handle in source_handles]
        weighted_average = sum(
            tensor.float() * weight for tensor, weight in zip(source_tensors, normalized_weights)
        )
        similarity_profile = {
            handle.path_or_repo: round(cosine_similarity(tensor, weighted_average), 4)
            for handle, tensor in zip(source_handles, source_tensors)
        }

        if base_handle is not None:
            base_tensor = base_handle.get_tensor(name)
            task_vectors = [compute_task_vector(tensor, base_tensor) for tensor in source_tensors]
            interference = _weighted_task_vector_interference(task_vectors, normalized_weights)
        else:
            deviations = [
                1.0 - cosine_similarity(tensor, weighted_average) for tensor in source_tensors
            ]
            interference = (
                sum(deviation * weight for deviation, weight in zip(deviations, normalized_weights))
                / 2.0
            )

        scores.append(
            InterferenceScore(
                tensor_name=name,
                score=round(max(0.0, min(1.0, float(interference))), 4),
                source_similarity_profile=similarity_profile,
            )
        )
    return scores


def _normalize_weights(weights: list[float] | None, count: int) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} source weights, received {len(weights)}.")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Source weights must be finite and non-negative.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Source weights must contain at least one positive value.")
    return [weight / total for weight in weights]


def _weighted_task_vector_interference(
    task_vectors: list[torch.Tensor], weights: list[float]
) -> float:
    weighted_disagreements: list[tuple[float, float]] = []
    for index, first in enumerate(task_vectors):
        for offset, second in enumerate(task_vectors[index + 1 :], start=index + 1):
            pair_weight = weights[index] * weights[offset]
            if pair_weight > 0:
                disagreement = (1.0 - cosine_similarity(first, second)) / 2.0
                weighted_disagreements.append((disagreement, pair_weight))
    total_weight = sum(weight for _, weight in weighted_disagreements)
    if total_weight == 0.0:
        return 0.0
    return sum(value * weight for value, weight in weighted_disagreements) / total_weight
