"""Descriptive interference proxies with explicit shared-base handling."""

from __future__ import annotations

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
            pairwise = _pairwise_cosines(task_vectors)
            interference = (1.0 - sum(pairwise) / len(pairwise)) / 2.0
        else:
            deviations = [
                1.0 - cosine_similarity(tensor, weighted_average) for tensor in source_tensors
            ]
            interference = sum(deviations) / (2.0 * len(deviations))

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
    total = sum(weights)
    if abs(total) <= 1e-12:
        raise ValueError("Source weights must not sum to zero.")
    return [weight / total for weight in weights]


def _pairwise_cosines(task_vectors: list[torch.Tensor]) -> list[float]:
    values: list[float] = []
    for index, first in enumerate(task_vectors):
        for second in task_vectors[index + 1 :]:
            values.append(cosine_similarity(first, second))
    if not values:
        raise ValueError("At least two task vectors are required.")
    return values
