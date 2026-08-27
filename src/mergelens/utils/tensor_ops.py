"""Tensor operations for metric computation."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch


def flatten_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to 2D matrix for SVD and other operations.

    For 1D tensors, reshapes to (1, N).
    For 3D+, reshapes to (first_dim, product_of_rest).
    """
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


MAX_ELEMENTS_FOR_SVD: int = 2_000_000
MAX_SVD_DIMENSION: int = 2_048


class SVDResourceLimitError(ValueError):
    """Raised when a full SVD would exceed the documented resource policy."""


def _prepare_bounded_svd_input(matrix: torch.Tensor) -> torch.Tensor:
    matrix = flatten_to_2d(matrix).float()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("SVD input contains NaN or infinite values.")
    if matrix.numel() > MAX_ELEMENTS_FOR_SVD:
        raise SVDResourceLimitError(
            f"Tensor has {matrix.numel():,} elements, above the "
            f"{MAX_ELEMENTS_FOR_SVD:,}-element full-SVD limit."
        )
    if min(matrix.shape) > MAX_SVD_DIMENSION:
        raise SVDResourceLimitError(
            f"Tensor minimum dimension is {min(matrix.shape):,}, above the "
            f"{MAX_SVD_DIMENSION:,}-dimension full-SVD limit."
        )
    return matrix


def bounded_full_svd(
    matrix: torch.Tensor, k: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a bounded full SVD and return only the leading ``k`` outputs.

    The decomposition itself is full, not truncated. The conservative input
    policy prevents large tensors from silently entering a cubic-time path.
    """

    matrix = _prepare_bounded_svd_input(matrix)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0]) == 0.0:
        retained_rank = 0
    else:
        tolerance = (
            torch.finfo(singular_values.dtype).eps * max(matrix.shape) * float(singular_values[0])
        )
        numerical_rank = int((singular_values > tolerance).sum())
        retained_rank = min(k, numerical_rank)
    return u[:, :retained_rank], singular_values[:retained_rank], vh[:retained_rank, :]


def bounded_singular_values(matrix: torch.Tensor) -> torch.Tensor:
    """Return all singular values under the same bounded resource policy."""

    return cast(torch.Tensor, torch.linalg.svdvals(_prepare_bounded_svd_input(matrix)))


def truncated_svd(
    matrix: torch.Tensor, k: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated v0.2 name for :func:`bounded_full_svd`.

    This function performs a bounded full decomposition and slices the returned
    factors; it is not an algorithmically truncated SVD.
    """

    return bounded_full_svd(matrix, k=k)


def effective_rank(matrix: torch.Tensor) -> float:
    """Compute effective rank via Shannon entropy of normalized singular values.

    erank = exp(-sum(p_i * log(p_i))) where p_i = s_i / sum(s_j)

    Higher effective rank = more distributed information.
    Returns 0.0 for a zero-rank matrix and at least 1.0 otherwise.
    """
    singular_values = bounded_singular_values(matrix)
    if len(singular_values) == 0 or float(singular_values[0]) == 0.0:
        return 0.0
    matrix_2d = flatten_to_2d(matrix)
    tolerance = (
        torch.finfo(singular_values.dtype).eps * max(matrix_2d.shape) * float(singular_values[0])
    )
    singular_values = singular_values[singular_values > tolerance]
    p = singular_values / singular_values.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return float(np.exp(entropy))


def grassmann_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Compute Grassmann distance between two subspaces.

    Uses principal angles: distance = sqrt(sum(theta_i^2))
    Returns value in [0, pi/2 * sqrt(k)], normalized to [0, 1].
    """
    if U1.shape[1] == 0 or U2.shape[1] == 0:
        return 1.0
    # Compute cosines of principal angles
    q1, _ = torch.linalg.qr(U1, mode="reduced")
    q2, _ = torch.linalg.qr(U2, mode="reduced")
    M = q1.T @ q2
    sigmas = torch.linalg.svdvals(M)
    sigmas = torch.clamp(sigmas, -1.0, 1.0)
    tolerance = torch.finfo(sigmas.dtype).eps * max(U1.shape[0], U1.shape[1], U2.shape[1]) * 10
    sigmas = torch.where(torch.abs(1.0 - sigmas) <= tolerance, torch.ones_like(sigmas), sigmas)
    # Principal angles
    angles = torch.acos(sigmas)
    distance = torch.sqrt((angles**2).sum()).item()
    # Normalize by max possible distance
    max_distance = (np.pi / 2) * np.sqrt(min(U1.shape[1], U2.shape[1]))
    if max_distance == 0:
        return 0.0
    return float(min(distance / max_distance, 1.0))


def compute_task_vector(model_weights: torch.Tensor, base_weights: torch.Tensor) -> torch.Tensor:
    """Compute task vector: model_weights - base_weights."""

    if model_weights.shape != base_weights.shape:
        raise ValueError(
            f"Exact tensor shape mismatch: {tuple(model_weights.shape)} vs "
            f"{tuple(base_weights.shape)}"
        )
    return model_weights.float() - base_weights.float()
