"""Non-causal source-similarity profiles for an existing merged checkpoint."""

from __future__ import annotations

from mergelens.compare.loader import ModelHandle, find_common_tensors
from mergelens.compare.metrics import cosine_similarity


def compute_source_similarity_profile(
    merged_handle: ModelHandle,
    source_handles: list[ModelHandle],
) -> dict[str, dict[str, float]]:
    """Return raw cosine similarities to each source for every comparable tensor.

    These values do not estimate causal contribution and are not normalized to
    imply that a source "won" a tensor.
    """

    common_names = find_common_tensors([merged_handle, *source_handles])
    profiles: dict[str, dict[str, float]] = {}
    for name in common_names:
        merged_tensor = merged_handle.get_tensor(name)
        profiles[name] = {
            source.path_or_repo: round(cosine_similarity(merged_tensor, source.get_tensor(name)), 4)
            for source in source_handles
        }
    return profiles


def compute_attribution(
    merged_handle: ModelHandle,
    source_handles: list[ModelHandle],
) -> dict[str, dict[str, float]]:
    """Deprecated alias for :func:`compute_source_similarity_profile`."""

    return compute_source_similarity_profile(merged_handle, source_handles)
