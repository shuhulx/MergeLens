"""Reusable model-resolution and bounded tensor utilities."""

from mergelens.utils.hf_utils import get_model_metadata, resolve_model_path
from mergelens.utils.tensor_ops import (
    SVDResourceLimitError,
    bounded_full_svd,
    bounded_singular_values,
    effective_rank,
    flatten_to_2d,
)

__all__ = [
    "SVDResourceLimitError",
    "bounded_full_svd",
    "bounded_singular_values",
    "effective_rank",
    "flatten_to_2d",
    "get_model_metadata",
    "resolve_model_path",
]
