"""Optional activation-based diagnostics (CKA similarity)."""

from mergelens.activations.cka import CKAComparison, compare_activations_cka
from mergelens.activations.extractor import ActivationExtractor, ActivationSet, extract_activations

__all__ = [
    "ActivationExtractor",
    "ActivationSet",
    "CKAComparison",
    "compare_activations_cka",
    "extract_activations",
]
