"""Optional activation-based diagnostics (CKA similarity)."""

from mergelens.activations.cka import compare_activations_cka
from mergelens.activations.extractor import ActivationExtractor, extract_activations

__all__ = ["ActivationExtractor", "compare_activations_cka", "extract_activations"]
