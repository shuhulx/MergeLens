"""Centered Kernel Alignment between model activations.

Compares functional similarity that weight-level metrics miss —
two layers can have different weights but similar behavior.
"""

from __future__ import annotations

import logging

import torch

from mergelens.compare.metrics import cka_similarity

logger = logging.getLogger(__name__)


def compare_activations_cka(
    activations_a: dict[str, torch.Tensor],
    activations_b: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute CKA similarity for each common layer between two models.

    Args:
        activations_a: {layer_name: (n_samples, hidden)} from model A
        activations_b: {layer_name: (n_samples, hidden)} from model B

    Returns:
        {layer_name: cka_score} for common layers. Layers with mismatched
        hidden dimensions are skipped with a warning (different architectures
        produce incompatible activation spaces for direct CKA comparison).
    """
    common_layers = set(activations_a.keys()) & set(activations_b.keys())
    results = {}

    for layer in sorted(common_layers):
        act_a = activations_a[layer]
        act_b = activations_b[layer]

        # Ensure same number of samples
        n = min(act_a.shape[0], act_b.shape[0])
        act_a = act_a[:n]
        act_b = act_b[:n]

        # Skip layers where hidden dimensions differ — CKA requires both
        # activation matrices to span the same sample space, but the hidden
        # dim can differ only when the underlying kernel matrices are formed
        # via X @ X^T (shape n x n).  However mismatched hidden dims indicate
        # architecturally incompatible layers, so we skip rather than silently
        # produce a meaningless score.
        if act_a.shape[1] != act_b.shape[1]:
            logger.warning(
                "Skipping CKA for layer %s: hidden dim mismatch (%d vs %d). "
                "Models have different layer widths at this position.",
                layer,
                act_a.shape[1],
                act_b.shape[1],
            )
            continue

        score = cka_similarity(act_a, act_b)
        results[layer] = round(score, 4)

    return results
