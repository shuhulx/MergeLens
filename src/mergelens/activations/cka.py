"""Centered Kernel Alignment between model activations.

Compares functional similarity that weight-level metrics miss —
two layers can have different weights but similar behavior.
"""

from __future__ import annotations

import torch

from mergelens.compare.metrics import cka_similarity


def compare_activations_cka(
    activations_a: dict[str, torch.Tensor],
    activations_b: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute CKA similarity for each common layer between two models.

    Args:
        activations_a: {layer_name: (n_samples, hidden)} from model A
        activations_b: {layer_name: (n_samples, hidden)} from model B

    Returns:
        {layer_name: cka_score} for common layers
    """
    common_layers = set(activations_a.keys()) & set(activations_b.keys())
    results = {}

    for layer in sorted(common_layers):
        act_a = activations_a[layer]
        act_b = activations_b[layer]

        # Ensure same number of samples
        n = min(act_a.shape[0], act_b.shape[0])
        score = cka_similarity(act_a[:n], act_b[:n])
        results[layer] = round(score, 4)

    return results
