"""Hook-based activation extraction from transformer models.

Uses PyTorch forward hooks on a small calibration dataset to extract
intermediate representations for CKA comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActivationExtractor:
    """Extract activations from specified layers using forward hooks.

    Usage:
        extractor = ActivationExtractor(model, layer_names=["model.layers.0", "model.layers.1"])
        with extractor:
            output = model(input_ids)
        activations = extractor.get_activations()
    """

    def __init__(self, model: nn.Module, layer_names: list[str] | None = None):
        self.model = model
        self.layer_names = layer_names or []
        self._activations: dict[str, list[torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                self._activations[name] = []
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Store mean-pooled activations to save memory
            if output.ndim == 3:  # (batch, seq, hidden)
                self._activations[name].append(output.mean(dim=1).detach().cpu())
            elif output.ndim == 2:  # (batch, hidden)
                self._activations[name].append(output.detach().cpu())

        return hook_fn

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_activations(self) -> dict[str, torch.Tensor]:
        """Get concatenated activations for each layer.

        Returns: {layer_name: tensor of shape (n_samples, hidden_dim)}
        """
        result = {}
        for name, acts in self._activations.items():
            if acts:
                result[name] = torch.cat(acts, dim=0)
        return result

    def clear(self):
        """Clear stored activations."""
        self._activations = {name: [] for name in self.layer_names}


def extract_activations(
    model: nn.Module,
    tokenizer,
    calibration_texts: list[str],
    layer_names: list[str],
    max_length: int = 512,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract activations from a model using calibration texts.

    Args:
        model: The transformer model.
        tokenizer: The tokenizer for the model.
        calibration_texts: List of text samples for calibration.
        layer_names: Which layers to extract from.
        max_length: Max sequence length.
        batch_size: Batch size for inference.
        device: Torch device.

    Returns:
        Dict mapping layer names to activation tensors (n_samples, hidden_dim).
    """
    model = model.to(device).eval()
    extractor = ActivationExtractor(model, layer_names=layer_names)

    with torch.no_grad(), extractor:
        for i in range(0, len(calibration_texts), batch_size):
            batch_texts = calibration_texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            model(**inputs)

    return extractor.get_activations()
