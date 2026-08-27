"""Hook-based activation extraction from transformer models.

Uses PyTorch forward hooks on a small calibration dataset to extract
intermediate representations for CKA comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class ActivationSet:
    """Aligned activations and the identity of their calibration inputs."""

    activations: dict[str, torch.Tensor]
    calibration_id: str
    sample_count: int
    pooling_rule: str = "attention_mask_weighted_mean"


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
        self._hooks: list[RemovableHandle] = []
        self._attention_mask: torch.Tensor | None = None

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
            if output.ndim == 3:  # (batch, seq, hidden)
                if self._attention_mask is None:
                    raise ValueError(
                        "An attention mask is required for sequence activation pooling."
                    )
                mask = self._attention_mask.to(device=output.device, dtype=output.dtype)
                if tuple(mask.shape) != tuple(output.shape[:2]):
                    raise ValueError(
                        f"Attention-mask shape {tuple(mask.shape)} does not align with "
                        f"activation shape {tuple(output.shape)}."
                    )
                denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = (output * mask.unsqueeze(-1)).sum(dim=1) / denominator
                self._activations[name].append(pooled.detach().cpu())
            elif output.ndim == 2:  # (batch, hidden)
                self._activations[name].append(output.detach().cpu())

        return hook_fn

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        """Set the mask used by hooks for the next forward pass."""

        self._attention_mask = attention_mask

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
        self._attention_mask = None


def extract_activations(
    model: nn.Module,
    tokenizer,
    calibration_texts: list[str],
    layer_names: list[str],
    max_length: int = 512,
    batch_size: int = 8,
    device: str = "cpu",
) -> ActivationSet:
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
        ActivationSet containing tensors and a stable calibration-text identity.
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
            extractor.set_attention_mask(inputs.get("attention_mask"))
            model(**inputs)

    encoded_identity = json.dumps(
        {"texts": calibration_texts, "max_length": max_length},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    calibration_id = hashlib.sha256(encoded_identity).hexdigest()
    return ActivationSet(
        activations=extractor.get_activations(),
        calibration_id=calibration_id,
        sample_count=len(calibration_texts),
    )
