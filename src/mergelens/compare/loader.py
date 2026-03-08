"""Memory-mapped model loading for efficient layer-by-layer comparison."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)
from collections.abc import Generator
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from mergelens.models import LayerType, ModelInfo
from mergelens.utils.hf_utils import get_model_metadata, resolve_model_path

# Pattern to classify layer names into types
_LAYER_TYPE_PATTERNS = {
    LayerType.ATTENTION_Q: re.compile(r"(self_attn|attention).*q_proj"),
    LayerType.ATTENTION_K: re.compile(r"(self_attn|attention).*k_proj"),
    LayerType.ATTENTION_V: re.compile(r"(self_attn|attention).*v_proj"),
    LayerType.ATTENTION_O: re.compile(r"(self_attn|attention).*o_proj"),
    LayerType.MLP_GATE: re.compile(r"mlp.*gate"),
    LayerType.MLP_UP: re.compile(r"mlp.*up"),
    LayerType.MLP_DOWN: re.compile(r"mlp.*down"),
    LayerType.NORM: re.compile(r"(layer_?norm|rms_?norm|norm)"),
    LayerType.EMBEDDING: re.compile(r"embed"),
    LayerType.LM_HEAD: re.compile(r"lm_head"),
}


def classify_layer(name: str) -> LayerType:
    """Classify a layer name into a LayerType."""
    name_lower = name.lower()
    for layer_type, pattern in _LAYER_TYPE_PATTERNS.items():
        if pattern.search(name_lower):
            return layer_type
    return LayerType.OTHER


class ModelHandle:
    """Handle for lazy access to a model's safetensors weights.

    Supports both local directories and HuggingFace Hub repos.
    Uses memory-mapped access — tensors are loaded on demand.
    """

    def __init__(self, path_or_repo: str, device: str = "cpu"):
        self.path_or_repo = path_or_repo
        self.device = device
        self._resolved_path, self._is_local = resolve_model_path(path_or_repo)
        self._metadata = get_model_metadata(path_or_repo)
        self._files: list[Path] = []
        self._tensor_to_file: dict[str, Path] = {}
        self._tensor_names: list[str] = []
        self._resolve_files()

    def _resolve_files(self) -> None:
        """Resolve safetensors file paths."""
        if self._is_local:
            local_dir = Path(self._resolved_path)
            self._files = sorted(local_dir.glob("*.safetensors"))
        else:
            # Download safetensors files from Hub
            self._files = []
            for fname in self._metadata.safetensors_files or ["model.safetensors"]:
                try:
                    local = hf_hub_download(self._resolved_path, fname)
                    self._files.append(Path(local))
                except Exception as exc:
                    logger.debug(
                        "Failed to download %s from %s: %s", fname, self._resolved_path, exc
                    )
                    continue

        if not self._files:
            raise FileNotFoundError(f"No safetensors files found for {self.path_or_repo}")

        # Map tensor names to files
        for fpath in self._files:
            with safe_open(str(fpath), framework="pt", device=self.device) as f:
                for name in f.keys():
                    self._tensor_to_file[name] = fpath
                    self._tensor_names.append(name)

    @property
    def tensor_names(self) -> list[str]:
        """All tensor names in this model."""
        return self._tensor_names

    @property
    def info(self) -> ModelInfo:
        """Model metadata as ModelInfo."""
        return ModelInfo(
            name=self._metadata.repo_id,
            path_or_repo=self.path_or_repo,
            num_parameters=self._metadata.num_parameters,
            architecture=self._metadata.architecture,
            num_layers=self._metadata.config.get("num_hidden_layers"),
        )

    def get_tensor(self, name: str) -> torch.Tensor:
        """Load a single tensor by name. Memory-mapped — only this tensor is loaded."""
        fpath = self._tensor_to_file.get(name)
        if fpath is None:
            raise KeyError(f"Tensor '{name}' not found in {self.path_or_repo}")
        with safe_open(str(fpath), framework="pt", device=self.device) as f:
            return f.get_tensor(name)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Get shape of a tensor without loading it."""
        fpath = self._tensor_to_file.get(name)
        if fpath is None:
            raise KeyError(f"Tensor '{name}' not found in {self.path_or_repo}")
        with safe_open(str(fpath), framework="pt", device=self.device) as f:
            return tuple(f.get_slice(name).get_shape())


def find_common_tensors(handles: list[ModelHandle]) -> list[str]:
    """Find tensor names common to all model handles."""
    if not handles:
        return []
    common = set(handles[0].tensor_names)
    for h in handles[1:]:
        common &= set(h.tensor_names)

    # Sort by layer number for consistent ordering
    def _sort_key(name: str) -> tuple:
        numbers = re.findall(r"\d+", name)
        return tuple(int(n) for n in numbers) if numbers else (999999,)

    return sorted(common, key=_sort_key)


def iter_aligned_tensors(
    handles: list[ModelHandle],
    tensor_names: list[str] | None = None,
) -> Generator[tuple[str, LayerType, list[torch.Tensor]], None, None]:
    """Yield aligned tensors from multiple models one layer at a time.

    Peak memory: only tensors for one layer across all models.

    Yields: (tensor_name, layer_type, [tensor_per_model])
    """
    if tensor_names is None:
        tensor_names = find_common_tensors(handles)

    for name in tensor_names:
        layer_type = classify_layer(name)
        tensors = [h.get_tensor(name) for h in handles]
        yield name, layer_type, tensors
