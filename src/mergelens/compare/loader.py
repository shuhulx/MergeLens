"""Memory-mapped model loading for lazy tensor-by-tensor comparison."""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from math import prod
from pathlib import Path
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from mergelens.models import (
    ComparisonCoverage,
    LayerType,
    ModelInfo,
    ModelRole,
    TensorShapeMismatch,
)
from mergelens.utils.hf_utils import get_model_metadata, resolve_model_path

logger = logging.getLogger(__name__)

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
        self._tensor_shapes: dict[str, tuple[int, ...]] = {}
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
                    self._tensor_shapes[name] = tuple(f.get_slice(name).get_shape())

        self._tensor_names = sorted(self._tensor_to_file, key=tensor_sort_key)

    @property
    def tensor_names(self) -> list[str]:
        """All tensor names in this model."""
        return self._tensor_names

    @property
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Tensor shapes read from safetensors headers without loading weights."""

        return dict(self._tensor_shapes)

    @property
    def exact_parameter_count(self) -> int:
        """Exact number of scalar parameters represented by safetensors files."""

        return sum(prod(shape) for shape in self._tensor_shapes.values())

    @property
    def info(self) -> ModelInfo:
        """Model metadata as ModelInfo."""
        return self.info_with_role(ModelRole.CANDIDATE)

    def info_with_role(self, role: ModelRole) -> ModelInfo:
        """Model metadata annotated with its role in a comparison."""

        config = self._metadata.config or {}
        return ModelInfo(
            name=self._metadata.repo_id,
            path_or_repo=self.path_or_repo,
            role=role,
            num_parameters=self.exact_parameter_count,
            tensor_count=len(self._tensor_names),
            architecture=self._metadata.architecture,
            hidden_size=config.get("hidden_size"),
            num_layers=config.get("num_hidden_layers"),
            vocab_size=config.get("vocab_size"),
            embedding_shape=self._first_shape_for_type(LayerType.EMBEDDING),
            lm_head_shape=self._first_shape_for_type(LayerType.LM_HEAD),
        )

    def _first_shape_for_type(self, tensor_type: LayerType) -> tuple[int, ...] | None:
        for name in self._tensor_names:
            if classify_layer(name) == tensor_type:
                return self._tensor_shapes[name]
        return None

    def get_tensor(self, name: str) -> torch.Tensor:
        """Load a single tensor by name. Memory-mapped — only this tensor is loaded."""
        fpath = self._tensor_to_file.get(name)
        if fpath is None:
            raise KeyError(f"Tensor '{name}' not found in {self.path_or_repo}")
        with safe_open(str(fpath), framework="pt", device=self.device) as f:
            return cast(torch.Tensor, f.get_tensor(name))

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Get shape of a tensor without loading it."""
        if name not in self._tensor_shapes:
            raise KeyError(f"Tensor '{name}' not found in {self.path_or_repo}")
        return self._tensor_shapes[name]


def tensor_sort_key(name: str) -> tuple[tuple[int, ...], str]:
    """Return a deterministic numeric-path key with a full-name tie-breaker."""

    numbers = tuple(int(number) for number in re.findall(r"\d+", name))
    return (numbers if numbers else (1_000_000_000,), name)


def transformer_block_index(name: str) -> int | None:
    """Infer a transformer block index, distinct from tensor list position."""

    match = re.search(r"(?:^|\.)(?:layers?|blocks?|h)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def find_common_tensors(handles: list[ModelHandle]) -> list[str]:
    """Find exact-shape-compatible tensor names common to all handles."""
    if not handles:
        return []
    common = set(handles[0].tensor_names)
    for h in handles[1:]:
        common &= set(h.tensor_names)

    compatible = {
        name for name in common if len({handle.get_tensor_shape(name) for handle in handles}) == 1
    }
    return sorted(compatible, key=tensor_sort_key)


def comparison_coverage(
    reference: ModelHandle,
    candidate: ModelHandle,
    comparison_id: str,
    *,
    explicit_shared_base: bool,
) -> ComparisonCoverage:
    """Compute exact, header-only coverage and structural compatibility."""

    reference_names = set(reference.tensor_names)
    candidate_names = set(candidate.tensor_names)
    common_names = reference_names & candidate_names
    shape_mismatches: list[TensorShapeMismatch] = []
    compatible_names: list[str] = []
    for name in sorted(common_names, key=tensor_sort_key):
        reference_shape = reference.get_tensor_shape(name)
        candidate_shape = candidate.get_tensor_shape(name)
        if reference_shape == candidate_shape:
            compatible_names.append(name)
        else:
            shape_mismatches.append(
                TensorShapeMismatch(
                    tensor_name=name,
                    reference_shape=reference_shape,
                    candidate_shape=candidate_shape,
                )
            )

    reference_info = reference.info_with_role(
        ModelRole.EXPLICIT_SHARED_BASE if explicit_shared_base else ModelRole.IMPLICIT_REFERENCE
    )
    candidate_info = candidate.info_with_role(ModelRole.CANDIDATE)
    common_parameter_count = sum(
        prod(reference.get_tensor_shape(name)) for name in compatible_names
    )
    reference_parameters = reference.exact_parameter_count
    candidate_parameters = candidate.exact_parameter_count

    unsupported_conditions: list[str] = []
    warnings: list[str] = []

    structural_pairs = [
        ("architecture", reference_info.architecture, candidate_info.architecture),
        ("hidden size", reference_info.hidden_size, candidate_info.hidden_size),
        ("layer count", reference_info.num_layers, candidate_info.num_layers),
        ("vocabulary size", reference_info.vocab_size, candidate_info.vocab_size),
    ]
    for label, reference_value, candidate_value in structural_pairs:
        if (
            reference_value is not None
            and candidate_value is not None
            and reference_value != candidate_value
        ):
            unsupported_conditions.append(
                f"Known {label} mismatch: {reference_value!r} vs {candidate_value!r}."
            )

    embedding_compatible = _optional_shape_compatibility(
        reference_info.embedding_shape, candidate_info.embedding_shape
    )
    lm_head_compatible = _optional_shape_compatibility(
        reference_info.lm_head_shape, candidate_info.lm_head_shape
    )
    if embedding_compatible is False:
        unsupported_conditions.append("Embedding tensor shapes are incompatible.")
    if lm_head_compatible is False:
        unsupported_conditions.append("LM-head tensor shapes are incompatible.")
    if shape_mismatches:
        unsupported_conditions.append(
            f"{len(shape_mismatches)} same-name tensor(s) have different exact shapes."
        )
    if not compatible_names:
        unsupported_conditions.append("No exact-shape-compatible tensor names were found.")

    missing_reference = sorted(candidate_names - reference_names, key=tensor_sort_key)
    missing_candidate = sorted(reference_names - candidate_names, key=tensor_sort_key)
    if missing_reference or missing_candidate:
        warnings.append(
            "The checkpoints have different tensor-name sets; coverage reports the compared subset."
        )
        unsupported_conditions.append(
            "Aggregate scoring requires complete tensor-name coverage; partial coverage is reported without a score."
        )
    if reference_info.architecture is None or candidate_info.architecture is None:
        warnings.append(
            "Architecture metadata is incomplete; homology is only an observed indication."
        )

    no_known_structural_conflict = not any(
        condition.startswith("Known ")
        or condition.startswith("Embedding")
        or condition.startswith("LM-head")
        for condition in unsupported_conditions
    )
    appears_homologous = (
        bool(compatible_names) and no_known_structural_conflict and not shape_mismatches
    )
    scoring_supported = appears_homologous and not missing_reference and not missing_candidate

    return ComparisonCoverage(
        comparison_id=comparison_id,
        reference_model=reference.path_or_repo,
        candidate_model=candidate.path_or_repo,
        reference_role=reference_info.role,
        total_tensor_count_reference=len(reference_names),
        total_tensor_count_candidate=len(candidate_names),
        total_parameter_count_reference=reference_parameters,
        total_parameter_count_candidate=candidate_parameters,
        common_tensor_name_count=len(common_names),
        exact_shape_compatible_tensor_count=len(compatible_names),
        common_parameter_count=common_parameter_count,
        parameter_coverage_reference=(
            common_parameter_count / reference_parameters if reference_parameters else None
        ),
        parameter_coverage_candidate=(
            common_parameter_count / candidate_parameters if candidate_parameters else None
        ),
        tensors_missing_from_reference=missing_reference,
        tensors_missing_from_candidate=missing_candidate,
        shape_mismatches=shape_mismatches,
        reference_architecture=reference_info.architecture,
        candidate_architecture=candidate_info.architecture,
        reference_hidden_size=reference_info.hidden_size,
        candidate_hidden_size=candidate_info.hidden_size,
        reference_layer_count=reference_info.num_layers,
        candidate_layer_count=candidate_info.num_layers,
        reference_vocab_size=reference_info.vocab_size,
        candidate_vocab_size=candidate_info.vocab_size,
        embedding_compatible=embedding_compatible,
        lm_head_compatible=lm_head_compatible,
        explicit_shared_base=explicit_shared_base,
        appears_homologous=appears_homologous,
        scoring_supported=scoring_supported,
        warnings=warnings,
        unsupported_conditions=unsupported_conditions,
    )


def _optional_shape_compatibility(
    reference_shape: tuple[int, ...] | None,
    candidate_shape: tuple[int, ...] | None,
) -> bool | None:
    if reference_shape is None or candidate_shape is None:
        return None
    return reference_shape == candidate_shape


def iter_aligned_tensors(
    handles: list[ModelHandle],
    tensor_names: list[str] | None = None,
) -> Generator[tuple[str, LayerType, list[torch.Tensor]], None, None]:
    """Yield aligned tensors from multiple models one layer at a time.

    This iterator does not accumulate tensor groups. Actual peak memory also
    depends on float32 conversions and metric workspaces used by the caller.

    Yields: (tensor_name, layer_type, [tensor_per_model])
    """
    if tensor_names is None:
        tensor_names = find_common_tensors(handles)

    for name in tensor_names:
        layer_type = classify_layer(name)
        shapes = [handle.get_tensor_shape(name) for handle in handles]
        if len(set(shapes)) != 1:
            raise ValueError(f"Exact tensor shape mismatch for {name}: {shapes}")
        tensors = [h.get_tensor(name) for h in handles]
        yield name, layer_type, tensors
