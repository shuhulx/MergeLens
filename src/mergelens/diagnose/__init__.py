"""MergeKit configuration diagnosis with explicit simulation-scope disclosure."""

from __future__ import annotations

import logging
from pathlib import Path

from mergelens.compare.loader import ModelHandle
from mergelens.diagnose.attribution import (
    compute_attribution,
    compute_source_similarity_profile,
)
from mergelens.diagnose.config_parser import parse_mergekit_config
from mergelens.diagnose.interference import compute_interference
from mergelens.models import DiagnoseResult

logger = logging.getLogger(__name__)


def diagnose_config(config_path: str, device: str = "cpu") -> DiagnoseResult:
    """Analyze the supported static subset of a MergeKit configuration."""

    config = parse_mergekit_config(Path(config_path).read_text())
    if config.unsupported_features:
        return DiagnoseResult(
            config=config,
            interference_scores=[],
            overall_interference=0.0,
            analysis_status="unsupported_configuration",
            honored_features=config.honored_features,
            ignored_features=config.ignored_features,
            unsupported_features=config.unsupported_features,
            recommendations=[
                "No interference proxy was computed because the configuration uses unsupported semantics."
            ],
        )

    source_paths = [path for path in config.models if path != config.base_model]
    source_handles: list[ModelHandle] = []
    load_failures: list[str] = []
    for path in source_paths:
        try:
            source_handles.append(ModelHandle(path, device=device))
        except (FileNotFoundError, OSError, ValueError) as exc:
            load_failures.append(f"{path}: {exc}")
            logger.warning("Failed to load model %s: %s", path, exc)

    base_handle: ModelHandle | None = None
    if config.base_model:
        try:
            base_handle = ModelHandle(config.base_model, device=device)
        except (FileNotFoundError, OSError, ValueError) as exc:
            load_failures.append(f"base model {config.base_model}: {exc}")

    if len(source_handles) < 2 or (config.base_model and base_handle is None):
        return DiagnoseResult(
            config=config,
            interference_scores=[],
            overall_interference=0.0,
            analysis_status="insufficient_loadable_checkpoints",
            honored_features=config.honored_features,
            ignored_features=config.ignored_features,
            unsupported_features=config.unsupported_features,
            recommendations=[
                "No interference proxy was computed. Load failures: " + "; ".join(load_failures)
            ],
        )

    weights = _scalar_model_weights(config)
    scores = compute_interference(
        source_handles,
        base_handle=base_handle,
        weights=weights,
    )
    overall = sum(score.score for score in scores) / len(scores) if scores else 0.0
    recommendations = _inspection_notes(overall, scores, base_handle is not None)
    if load_failures:
        recommendations.append(
            "Some checkpoint references were not loaded: " + "; ".join(load_failures)
        )
    return DiagnoseResult(
        config=config,
        interference_scores=scores,
        source_similarity_profiles={
            score.tensor_name: score.source_similarity_profile for score in scores
        },
        overall_interference=round(overall, 4),
        analysis_status=(
            "task_vector_proxy_with_explicit_base"
            if base_handle is not None
            else "weighted_average_deviation_proxy_without_shared_base"
        ),
        honored_features=config.honored_features,
        ignored_features=config.ignored_features,
        unsupported_features=config.unsupported_features,
        recommendations=recommendations,
    )


def _scalar_model_weights(config) -> list[float] | None:
    if "scalar non-negative top-level model weights" not in config.honored_features:
        return None
    weights: list[float] = []
    for model in config.models:
        if model == config.base_model:
            continue
        value = config.model_parameters.get(model, {}).get("weight", 1.0)
        if not isinstance(value, (int, float)):
            return None
        weights.append(float(value))
    return weights


def _inspection_notes(overall: float, scores, has_base: bool) -> list[str]:
    notes = [
        "The score is a descriptive static proxy and does not simulate MergeKit or predict merged behavior."
    ]
    if overall > 0.5:
        notes.append(
            "The proxy is elevated; prioritize post-merge evaluation and inspect the highest-scoring tensors."
        )
    hotspots = [score.tensor_name for score in scores if score.score > 0.7]
    if hotspots:
        notes.append("Heuristic inspection priorities: " + ", ".join(hotspots[:5]))
    if not has_base:
        notes.append(
            "No explicit shared base was supplied, so task-vector semantics were not inferred."
        )
    return notes


__all__ = [
    "compute_attribution",
    "compute_interference",
    "compute_source_similarity_profile",
    "diagnose_config",
    "parse_mergekit_config",
]
