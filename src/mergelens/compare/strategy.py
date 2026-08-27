"""Rule-based MergeKit starting configurations with explicit limitations."""

from __future__ import annotations

from importlib import metadata
from typing import Any, cast

import yaml

from mergelens.models import CompareResult, MergeMethod, StrategyRecommendation


def recommend_strategy(result: CompareResult) -> StrategyRecommendation:
    """Choose a low-authority starting point from transparent static rules."""

    rows = result.tensor_metrics
    cosine_values = [row.cosine_similarity for row in rows if row.cosine_similarity is not None]
    sign_values = [
        row.sign_disagreement_rate for row in rows if row.sign_disagreement_rate is not None
    ]
    energy_values = [row.task_vector_energy for row in rows if row.task_vector_energy is not None]
    spectral_values = [row.spectral_overlap for row in rows if row.spectral_overlap is not None]
    average_cosine = sum(cosine_values) / len(cosine_values) if cosine_values else 0.0
    average_sign = sum(sign_values) / len(sign_values) if sign_values else None
    average_energy = sum(energy_values) / len(energy_values) if energy_values else None
    average_spectral = sum(spectral_values) / len(spectral_values) if spectral_values else None
    has_explicit_base = result.explicit_base is not None
    signals = [f"parameter-weighted aggregate score: {result.mci.score}"]
    if cosine_values:
        signals.append(f"mean tensor cosine similarity: {average_cosine:.4f}")
    if average_sign is not None:
        signals.append(f"mean sign disagreement: {average_sign:.4f}")
    if average_energy is not None:
        signals.append(f"mean leading task-vector energy: {average_energy:.4f}")
    if average_spectral is not None:
        signals.append(f"mean left-subspace overlap: {average_spectral:.4f}")

    warnings = [
        "This is a hand-specified starting point, not an optimal-method or quality prediction.",
        "Evaluate merged-model behavior and capability retention after every candidate merge.",
    ]
    if result.tensor_conflict_regions:
        warnings.append(
            f"{len(result.tensor_conflict_regions)} ordered tensor region(s) were flagged for inspection; "
            "no per-tensor MergeKit overrides were inferred from cosine alone."
        )

    if has_explicit_base and average_sign is not None and average_sign > 0.30:
        method = MergeMethod.TIES
        strength = 0.55
        reasoning = (
            "The explicit shared base makes task vectors interpretable, and the observed sign "
            "disagreement exceeds the hand-specified 0.30 inspection threshold. TIES is offered "
            "as a testable starting hypothesis because it includes sign consensus."
        )
        config = _generate_yaml(method, result, density=0.5)
    elif has_explicit_base and average_energy is not None and average_energy > 0.80:
        method = MergeMethod.DARE_TIES
        strength = 0.50
        reasoning = (
            "The explicit shared base supports task-vector analysis and leading singular values "
            "contain more than 80% of measured task-vector energy. DARE-TIES is an illustrative "
            "prune-and-consensus experiment, not an evidence-backed optimum."
        )
        config = _generate_yaml(method, result, density=0.5)
    elif not has_explicit_base and len(result.models) == 2:
        method = MergeMethod.SLERP
        strength = 0.40
        reasoning = (
            "Only pairwise static checkpoint signals are available. SLERP is provided as one "
            "simple two-endpoint baseline to compare against linear interpolation."
        )
        config = _generate_yaml(method, result, t=0.5)
    elif has_explicit_base:
        method = MergeMethod.TASK_ARITHMETIC
        strength = 0.40
        reasoning = (
            "An explicit shared base was supplied, so an equal-weight task-arithmetic configuration "
            "preserves that role directly. The weights are placeholders for post-merge evaluation."
        )
        config = _generate_yaml(method, result)
    else:
        method = MergeMethod.LINEAR
        strength = 0.30
        reasoning = (
            "No explicit shared base was supplied, so task-vector methods are not proposed. An "
            "equal-weight linear merge is emitted only as a transparent baseline."
        )
        config = _generate_yaml(method, result)

    valid, target = validate_mergekit_yaml(config)
    return StrategyRecommendation(
        method=method,
        heuristic_strength=strength,
        reasoning=reasoning,
        triggering_signals=signals,
        mergekit_yaml=config,
        config_status="schema_validated" if valid else "illustrative",
        validated_against=target if valid else None,
        warnings=warnings
        + (
            []
            if valid
            else ["MergeKit is not installed in this runtime; schema validation was not run."]
        ),
    )


def _generate_yaml(method: MergeMethod, result: CompareResult, **parameters: Any) -> str:
    """Generate a conservative MergeKit configuration using current field placement."""

    candidates = [model for model in result.models if model.role.value == "candidate"]
    if not candidates:
        candidates = result.models
    merge_inputs = candidates if result.explicit_base is not None else result.models

    if method == MergeMethod.SLERP:
        if result.explicit_base is not None:
            endpoints = [result.explicit_base, candidates[0]]
        else:
            endpoints = result.models[:2]
        if len(endpoints) != 2:
            raise ValueError("SLERP generation requires exactly two endpoints.")
        config: dict[str, Any] = {
            "models": [{"model": endpoint.path_or_repo} for endpoint in endpoints],
            "merge_method": "slerp",
            "base_model": endpoints[0].path_or_repo,
            "parameters": {"t": float(parameters.get("t", 0.5))},
            "dtype": "bfloat16",
        }
    elif method in {
        MergeMethod.TASK_ARITHMETIC,
        MergeMethod.TIES,
        MergeMethod.DARE_TIES,
        MergeMethod.DARE_LINEAR,
        MergeMethod.DELLA,
        MergeMethod.DELLA_LINEAR,
        MergeMethod.BREADCRUMBS,
        MergeMethod.BREADCRUMBS_TIES,
    }:
        if result.explicit_base is None:
            raise ValueError(f"{method.value} generation requires an explicit shared base.")
        per_model_weight = 1.0 / len(candidates)
        models: list[dict[str, Any]] = []
        for model in candidates:
            model_parameters: dict[str, float] = {"weight": per_model_weight}
            if method not in {MergeMethod.TASK_ARITHMETIC}:
                model_parameters["density"] = float(parameters.get("density", 0.5))
            models.append({"model": model.path_or_repo, "parameters": model_parameters})
        config = {
            "models": models,
            "merge_method": method.value,
            "base_model": result.explicit_base.path_or_repo,
            "parameters": {"normalize": True},
            "dtype": "bfloat16",
        }
    elif method == MergeMethod.LINEAR:
        if len(merge_inputs) < 2:
            raise ValueError("Linear generation requires at least two candidate models.")
        alpha = parameters.get("alpha")
        if alpha is not None and len(merge_inputs) == 2:
            weights = [1.0 - float(alpha), float(alpha)]
        else:
            weights = [1.0 / len(merge_inputs)] * len(merge_inputs)
        config = {
            "models": [
                {"model": model.path_or_repo, "parameters": {"weight": weight}}
                for model, weight in zip(merge_inputs, weights)
            ],
            "merge_method": "linear",
            "dtype": "bfloat16",
        }
    else:
        raise ValueError(f"Configuration generation is not implemented for {method.value}.")

    return cast(str, yaml.safe_dump(config, sort_keys=False)).rstrip() + "\n"


def validate_mergekit_yaml(yaml_content: str) -> tuple[bool, str | None]:
    """Validate with the installed MergeKit parser when it is available."""

    try:
        from mergekit.config import MergeConfiguration  # type: ignore[import-not-found]
    except ImportError:
        return False, None
    raw = yaml.safe_load(yaml_content)
    MergeConfiguration.model_validate(raw)
    try:
        version = metadata.version("mergekit")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return True, f"mergekit {version} MergeConfiguration"
