"""Fail-closed parsing for the MergeKit configuration subset MergeLens analyzes."""

from __future__ import annotations

from typing import Any

import yaml

from mergelens.models import MergeConfig, MergeMethod


def parse_mergekit_config(yaml_content: str) -> MergeConfig:
    """Parse a current MergeKit-style YAML mapping without method coercion."""

    raw = yaml.safe_load(yaml_content)
    if not isinstance(raw, dict):
        raise ValueError("Invalid MergeKit config: expected a YAML mapping.")
    method_value = raw.get("merge_method")
    if not isinstance(method_value, str):
        raise ValueError("MergeKit config must declare a string 'merge_method'.")
    try:
        method = MergeMethod(method_value)
    except ValueError as exc:
        supported = ", ".join(member.value for member in MergeMethod)
        raise ValueError(
            f"Unsupported MergeKit method {method_value!r}; no method-specific analysis was run. "
            f"Known methods: {supported}"
        ) from exc

    input_sections = [name for name in ("models", "slices", "modules") if raw.get(name)]
    if len(input_sections) != 1:
        raise ValueError("Exactly one of 'models', 'slices', or 'modules' must be present.")
    if raw.get("tokenizer") is not None and raw.get("tokenizer_source") is not None:
        raise ValueError("MergeKit config cannot specify both 'tokenizer' and 'tokenizer_source'.")

    models: list[str] = []
    model_parameters: dict[str, dict[str, Any]] = {}
    honored: list[str] = ["merge method identity", "checkpoint references"]
    ignored: list[str] = []
    unsupported: list[str] = []

    if input_sections[0] == "models":
        _collect_models(raw["models"], models, model_parameters)
        honored.append("top-level full-checkpoint model list")
        if all(_has_scalar_weight(model_parameters.get(model, {})) for model in models):
            honored.append("scalar top-level model weights")
        elif any("weight" in model_parameters.get(model, {}) for model in models):
            ignored.append("gradient, filtered, or non-scalar model weights")
    elif input_sections[0] == "slices":
        for slice_definition in raw["slices"]:
            if not isinstance(slice_definition, dict):
                raise ValueError("Each slice must be a YAML mapping.")
            _collect_models(slice_definition.get("sources", []), models, model_parameters)
        honored.append("model references found in slices")
        ignored.extend(
            [
                "slice layer ranges and layer assembly",
                "slice-specific and source-specific parameters",
            ]
        )
    else:
        _collect_module_models(raw["modules"], models, model_parameters)
        unsupported.append("modules configurations are parsed for references but not simulated")

    if not models:
        raise ValueError("MergeKit config contains no usable model references.")

    base_model = raw.get("base_model")
    if base_model is not None and not isinstance(base_model, str):
        raise ValueError("'base_model' must be a string model reference.")
    if base_model:
        honored.append("explicit base-model identity for task-vector construction")

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("Top-level 'parameters' must be a mapping.")
    if parameters:
        ignored.append("merge-method-specific top-level parameters")
    if raw.get("tokenizer") is not None or raw.get("tokenizer_source") is not None:
        ignored.append("tokenizer and embedding remapping semantics")
    if raw.get("chat_template") is not None:
        ignored.append("chat-template selection")
    if raw.get("dtype") is not None or raw.get("out_dtype") is not None:
        ignored.append("merge output dtype")

    return MergeConfig(
        merge_method=method,
        base_model=base_model,
        models=models,
        model_parameters=model_parameters,
        parameters=parameters,
        slices=raw.get("slices"),
        tokenizer=raw.get("tokenizer"),
        tokenizer_source=raw.get("tokenizer_source"),
        chat_template=raw.get("chat_template"),
        dtype=raw.get("out_dtype") or raw.get("dtype"),
        honored_features=list(dict.fromkeys(honored)),
        ignored_features=list(dict.fromkeys(ignored)),
        unsupported_features=list(dict.fromkeys(unsupported)),
        raw_yaml=yaml_content,
    )


def _collect_models(
    definitions: Any,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(definitions, list):
        raise ValueError("Model/source definitions must be a list.")
    for definition in definitions:
        if isinstance(definition, str):
            model = definition
            parameters: dict[str, Any] = {}
        elif isinstance(definition, dict):
            candidate_model = definition.get("model")
            parameters = definition.get("parameters") or {}
            if not isinstance(candidate_model, str) or not candidate_model:
                raise ValueError("Every model/source mapping must contain a non-empty 'model'.")
            model = candidate_model
            if not isinstance(parameters, dict):
                raise ValueError("Per-model 'parameters' must be a mapping.")
        else:
            raise ValueError("Each model/source must be a string or mapping.")
        if model not in models:
            models.append(model)
        model_parameters[model] = parameters


def _collect_module_models(
    modules: Any,
    models: list[str],
    model_parameters: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(modules, dict):
        raise ValueError("'modules' must be a mapping.")
    for module in modules.values():
        if not isinstance(module, dict):
            raise ValueError("Each module definition must be a mapping.")
        if module.get("models"):
            _collect_models(module["models"], models, model_parameters)
        if module.get("slices"):
            for slice_definition in module["slices"]:
                _collect_models(slice_definition.get("sources", []), models, model_parameters)


def _has_scalar_weight(parameters: dict[str, Any]) -> bool:
    return isinstance(parameters.get("weight", 1.0), (int, float))
