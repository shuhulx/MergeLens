"""Tests for transparent strategy rules and current MergeKit schema placement."""

import pytest
import yaml

from mergelens.compare.strategy import _generate_yaml, recommend_strategy, validate_mergekit_yaml
from mergelens.models import (
    CandidateSetTensorMetrics,
    CompareResult,
    MergeCompatibilityIndex,
    MergeMethod,
    ModelInfo,
    ModelRole,
    TensorMetrics,
)


def _result(*, explicit=False, candidate_count=2, sign=None, energy=None):
    candidates = [
        ModelInfo(name=f"candidate_{index}", path_or_repo=f"org/candidate-{index}")
        for index in range(candidate_count)
    ]
    base = (
        ModelInfo(
            name="base",
            path_or_repo="org/actual-base",
            role=ModelRole.EXPLICIT_SHARED_BASE,
        )
        if explicit
        else None
    )
    reference = base or candidates[0].model_copy(update={"role": ModelRole.IMPLICIT_REFERENCE})
    rows = [
        TensorMetrics(
            reference_model=reference.path_or_repo,
            candidate_model=model.path_or_repo,
            comparison_id=f"comparison_{index}",
            tensor_name="model.layers.0.self_attn.q_proj.weight",
            parameter_count=16,
            cosine_similarity=0.8,
            l2_distance=0.2,
            sign_disagreement_rate=sign,
            task_vector_energy=energy,
        )
        for index, model in enumerate(candidates if explicit else candidates[1:])
    ]
    return CompareResult(
        models=candidates,
        reference_model=reference,
        explicit_base=base,
        tensor_metrics=rows,
        candidate_set_metrics=(
            [
                CandidateSetTensorMetrics(
                    candidate_set_id="candidate_set_0",
                    base_model=base.path_or_repo,
                    candidate_models=[model.path_or_repo for model in candidates],
                    tensor_name="model.layers.0.self_attn.q_proj.weight",
                    parameter_count=16,
                    tensor_position=0,
                    sign_disagreement_rate=sign,
                )
            ]
            if base is not None and sign is not None
            else []
        ),
        mci=MergeCompatibilityIndex(
            score=70,
            risk_tier="mixed_static_signals",
            evidence_coverage=0.7,
        ),
    )


def test_pair_without_explicit_base_emits_slerp_baseline():
    result = _result(explicit=False, candidate_count=2)
    recommendation = recommend_strategy(result)
    config = yaml.safe_load(recommendation.mergekit_yaml)
    assert recommendation.method == MergeMethod.SLERP
    assert config["base_model"] == "org/candidate-0"
    assert [item["model"] for item in config["models"]] == [
        "org/candidate-0",
        "org/candidate-1",
    ]
    assert "optimal" not in recommendation.reasoning.lower()


def test_high_sign_signal_does_not_claim_a_method_optimum():
    result = _result(explicit=True, candidate_count=2, sign=0.5, energy=0.2)
    recommendation = recommend_strategy(result)
    config = yaml.safe_load(recommendation.mergekit_yaml)
    assert recommendation.method == MergeMethod.TASK_ARITHMETIC
    assert config["base_model"] == "org/actual-base"
    assert {item["model"] for item in config["models"]} == {
        "org/candidate-0",
        "org/candidate-1",
    }
    assert all("weight" in item["parameters"] for item in config["models"])
    assert all("density" not in item["parameters"] for item in config["models"])


def test_explicit_base_default_is_task_arithmetic_not_list_position():
    result = _result(explicit=True, candidate_count=2, sign=0.1, energy=0.4)
    recommendation = recommend_strategy(result)
    config = yaml.safe_load(recommendation.mergekit_yaml)
    assert recommendation.method == MergeMethod.TASK_ARITHMETIC
    assert config["base_model"] == "org/actual-base"


def test_three_models_without_base_emits_equal_linear_baseline():
    result = _result(explicit=False, candidate_count=3)
    config = yaml.safe_load(_generate_yaml(MergeMethod.LINEAR, result))
    assert [item["model"] for item in config["models"]] == [
        "org/candidate-0",
        "org/candidate-1",
        "org/candidate-2",
    ]
    assert [item["parameters"]["weight"] for item in config["models"]] == [1 / 3] * 3


def test_generated_yaml_is_accepted_by_installed_mergekit_parser():
    pytest.importorskip("mergekit")

    configurations = [
        _generate_yaml(MergeMethod.SLERP, _result(explicit=False, candidate_count=2)),
        _generate_yaml(
            MergeMethod.TIES,
            _result(explicit=True, candidate_count=2, sign=0.5),
            density=0.5,
        ),
        _generate_yaml(MergeMethod.LINEAR, _result(explicit=False, candidate_count=3)),
    ]
    for configuration in configurations:
        valid, target = validate_mergekit_yaml(configuration)
        assert valid
        assert target and target.startswith("mergekit ")
