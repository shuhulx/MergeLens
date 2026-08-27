"""Regression tests for comparison identity, coverage, selection, and streaming."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import mergelens.compare.analyzer as analyzer
from mergelens.activations.cka import compare_activations_cka
from mergelens.activations.extractor import ActivationSet
from mergelens.compare.analyzer import compare_models
from mergelens.models import MetricStatus, ModelRole
from tests.conftest import _create_tiny_model


def _models(tmp_path: Path, seeds: list[int]) -> list[str]:
    paths = []
    for index, seed in enumerate(seeds):
        path = tmp_path / f"model_{index}"
        path.mkdir()
        _create_tiny_model(path, seed=seed, hidden=8, layers=1)
        paths.append(str(path))
    return paths


def test_two_model_implicit_reference_has_attributable_rows(tmp_models):
    result = compare_models(list(tmp_models), show_progress=False, metrics=["cosine_similarity"])
    assert result.reference_model.role == ModelRole.IMPLICIT_REFERENCE
    assert result.explicit_base is None
    assert len(result.coverage) == 1
    assert all(row.reference_model == tmp_models[0] for row in result.tensor_metrics)
    assert all(row.candidate_model == tmp_models[1] for row in result.tensor_metrics)
    assert all(row.comparison_id == "comparison_0" for row in result.tensor_metrics)


def test_two_models_with_explicit_base_preserves_all_roles(tmp_path):
    base, first, second = _models(tmp_path, [1, 2, 3])
    result = compare_models(
        [first, second], base_model=base, show_progress=False, metrics=["cosine_similarity"]
    )
    assert result.explicit_base.path_or_repo == base
    assert result.explicit_base.role == ModelRole.EXPLICIT_SHARED_BASE
    assert {model.path_or_repo for model in result.models} == {first, second}
    assert len(result.coverage) == 2
    assert {row.candidate_model for row in result.tensor_metrics} == {first, second}


def test_three_model_comparison_creates_two_pair_groups(tmp_path):
    models = _models(tmp_path, [4, 5, 6])
    result = compare_models(models, show_progress=False, metrics=["cosine_similarity"])
    assert set(result.pair_assessments) == {"comparison_0", "comparison_1"}
    assert {row.comparison_id for row in result.tensor_metrics} == {
        "comparison_0",
        "comparison_1",
    }
    assert result.metadata["aggregate_rule"] == "minimum_pair_score"
    scores = [item.score for item in result.pair_assessments.values()]
    assert result.mci.score == min(score for score in scores if score is not None)


def test_three_models_plus_explicit_base_creates_three_groups(tmp_path):
    base, *candidates = _models(tmp_path, [7, 8, 9, 10])
    result = compare_models(
        candidates,
        base_model=base,
        show_progress=False,
        metrics=["cosine_similarity", "sign_disagreement_rate", "tsv_interference"],
    )
    assert len(result.coverage) == 3
    assert {row.comparison_id for row in result.tensor_metrics} == {
        "comparison_0",
        "comparison_1",
        "comparison_2",
    }
    assert all(row.reference_model == base for row in result.tensor_metrics)
    assert all(row.sign_disagreement_rate is not None for row in result.tensor_metrics)


def test_conflict_regions_never_cross_pair_boundaries(tmp_path):
    base, *candidates = _models(tmp_path, [11, 12, 13])
    result = compare_models(
        candidates,
        base_model=base,
        show_progress=False,
        metrics=["cosine_similarity", "sign_disagreement_rate"],
    )
    rows_by_pair = {}
    for row in result.tensor_metrics:
        rows_by_pair.setdefault(row.comparison_id, set()).add(row.tensor_name)
    assert result.tensor_conflict_regions
    for region in result.tensor_conflict_regions:
        assert set(region.tensor_names) <= rows_by_pair[region.comparison_id]
        assert all(
            row.candidate_model == region.candidate_model
            for row in result.tensor_metrics
            if row.comparison_id == region.comparison_id
        )


def test_selective_metric_execution_marks_other_signals_skipped(tmp_models):
    result = compare_models(list(tmp_models), show_progress=False, metrics=["cosine_similarity"])
    row = result.tensor_metrics[0]
    assert row.cosine_similarity is not None
    assert row.l2_distance is None
    assert row.metric_observations["l2_distance"].status == MetricStatus.SKIPPED_BY_USER
    availability = {item.metric: item for item in result.metric_availability}
    assert availability["l2_distance"].status == MetricStatus.SKIPPED_BY_USER


def test_unknown_metric_fails_visibly(tmp_models):
    with pytest.raises(ValueError, match="Unknown metric"):
        compare_models(list(tmp_models), show_progress=False, metrics=["imaginary"])


def test_progress_wrapper_does_not_eagerly_consume_iterator(tmp_models, monkeypatch):
    original = analyzer.iter_aligned_tensors
    consumed = []

    def observed_iterator(handles, names):
        for item in original(handles, names):
            consumed.append(item[0])
            yield item

    def observed_track(iterator, *, total, description):
        assert consumed == []
        assert total > 0
        assert "Comparing" in description
        return iterator

    monkeypatch.setattr(analyzer, "iter_aligned_tensors", observed_iterator)
    monkeypatch.setattr(analyzer, "track", observed_track)
    compare_models(list(tmp_models), show_progress=True, metrics=["cosine_similarity"])
    assert consumed


def test_shape_mismatch_is_excluded_and_suppresses_score(tmp_path):
    first, second = _models(tmp_path, [14, 15])
    tensors = load_file(str(Path(second) / "model.safetensors"))
    original = tensors["model.embed_tokens.weight"]
    tensors["model.embed_tokens.weight"] = original.reshape(50, 16)
    save_file(tensors, str(Path(second) / "model.safetensors"))
    config = json.loads((Path(second) / "config.json").read_text())
    config["vocab_size"] = 50
    config["hidden_size"] = 16
    (Path(second) / "config.json").write_text(json.dumps(config))

    result = compare_models([first, second], show_progress=False, metrics=["cosine_similarity"])
    coverage = result.coverage[0]
    assert any(
        item.tensor_name == "model.embed_tokens.weight" for item in coverage.shape_mismatches
    )
    assert all(row.tensor_name != "model.embed_tokens.weight" for row in result.tensor_metrics)
    assert result.mci.score is None
    assert coverage.scoring_supported is False


def test_identical_architecture_coverage_accounts_all_parameters(tmp_identical_models):
    result = compare_models(
        list(tmp_identical_models), show_progress=False, metrics=["cosine_similarity"]
    )
    coverage = result.coverage[0]
    assert coverage.exact_shape_compatible_tensor_count == coverage.total_tensor_count_reference
    assert coverage.common_parameter_count == coverage.total_parameter_count_reference
    assert coverage.parameter_coverage_reference == 1.0
    assert coverage.parameter_coverage_candidate == 1.0
    assert coverage.appears_homologous


def test_metric_availability_explains_missing_task_and_activation_signals(tmp_models):
    result = compare_models(list(tmp_models), show_progress=False)
    availability = {item.metric: item for item in result.metric_availability}
    assert availability["cka_similarity"].status == MetricStatus.STRUCTURALLY_UNAVAILABLE
    assert "calibration activations" in availability["cka_similarity"].reason
    assert availability["sign_disagreement_rate"].status == MetricStatus.STRUCTURALLY_UNAVAILABLE
    assert "explicit shared base" in availability["sign_disagreement_rate"].reason


def test_only_provenance_bearing_aligned_cka_enters_comparison(tmp_models):
    first = ActivationSet({"model.layers.0": torch.randn(12, 5)}, "calibration-id", 12)
    second = ActivationSet({"model.layers.0": torch.randn(12, 7)}, "calibration-id", 12)
    comparison = compare_activations_cka(first, second)
    result = compare_models(
        list(tmp_models),
        show_progress=False,
        metrics=["cka_similarity"],
        cka_comparisons={tmp_models[1]: comparison},
    )
    block_rows = [row for row in result.tensor_metrics if row.transformer_block == 0]
    assert block_rows
    assert all(row.cka_similarity == comparison["model.layers.0"] for row in block_rows)
    assert result.metadata["cka_provenance"][tmp_models[1]]["calibration_id"] == "calibration-id"


def test_plain_cka_score_dictionary_is_rejected(tmp_models):
    with pytest.raises(TypeError, match="CKAComparison"):
        compare_models(
            list(tmp_models),
            show_progress=False,
            metrics=["cka_similarity"],
            cka_comparisons={tmp_models[1]: {"model.layers.0": 0.5}},  # type: ignore[dict-item]
        )


def test_too_few_models_is_rejected(tmp_model_path):
    with pytest.raises(ValueError, match="at least 2"):
        compare_models([tmp_model_path], show_progress=False)
