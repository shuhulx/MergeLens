"""Tests for the comparison analyzer."""

import pytest

from mergelens.compare.analyzer import compare_models


def test_compare_basic(tmp_models):
    result = compare_models(
        model_paths=list(tmp_models),
        show_progress=False,
    )
    assert result.mci.score >= 0
    assert result.mci.score <= 100
    assert len(result.layer_metrics) > 0
    assert len(result.models) == 2


def test_compare_identical(tmp_identical_models):
    result = compare_models(
        model_paths=list(tmp_identical_models),
        show_progress=False,
    )
    assert result.mci.score > 90
    assert result.mci.verdict == "highly compatible"


def test_compare_with_strategy(tmp_models):
    result = compare_models(
        model_paths=list(tmp_models),
        show_progress=False,
        include_strategy=True,
    )
    assert result.strategy is not None
    assert result.strategy.method is not None
    assert len(result.strategy.mergekit_yaml) > 0


def test_compare_too_few_models(tmp_model_path):
    with pytest.raises(ValueError, match="at least 2"):
        compare_models([tmp_model_path], show_progress=False)
