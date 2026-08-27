"""Tests for explicit-base interference and non-causal source similarity."""

import torch

from mergelens.compare.loader import ModelHandle
from mergelens.diagnose.attribution import compute_source_similarity_profile
from mergelens.diagnose.interference import compute_interference
from tests.conftest import _create_tiny_model


def test_interference_requires_two_sources(tmp_model_path):
    assert compute_interference([ModelHandle(tmp_model_path)]) == []


def test_explicit_base_changes_task_vector_construction(tmp_path):
    paths = []
    for index, seed in enumerate([1, 2, 3, 4]):
        path = tmp_path / str(index)
        path.mkdir()
        _create_tiny_model(path, seed=seed, hidden=8, layers=1)
        paths.append(str(path))
    sources = [ModelHandle(paths[0]), ModelHandle(paths[1])]
    first = compute_interference(sources, base_handle=ModelHandle(paths[2]))
    second = compute_interference(sources, base_handle=ModelHandle(paths[3]))
    assert [item.score for item in first] != [item.score for item in second]


def test_scalar_model_weights_affect_no_base_proxy(tmp_models):
    sources = [ModelHandle(path) for path in tmp_models]
    equal = compute_interference(sources, weights=[0.5, 0.5])
    skewed = compute_interference(sources, weights=[0.9, 0.1])
    assert [item.score for item in equal] != [item.score for item in skewed]


def test_source_similarity_profile_is_raw_not_normalized(tmp_models):
    first, second = [ModelHandle(path) for path in tmp_models]
    profiles = compute_source_similarity_profile(first, [first, second])
    assert profiles
    assert all(profile[first.path_or_repo] == 1.0 for profile in profiles.values())
    assert any(abs(sum(profile.values()) - 1.0) > 0.1 for profile in profiles.values())
    assert any(value < 0 for profile in profiles.values() for value in profile.values())


def test_task_vector_interference_is_bounded(tmp_path):
    paths = []
    for index, seed in enumerate([10, 11, 12]):
        path = tmp_path / str(index)
        path.mkdir()
        _create_tiny_model(path, seed=seed, hidden=8, layers=1)
        paths.append(str(path))
    scores = compute_interference(
        [ModelHandle(paths[1]), ModelHandle(paths[2])], base_handle=ModelHandle(paths[0])
    )
    assert scores
    assert all(0 <= item.score <= 1 for item in scores)
    assert all(item.source_similarity_profile for item in scores)


def test_zero_sum_weights_are_rejected(tmp_models):
    sources = [ModelHandle(path) for path in tmp_models]
    try:
        compute_interference(sources, weights=[1.0, -1.0])
    except ValueError as exc:
        assert "sum to zero" in str(exc)
    else:
        raise AssertionError("zero-sum source weights were accepted")
