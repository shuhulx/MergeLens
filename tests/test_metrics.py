"""Known-answer and resource-policy tests for underlying diagnostic signals."""

import pytest
import torch

from mergelens.compare.metrics import (
    DEFAULT_METRICS,
    DIAGNOSTIC_METRIC_NAMES,
    METRIC_REGISTRY,
    centered_task_vector_energy,
    cosine_similarity,
    effective_rank_ratio,
    l2_distance,
    sign_disagreement_rate,
    spectral_subspace_overlap,
    tsv_interference_score,
    weight_distribution_divergence,
)
from mergelens.utils.tensor_ops import SVDResourceLimitError, bounded_singular_values


def test_exact_shape_is_required_even_when_numel_matches():
    first = torch.arange(6).reshape(2, 3)
    second = torch.arange(6).reshape(3, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_similarity(first, second)
    with pytest.raises(ValueError, match="shape mismatch"):
        l2_distance(first, second)


def test_cosine_zero_vector_semantics_are_deliberate():
    zero = torch.zeros(4)
    nonzero = torch.ones(4)
    assert cosine_similarity(zero, zero) == 1.0
    assert cosine_similarity(zero, nonzero) == 0.0


def test_l2_and_cosine_known_answers():
    values = torch.tensor([1.0, -1.0])
    assert cosine_similarity(values, values) == pytest.approx(1.0)
    assert cosine_similarity(values, -values) == pytest.approx(-1.0)
    assert l2_distance(values, values) == 0.0


def test_directional_weight_divergence_is_experimental_and_not_default():
    first = torch.tensor([1.0, 2.0, 3.0])
    second = torch.tensor([3.0, 2.0, 1.0])
    assert weight_distribution_divergence(first, second) >= 0.0
    assert "weight_distribution_divergence" not in DEFAULT_METRICS
    assert weight_distribution_divergence(torch.tensor([1.0]), torch.tensor([2.0])) >= 0


def test_spectral_overlap_rejects_vectors_and_handles_matrices():
    with pytest.raises(ValueError, match="at least two rows and columns"):
        spectral_subspace_overlap(torch.ones(8), torch.ones(8))
    matrix = torch.randn(8, 8)
    assert spectral_subspace_overlap(matrix, matrix, k=3) == pytest.approx(1.0, abs=1e-4)
    assert effective_rank_ratio(matrix, matrix) == pytest.approx(1.0)


def test_sign_disagreement_counts_zero_vs_nonzero():
    assert sign_disagreement_rate([torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])]) == 0.5
    assert sign_disagreement_rate([torch.ones(4), -torch.ones(4)]) == 1.0


def test_task_vector_metrics_require_two_vectors():
    with pytest.raises(ValueError, match="at least two"):
        sign_disagreement_rate([torch.ones(2, 2)])
    with pytest.raises(ValueError, match="at least two"):
        tsv_interference_score([torch.ones(2, 2)])


def test_tsv_uses_actual_retained_rank_when_k_exceeds_rank():
    first = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    second = first.clone()
    assert tsv_interference_score([first, second], k=64) == pytest.approx(1.0)


def test_tsv_opposite_and_identical_vectors_have_same_subspace():
    first = torch.diag(torch.tensor([2.0, 1.0]))
    assert tsv_interference_score([first, first], k=2) == pytest.approx(1.0)
    assert tsv_interference_score([first, -first], k=2) == pytest.approx(1.0)


def test_task_vector_energy_is_bounded_and_rank_one_is_concentrated():
    column = torch.randn(16, 1)
    rank_one = column @ column.T
    assert centered_task_vector_energy(rank_one, k=1) > 0.99
    assert 0 <= centered_task_vector_energy(torch.randn(8, 8), k=2) <= 1


def test_every_svd_path_applies_resource_limit(monkeypatch):
    import mergelens.utils.tensor_ops as tensor_ops

    monkeypatch.setattr(tensor_ops, "MAX_ELEMENTS_FOR_SVD", 4)
    with pytest.raises(SVDResourceLimitError):
        bounded_singular_values(torch.ones(3, 3))
    with pytest.raises(SVDResourceLimitError):
        centered_task_vector_energy(torch.ones(3, 3))
    with pytest.raises(SVDResourceLimitError):
        tsv_interference_score([torch.ones(3, 3), torch.eye(3)])


def test_registry_and_exact_signal_count_are_consistent():
    assert len(DIAGNOSTIC_METRIC_NAMES) == 9
    assert set(METRIC_REGISTRY) == set(DIAGNOSTIC_METRIC_NAMES)
