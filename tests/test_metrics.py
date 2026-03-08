"""Known-answer tests for all metrics."""

import pytest
import torch

from mergelens.compare.metrics import (
    METRIC_REGISTRY,
    centered_task_vector_energy,
    cka_similarity,
    cosine_similarity,
    effective_rank_ratio,
    kl_divergence,
    l2_distance,
    merge_compatibility_index,
    sign_disagreement_rate,
    spectral_subspace_overlap,
    tsv_interference_score,
)


class TestCosine:
    def test_identical(self):
        a = torch.randn(32, 32)
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_negated(self):
        a = torch.randn(32, 32)
        assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal(self):
        a = torch.zeros(4)
        a[0] = 1.0
        b = torch.zeros(4)
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_bounds(self):
        a, b = torch.randn(64, 64), torch.randn(64, 64)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_cosine_similarity_zero_vector(self):
        a = torch.zeros(10, 10)
        b = torch.randn(10, 10)
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestL2:
    def test_identical(self):
        a = torch.randn(32, 32)
        assert l2_distance(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_positive(self):
        a, b = torch.randn(32, 32), torch.randn(32, 32)
        assert l2_distance(a, b) >= 0.0


class TestKL:
    def test_identical(self):
        a = torch.randn(100)
        assert kl_divergence(a, a) == pytest.approx(0.0, abs=1e-3)

    def test_positive(self):
        a, b = torch.randn(100), torch.randn(100)
        assert kl_divergence(a, b) >= 0.0


class TestSpectral:
    def test_identical(self):
        a = torch.randn(32, 32)
        assert spectral_subspace_overlap(a, a, k=8) == pytest.approx(1.0, abs=0.05)

    def test_bounded(self):
        a, b = torch.randn(32, 32), torch.randn(32, 32)
        score = spectral_subspace_overlap(a, b, k=8)
        assert 0.0 <= score <= 1.0


class TestEffectiveRank:
    def test_identical(self):
        a = torch.randn(32, 32)
        assert effective_rank_ratio(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_bounded(self):
        a, b = torch.randn(32, 32), torch.randn(32, 32)
        ratio = effective_rank_ratio(a, b)
        assert 0.0 <= ratio <= 1.0


class TestSignDisagreement:
    def test_identical_zero(self):
        a = torch.randn(100)
        assert sign_disagreement_rate([a, a]) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_high(self):
        a = torch.ones(100)
        b = -torch.ones(100)
        rate = sign_disagreement_rate([a, b])
        assert rate == pytest.approx(1.0, abs=1e-5)

    def test_bounded(self):
        vecs = [torch.randn(100) for _ in range(3)]
        rate = sign_disagreement_rate(vecs)
        assert 0.0 <= rate <= 1.0


class TestTSV:
    def test_single_returns_zero(self):
        assert tsv_interference_score([torch.randn(32, 32)]) == 0.0

    def test_positive(self):
        vecs = [torch.randn(32, 32) for _ in range(3)]
        score = tsv_interference_score(vecs, k=8)
        assert score >= 0.0


class TestEnergy:
    def test_bounded(self):
        tv = torch.randn(32, 32)
        e = centered_task_vector_energy(tv, k=8)
        assert 0.0 <= e <= 1.0

    def test_low_rank_concentrated(self):
        # Create a rank-1 matrix — energy should be concentrated
        a = torch.randn(32, 1)
        tv = a @ a.T
        e = centered_task_vector_energy(tv, k=1)
        assert e > 0.9


class TestCKA:
    def test_identical(self):
        X = torch.randn(50, 32)
        assert cka_similarity(X, X) == pytest.approx(1.0, abs=1e-4)

    def test_bounded(self):
        X, Y = torch.randn(50, 32), torch.randn(50, 32)
        score = cka_similarity(X, Y)
        assert 0.0 <= score <= 1.0


class TestMCI:
    def test_perfect_score(self):
        mci = merge_compatibility_index(cosine_sims=[1.0] * 10)
        assert mci.score > 90
        assert mci.verdict == "highly compatible"

    def test_low_score(self):
        mci = merge_compatibility_index(cosine_sims=[0.1] * 10)
        assert mci.score < 40

    def test_with_all_metrics(self):
        mci = merge_compatibility_index(
            cosine_sims=[0.95, 0.92, 0.88],
            spectral_overlaps=[0.85, 0.80],
            rank_ratios=[0.90, 0.88],
            sign_disagreements=[0.1, 0.15],
            tsv_scores=[0.2, 0.3],
            energy_scores=[0.5, 0.6],
        )
        assert 0 <= mci.score <= 100
        assert mci.confidence > 0.5

    def test_registry(self):
        assert "cosine_similarity" in METRIC_REGISTRY
        assert "merge_compatibility_index" in METRIC_REGISTRY
        assert len(METRIC_REGISTRY) == 10
