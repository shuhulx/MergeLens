"""Tests for strategy recommender."""

from mergelens.compare.strategy import recommend_strategy
from mergelens.models import (
    CompareResult,
    LayerMetrics,
    LayerType,
    MergeCompatibilityIndex,
    MergeMethod,
    ModelInfo,
)


def _make_result(avg_cos=0.95, sign_disagree=0.1, energy=0.5, spectral=0.8, rank=0.9, mci_score=80):
    metrics = [
        LayerMetrics(
            layer_name=f"layer.{i}", layer_type=LayerType.OTHER,
            cosine_similarity=avg_cos, l2_distance=0.1,
            spectral_overlap=spectral, effective_rank_ratio=rank,
            sign_disagreement_rate=sign_disagree, task_vector_energy=energy,
        )
        for i in range(10)
    ]
    mci = MergeCompatibilityIndex(
        score=mci_score, confidence=0.8, ci_lower=mci_score-10, ci_upper=mci_score+10,
        verdict="compatible",
    )
    return CompareResult(
        models=[ModelInfo(name="a", path_or_repo="a"), ModelInfo(name="b", path_or_repo="b")],
        layer_metrics=metrics, conflict_zones=[], mci=mci,
    )


def test_slerp_for_compatible():
    result = _make_result(avg_cos=0.95, sign_disagree=0.1, mci_score=85)
    rec = recommend_strategy(result)
    assert rec.method == MergeMethod.SLERP


def test_ties_for_high_sign_disagreement():
    result = _make_result(sign_disagree=0.4, mci_score=60)
    rec = recommend_strategy(result)
    assert rec.method == MergeMethod.TIES


def test_dare_for_concentrated_energy():
    result = _make_result(energy=0.85, sign_disagree=0.1, mci_score=70)
    rec = recommend_strategy(result)
    assert rec.method == MergeMethod.DARE_TIES


def test_linear_for_low_spectral():
    result = _make_result(spectral=0.3, rank=0.4, sign_disagree=0.1, mci_score=50)
    rec = recommend_strategy(result)
    assert rec.method == MergeMethod.LINEAR


def test_warning_for_low_mci():
    result = _make_result(mci_score=20)
    rec = recommend_strategy(result)
    assert rec.confidence <= 0.4
    assert len(rec.warnings) > 0
