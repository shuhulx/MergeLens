"""Tests for diagnose module — interference scoring and attribution."""

import pytest

from mergelens.compare.loader import ModelHandle, find_common_tensors
from mergelens.diagnose.attribution import compute_attribution
from mergelens.diagnose.interference import compute_interference
from mergelens.models import InterferenceScore

# ── Interference ─────────────────────────────────────────────────


def test_interference_returns_empty_for_single_model(tmp_model_path):
    h = ModelHandle(tmp_model_path)
    scores = compute_interference([h])
    assert scores == []


def test_interference_returns_scores_for_two_models(tmp_models):
    a, b = tmp_models
    ha, hb = ModelHandle(a), ModelHandle(b)
    scores = compute_interference([ha, hb])
    assert len(scores) > 0
    assert all(isinstance(s, InterferenceScore) for s in scores)


def test_interference_scores_bounded(tmp_models):
    a, b = tmp_models
    ha, hb = ModelHandle(a), ModelHandle(b)
    scores = compute_interference([ha, hb])
    for s in scores:
        assert 0.0 <= s.score <= 1.0


def test_interference_identical_models_low(tmp_identical_models):
    a, b = tmp_identical_models
    ha, hb = ModelHandle(a), ModelHandle(b)
    scores = compute_interference([ha, hb])
    for s in scores:
        assert s.score < 0.01


def test_interference_source_contributions_present(tmp_models):
    a, b = tmp_models
    ha, hb = ModelHandle(a), ModelHandle(b)
    scores = compute_interference([ha, hb])
    for s in scores:
        assert len(s.source_contributions) == 2


def test_interference_covers_all_common_layers(tmp_models):
    a, b = tmp_models
    ha, hb = ModelHandle(a), ModelHandle(b)
    common = find_common_tensors([ha, hb])
    scores = compute_interference([ha, hb])
    score_names = {s.layer_name for s in scores}
    assert score_names == set(common)


def test_interference_three_models(tmp_path):
    from tests.conftest import _create_tiny_model

    dirs = []
    for i, seed in enumerate([42, 123, 999]):
        d = tmp_path / f"model_{i}"
        d.mkdir()
        _create_tiny_model(d, seed=seed)
        dirs.append(str(d))

    handles = [ModelHandle(d) for d in dirs]
    scores = compute_interference(handles)
    assert len(scores) > 0
    for s in scores:
        assert 0.0 <= s.score <= 1.0
        assert len(s.source_contributions) == 3


def test_interference_layer_name_matches(tmp_models):
    a, b = tmp_models
    ha, hb = ModelHandle(a), ModelHandle(b)
    scores = compute_interference([ha, hb])
    for s in scores:
        assert isinstance(s.layer_name, str)
        assert len(s.layer_name) > 0


# ── Attribution ──────────────────────────────────────────────────


def test_attribution_returns_dict(tmp_models):
    a, b = tmp_models
    merged = ModelHandle(a)
    sources = [ModelHandle(b)]
    result = compute_attribution(merged, sources)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_attribution_contributions_sum_to_one(tmp_models):
    a, b = tmp_models
    merged = ModelHandle(a)
    sources = [ModelHandle(a), ModelHandle(b)]
    result = compute_attribution(merged, sources)
    for _layer, contribs in result.items():
        total = sum(contribs.values())
        assert total == pytest.approx(1.0, abs=0.01)


def test_attribution_merged_matches_source_high(tmp_identical_models):
    a, b = tmp_identical_models
    merged = ModelHandle(a)
    sources = [ModelHandle(a), ModelHandle(b)]
    result = compute_attribution(merged, sources)
    for _layer, contribs in result.items():
        for _name, score in contribs.items():
            assert score == pytest.approx(0.5, abs=0.01)


def test_attribution_self_dominates(tmp_models):
    a, b = tmp_models
    merged = ModelHandle(a)
    sources = [ModelHandle(a), ModelHandle(b)]
    result = compute_attribution(merged, sources)
    merged_name = merged.info.name
    other_name = sources[1].info.name
    for _layer, contribs in result.items():
        assert contribs[merged_name] >= contribs[other_name]


def test_attribution_covers_common_layers(tmp_models):
    a, b = tmp_models
    merged = ModelHandle(a)
    sources = [ModelHandle(b)]
    all_handles = [merged, *sources]
    common = find_common_tensors(all_handles)
    result = compute_attribution(merged, sources)
    assert set(result.keys()) == set(common)


def test_attribution_values_non_negative(tmp_models):
    a, b = tmp_models
    merged = ModelHandle(a)
    sources = [ModelHandle(a), ModelHandle(b)]
    result = compute_attribution(merged, sources)
    for _layer, contribs in result.items():
        for _name, score in contribs.items():
            assert score >= 0.0


def test_attribution_three_sources(tmp_path):
    from tests.conftest import _create_tiny_model

    dirs = []
    for i, seed in enumerate([42, 123, 999]):
        d = tmp_path / f"model_{i}"
        d.mkdir()
        _create_tiny_model(d, seed=seed)
        dirs.append(str(d))

    merged = ModelHandle(dirs[0])
    sources = [ModelHandle(d) for d in dirs[1:]]
    result = compute_attribution(merged, sources)
    assert len(result) > 0
    for _layer, contribs in result.items():
        assert len(contribs) == 2
        total = sum(contribs.values())
        assert total == pytest.approx(1.0, abs=0.01)
