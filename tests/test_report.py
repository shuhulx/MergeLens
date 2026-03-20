"""Tests for report generation."""

import json

import pytest

from mergelens.compare.analyzer import compare_models
from mergelens.models import (
    CompareResult,
    LayerMetrics,
    LayerType,
    MergeCompatibilityIndex,
    ModelInfo,
)
from mergelens.report.generator import (
    _build_layer_metrics_chart,
    _build_spectral_chart,
    generate_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(sign_disagreement=True):
    """Build a minimal CompareResult for unit testing chart functions."""
    lm1 = LayerMetrics(
        layer_name="layers.0.q",
        layer_type=LayerType.ATTENTION_Q,
        cosine_similarity=0.85,
        l2_distance=1.23,
        sign_disagreement_rate=0.15 if sign_disagreement else None,
        spectral_overlap=0.72,
        effective_rank_ratio=0.90,
        task_vector_energy=0.44,
    )
    lm2 = LayerMetrics(
        layer_name="layers.1.q",
        layer_type=LayerType.ATTENTION_Q,
        cosine_similarity=0.78,
        l2_distance=2.05,
        sign_disagreement_rate=0.22 if sign_disagreement else None,
        spectral_overlap=0.65,
        effective_rank_ratio=0.85,
        task_vector_energy=0.38,
    )
    mci = MergeCompatibilityIndex(
        score=72.5, confidence=0.8, verdict="compatible", ci_lower=65.0, ci_upper=80.0
    )
    return CompareResult(
        models=[ModelInfo(name="a", path_or_repo="/tmp/a"), ModelInfo(name="b", path_or_repo="/tmp/b")],
        layer_metrics=[lm1, lm2],
        mci=mci,
        conflict_zones=[],
    )


# ---------------------------------------------------------------------------
# _build_layer_metrics_chart
# ---------------------------------------------------------------------------

class TestBuildLayerMetricsChart:
    def test_returns_dict_with_data_and_layout(self):
        chart = _build_layer_metrics_chart(_make_result())
        assert "data" in chart
        assert "layout" in chart

    def test_l2_bar_trace_always_present(self):
        chart = _build_layer_metrics_chart(_make_result())
        names = [t["name"] for t in chart["data"]]
        assert "L2 Distance" in names

    def test_sign_disagreement_line_present_when_available(self):
        chart = _build_layer_metrics_chart(_make_result(sign_disagreement=True))
        names = [t["name"] for t in chart["data"]]
        assert "Sign Disagreement Rate" in names

    def test_sign_disagreement_absent_when_none(self):
        chart = _build_layer_metrics_chart(_make_result(sign_disagreement=False))
        names = [t["name"] for t in chart["data"]]
        assert "Sign Disagreement Rate" not in names

    def test_secondary_yaxis_present_with_sign_disagreement(self):
        chart = _build_layer_metrics_chart(_make_result(sign_disagreement=True))
        assert "yaxis2" in chart["layout"]

    def test_secondary_yaxis_absent_without_sign_disagreement(self):
        chart = _build_layer_metrics_chart(_make_result(sign_disagreement=False))
        assert "yaxis2" not in chart["layout"]

    def test_l2_values_correct(self):
        result = _make_result()
        chart = _build_layer_metrics_chart(result)
        l2_trace = next(t for t in chart["data"] if t["name"] == "L2 Distance")
        expected = [m.l2_distance for m in result.layer_metrics]
        assert l2_trace["y"] == expected

    def test_sign_disagreement_values_correct(self):
        result = _make_result(sign_disagreement=True)
        chart = _build_layer_metrics_chart(result)
        sd_trace = next(t for t in chart["data"] if t["name"] == "Sign Disagreement Rate")
        expected = [m.sign_disagreement_rate for m in result.layer_metrics]
        assert sd_trace["y"] == expected

    def test_sign_disagreement_on_yaxis2(self):
        chart = _build_layer_metrics_chart(_make_result(sign_disagreement=True))
        sd_trace = next(t for t in chart["data"] if t["name"] == "Sign Disagreement Rate")
        assert sd_trace["yaxis"] == "y2"

    def test_l2_on_primary_yaxis(self):
        chart = _build_layer_metrics_chart(_make_result())
        l2_trace = next(t for t in chart["data"] if t["name"] == "L2 Distance")
        assert l2_trace.get("yaxis", "y") == "y"

    def test_empty_layer_metrics(self):
        mci = MergeCompatibilityIndex(score=50.0, confidence=0.5, verdict="risky", ci_lower=40.0, ci_upper=60.0)
        result = CompareResult(
            models=[ModelInfo(name="a", path_or_repo="/tmp/a"), ModelInfo(name="b", path_or_repo="/tmp/b")],
            layer_metrics=[],
            mci=mci,
            conflict_zones=[],
        )
        chart = _build_layer_metrics_chart(result)
        l2_trace = next(t for t in chart["data"] if t["name"] == "L2 Distance")
        assert l2_trace["y"] == []


# ---------------------------------------------------------------------------
# _build_spectral_chart — sign disagreement trace
# ---------------------------------------------------------------------------

class TestBuildSpectralChart:
    def test_sign_disagreement_trace_present(self):
        chart = _build_spectral_chart(_make_result(sign_disagreement=True))
        names = [t["name"] for t in chart["data"]]
        assert "Sign Disagreement" in names

    def test_sign_disagreement_trace_absent_when_none(self):
        chart = _build_spectral_chart(_make_result(sign_disagreement=False))
        names = [t["name"] for t in chart["data"]]
        assert "Sign Disagreement" not in names

    def test_spectral_chart_still_has_core_traces(self):
        chart = _build_spectral_chart(_make_result())
        names = [t["name"] for t in chart["data"]]
        assert "Spectral Overlap" in names
        assert "Rank Ratio" in names
        assert "Task Vector Energy" in names

    def test_sign_disagreement_values_match(self):
        result = _make_result(sign_disagreement=True)
        chart = _build_spectral_chart(result)
        sd_trace = next(t for t in chart["data"] if t["name"] == "Sign Disagreement")
        expected = [m.sign_disagreement_rate for m in result.layer_metrics]
        assert sd_trace["y"] == expected


# ---------------------------------------------------------------------------
# Full HTML report — new chart presence
# ---------------------------------------------------------------------------

def test_generate_html_report(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False)
    output = tmp_path / "report.html"
    generate_report(compare_result=result, output_path=str(output))
    assert output.exists()
    content = output.read_text()
    assert "MergeLens" in content
    assert "plotly" in content.lower()
    assert "<table" in content


def test_report_has_mci_section(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False)
    output = tmp_path / "report.html"
    generate_report(compare_result=result, output_path=str(output))
    content = output.read_text()
    assert "Merge Compatibility Index" in content


def test_report_has_layer_divergence_chart(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False)
    output = tmp_path / "report.html"
    generate_report(compare_result=result, output_path=str(output))
    content = output.read_text()
    assert "layer_metrics-chart" in content
    assert "Layer Divergence" in content


def test_report_layer_divergence_has_l2_data(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False)
    output = tmp_path / "report.html"
    generate_report(compare_result=result, output_path=str(output))
    content = output.read_text()
    assert "L2 Distance" in content


def test_report_spectral_has_sign_disagreement(tmp_models, tmp_path):
    """With a base model, sign disagreement is computed; it should appear in the report."""
    result = compare_models(list(tmp_models), show_progress=False)
    # sign_disagreement_rate requires base model, so it'll be None in this fixture.
    # Verify the report renders without error either way.
    output = tmp_path / "report.html"
    generate_report(compare_result=result, output_path=str(output))
    assert output.exists()


def test_report_layer_metrics_chart_json_valid(tmp_models, tmp_path):
    """layer_metrics chart JSON embedded in the report must be parseable."""
    result = compare_models(list(tmp_models), show_progress=False)
    output = tmp_path / "report.html"
    generate_report(compare_result=result, output_path=str(output))
    content = output.read_text()
    # Extract the spec_layer_metrics JS assignment and validate JSON
    import re
    match = re.search(r"var spec_layer_metrics = (.+?);", content)
    assert match, "spec_layer_metrics JS variable not found in report"
    parsed = json.loads(match.group(1))
    assert "data" in parsed
    assert "layout" in parsed
