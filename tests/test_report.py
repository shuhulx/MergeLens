"""Tests for offline reports and explicit multi-model grouping."""

import json
import re

from mergelens.compare.analyzer import compare_models
from mergelens.report.generator import (
    _build_similarity_heatmap,
    _build_spectral_chart,
    _build_tensor_metrics_chart,
    generate_report,
)
from tests.conftest import _create_tiny_model


def _four_models(tmp_path):
    paths = []
    for index, seed in enumerate([21, 22, 23, 24]):
        path = tmp_path / f"checkpoint_{index}"
        path.mkdir()
        _create_tiny_model(path, seed=seed, hidden=8, layers=1)
        paths.append(str(path))
    return paths


def test_heatmap_groups_interleaved_rows_by_comparison(tmp_path):
    base, *candidates = _four_models(tmp_path)
    result = compare_models(
        candidates, base_model=base, show_progress=False, metrics=["cosine_similarity"]
    )
    result.tensor_metrics = result.tensor_metrics[::2] + result.tensor_metrics[1::2]
    chart = _build_similarity_heatmap(result)
    assert len(chart["data"][0]["z"]) == 3
    assert set(chart["data"][0]["y"]) == {f"{base} vs {candidate}" for candidate in candidates}


def test_line_charts_keep_pair_identity(tmp_path):
    base, *candidates = _four_models(tmp_path)
    result = compare_models(
        candidates,
        base_model=base,
        show_progress=False,
        metrics=["l2_distance", "spectral_overlap"],
    )
    spectral = _build_spectral_chart(result)
    tensor = _build_tensor_metrics_chart(result)
    names = [trace["name"] for trace in spectral["data"] + tensor["data"]]
    for candidate in candidates:
        assert any(f"{base} vs {candidate}" in name for name in names)
    assert all(trace.get("connectgaps") is False for trace in spectral["data"] + tensor["data"])


def test_report_for_three_models_plus_explicit_base_is_offline_and_attributable(tmp_path):
    base, *candidates = _four_models(tmp_path)
    result = compare_models(candidates, base_model=base, show_progress=False)
    path = tmp_path / "report.html"
    generate_report(compare_result=result, output_path=str(path))
    content = path.read_text()
    assert "<script src=" not in content
    assert "plotly.js" in content.lower()
    assert "Pairwise comparison coverage" in content
    assert "Metric availability" in content
    assert "heuristic_unvalidated" in content
    assert "statistical confidence" not in content.lower()
    assert "tensor_metrics-chart" in content
    for candidate in candidates:
        assert candidate in content
    for chart_id in ("heuristic", "heatmap", "spectral", "tensor_metrics", "conflicts"):
        match = re.search(rf"const spec_{chart_id}=(.*?); Plotly", content)
        assert match, f"missing embedded {chart_id} chart specification"
        assert isinstance(json.loads(match.group(1)), dict)


def test_report_escapes_user_supplied_title(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False, metrics=["cosine_similarity"])
    path = tmp_path / "escaped.html"
    generate_report(compare_result=result, output_path=str(path), title="<script>alert(1)</script>")
    content = path.read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
