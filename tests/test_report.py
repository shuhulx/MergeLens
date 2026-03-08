"""Tests for report generation."""

from mergelens.compare.analyzer import compare_models
from mergelens.report.generator import generate_report


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
