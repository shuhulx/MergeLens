"""Tests for attributable JSON, CSV, and Markdown exports."""

import csv
import json

from mergelens.compare.analyzer import compare_models
from mergelens.report.exporters import export_csv, export_json, export_markdown


def test_json_uses_v03_evidence_model(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False, metrics=["cosine_similarity"])
    path = tmp_path / "result.json"
    assert export_json(result, str(path)) == str(path)
    data = json.loads(path.read_text())
    assert "coverage" in data
    assert "tensor_metrics" in data
    assert "metric_availability" in data
    assert "layer_metrics" not in data
    assert data["mci"]["validation_status"] == "heuristic_unvalidated"
    assert data["tensor_metrics"][0]["reference_model"] == tmp_models[0]


def test_csv_includes_pair_tensor_shape_and_position(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False, metrics=["cosine_similarity"])
    path = tmp_path / "result.csv"
    export_csv(result, str(path))
    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert rows
    assert rows[0]["comparison_id"] == "comparison_0"
    assert rows[0]["reference_model"] == tmp_models[0]
    assert rows[0]["candidate_model"] == tmp_models[1]
    assert rows[0]["tensor_name"]
    assert rows[0]["shape"]
    assert rows[0]["tensor_position"] == "0"


def test_markdown_uses_heuristic_and_evidence_language(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False)
    path = tmp_path / "result.md"
    export_markdown(result, str(path))
    content = path.read_text()
    assert "Unvalidated heuristic" in content
    assert "Validation status: `heuristic_unvalidated`" in content
    assert "## Coverage" in content
    assert "## Metric availability" in content
    assert "does not establish downstream merged-model quality" in content
    assert "Confidence:" not in content


def test_empty_tensor_export_has_header_only(tmp_models, tmp_path):
    result = compare_models(list(tmp_models), show_progress=False, metrics=["cosine_similarity"])
    result.tensor_metrics = []
    path = tmp_path / "empty.csv"
    export_csv(result, str(path))
    assert len(path.read_text().splitlines()) == 1
