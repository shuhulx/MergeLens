"""Tests for report exporters (JSON, CSV, Markdown)."""

import csv
import json
from io import StringIO

import pytest

from mergelens.models import (
    CompareResult,
    ConflictZone,
    LayerMetrics,
    LayerType,
    MergeCompatibilityIndex,
    MergeMethod,
    ModelInfo,
    Severity,
    StrategyRecommendation,
)
from mergelens.report.exporters import export_csv, export_json, export_markdown


@pytest.fixture
def compare_result():
    return CompareResult(
        models=[
            ModelInfo(name="model_a", path_or_repo="/tmp/model_a"),
            ModelInfo(name="model_b", path_or_repo="/tmp/model_b"),
        ],
        layer_metrics=[
            LayerMetrics(
                layer_name="layers.0.self_attn.q_proj",
                layer_type=LayerType.ATTENTION_Q,
                cosine_similarity=0.85,
                l2_distance=1.2,
                spectral_overlap=0.7,
                effective_rank_ratio=0.9,
                sign_disagreement_rate=0.15,
                tsv_interference=0.3,
                task_vector_energy=0.5,
                cka_similarity=0.8,
            ),
            LayerMetrics(
                layer_name="layers.0.mlp.gate_proj",
                layer_type=LayerType.MLP_GATE,
                cosine_similarity=0.60,
                l2_distance=2.5,
            ),
        ],
        conflict_zones=[
            ConflictZone(
                start_layer=0,
                end_layer=1,
                layer_names=["layers.0.self_attn.q_proj", "layers.0.mlp.gate_proj"],
                severity=Severity.MEDIUM,
                avg_cosine_sim=0.725,
                recommendation="Use SLERP with t=0.3",
            ),
        ],
        mci=MergeCompatibilityIndex(
            score=72.0,
            confidence=0.85,
            ci_lower=65.0,
            ci_upper=79.0,
            verdict="compatible",
        ),
        strategy=StrategyRecommendation(
            method=MergeMethod.SLERP,
            confidence=0.80,
            reasoning="Models are reasonably compatible.",
            mergekit_yaml="merge_method: slerp\nt: 0.5\n",
        ),
    )


@pytest.fixture
def compare_result_minimal():
    return CompareResult(
        models=[ModelInfo(name="a", path_or_repo="/a")],
        layer_metrics=[],
        conflict_zones=[],
        mci=MergeCompatibilityIndex(
            score=50.0, confidence=0.5, ci_lower=40.0, ci_upper=60.0, verdict="risky"
        ),
    )


class TestExportJSON:
    def test_creates_file(self, compare_result, tmp_path):
        path = str(tmp_path / "out.json")
        ret = export_json(compare_result, path)
        assert ret == path
        assert (tmp_path / "out.json").exists()

    def test_valid_json(self, compare_result, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(compare_result, path)
        data = json.loads((tmp_path / "out.json").read_text())
        assert isinstance(data, dict)

    def test_contains_expected_keys(self, compare_result, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(compare_result, path)
        data = json.loads((tmp_path / "out.json").read_text())
        for key in ("models", "layer_metrics", "conflict_zones", "mci", "strategy"):
            assert key in data

    def test_layer_metrics_count(self, compare_result, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(compare_result, path)
        data = json.loads((tmp_path / "out.json").read_text())
        assert len(data["layer_metrics"]) == 2

    def test_mci_score(self, compare_result, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(compare_result, path)
        data = json.loads((tmp_path / "out.json").read_text())
        assert data["mci"]["score"] == 72.0

    def test_no_strategy(self, compare_result_minimal, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(compare_result_minimal, path)
        data = json.loads((tmp_path / "out.json").read_text())
        assert data["strategy"] is None


class TestExportCSV:
    def test_creates_file(self, compare_result, tmp_path):
        path = str(tmp_path / "out.csv")
        ret = export_csv(compare_result, path)
        assert ret == path
        assert (tmp_path / "out.csv").exists()

    def test_valid_csv_with_header(self, compare_result, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(compare_result, path)
        reader = csv.reader(StringIO((tmp_path / "out.csv").read_text()))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 layers

    def test_correct_columns(self, compare_result, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(compare_result, path)
        reader = csv.reader(StringIO((tmp_path / "out.csv").read_text()))
        header = next(reader)
        expected = [
            "layer_name", "layer_type", "cosine_similarity", "l2_distance",
            "kl_divergence", "spectral_overlap", "effective_rank_ratio",
            "sign_disagreement_rate", "tsv_interference", "task_vector_energy",
            "cka_similarity",
        ]
        assert header == expected

    def test_row_values(self, compare_result, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(compare_result, path)
        reader = csv.reader(StringIO((tmp_path / "out.csv").read_text()))
        next(reader)
        row = next(reader)
        assert row[0] == "layers.0.self_attn.q_proj"
        assert row[1] == "attn_q"
        assert float(row[2]) == pytest.approx(0.85)

    def test_none_values_in_optional_fields(self, compare_result, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(compare_result, path)
        reader = csv.reader(StringIO((tmp_path / "out.csv").read_text()))
        next(reader)
        next(reader)
        row = next(reader)  # second layer has None optionals
        assert row[4] == ""

    def test_empty_layers(self, compare_result_minimal, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(compare_result_minimal, path)
        reader = csv.reader(StringIO((tmp_path / "out.csv").read_text()))
        rows = list(reader)
        assert len(rows) == 1  # header only


class TestExportMarkdown:
    def test_creates_file(self, compare_result, tmp_path):
        path = str(tmp_path / "out.md")
        ret = export_markdown(compare_result, path)
        assert ret == path
        assert (tmp_path / "out.md").exists()

    def test_has_title(self, compare_result, tmp_path):
        path = str(tmp_path / "out.md")
        export_markdown(compare_result, path)
        content = (tmp_path / "out.md").read_text()
        assert "# MergeLens Report" in content

    def test_has_mci_section(self, compare_result, tmp_path):
        path = str(tmp_path / "out.md")
        export_markdown(compare_result, path)
        content = (tmp_path / "out.md").read_text()
        assert "## Merge Compatibility Index: 72.0/100" in content
        assert "compatible" in content

    def test_has_models_section(self, compare_result, tmp_path):
        path = str(tmp_path / "out.md")
        export_markdown(compare_result, path)
        content = (tmp_path / "out.md").read_text()
        assert "## Models" in content
        assert "model_a" in content
        assert "model_b" in content

    def test_has_conflict_zones(self, compare_result, tmp_path):
        path = str(tmp_path / "out.md")
        export_markdown(compare_result, path)
        content = (tmp_path / "out.md").read_text()
        assert "## Conflict Zones" in content
        assert "medium" in content
        assert "SLERP" in content

    def test_has_strategy_section(self, compare_result, tmp_path):
        path = str(tmp_path / "out.md")
        export_markdown(compare_result, path)
        content = (tmp_path / "out.md").read_text()
        assert "## Recommended Strategy" in content
        assert "slerp" in content
        assert "mergekit_yaml" in content or "merge_method" in content

    def test_no_conflict_zones_when_empty(self, compare_result_minimal, tmp_path):
        path = str(tmp_path / "out.md")
        export_markdown(compare_result_minimal, path)
        content = (tmp_path / "out.md").read_text()
        assert "Conflict Zones" not in content

    def test_no_strategy_when_none(self, compare_result_minimal, tmp_path):
        path = str(tmp_path / "out.md")
        export_markdown(compare_result_minimal, path)
        content = (tmp_path / "out.md").read_text()
        assert "Recommended Strategy" not in content
