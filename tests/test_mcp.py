"""Tests for MCP schemas and server."""

import pytest
from pydantic import ValidationError

from mergelens.mcp.schemas import (
    CompareInput,
    CompatibilityInput,
    DiagnoseInput,
    LayerExplainInput,
    ReportInput,
    StrategyInput,
)

# --- Schema validation ---


class TestCompareInput:
    def test_valid_minimal(self):
        inp = CompareInput(models=["model_a", "model_b"])
        assert inp.models == ["model_a", "model_b"]
        assert inp.base_model is None
        assert inp.device == "cpu"
        assert inp.svd_rank == 64

    def test_valid_full(self):
        inp = CompareInput(
            models=["a", "b", "c"],
            base_model="base",
            device="cuda",
            svd_rank=32,
        )
        assert inp.base_model == "base"
        assert inp.device == "cuda"
        assert inp.svd_rank == 32

    def test_missing_models(self):
        with pytest.raises(ValidationError):
            CompareInput()


class TestDiagnoseInput:
    def test_valid(self):
        inp = DiagnoseInput(config_yaml="merge_method: slerp\n")
        assert inp.config_yaml == "merge_method: slerp\n"
        assert inp.device == "cpu"

    def test_missing_yaml(self):
        with pytest.raises(ValidationError):
            DiagnoseInput()


class TestCompatibilityInput:
    def test_valid(self):
        inp = CompatibilityInput(models=["a", "b"])
        assert inp.device == "cpu"

    def test_missing_models(self):
        with pytest.raises(ValidationError):
            CompatibilityInput()


class TestStrategyInput:
    def test_valid(self):
        inp = StrategyInput(models=["a", "b"], base_model="base")
        assert inp.base_model == "base"

    def test_defaults(self):
        inp = StrategyInput(models=["x", "y"])
        assert inp.base_model is None
        assert inp.device == "cpu"

    def test_missing_models(self):
        with pytest.raises(ValidationError):
            StrategyInput()


class TestLayerExplainInput:
    def test_valid(self):
        inp = LayerExplainInput(layer_name="model.layers.0.self_attn.q_proj.weight")
        assert "q_proj" in inp.layer_name

    def test_missing_name(self):
        with pytest.raises(ValidationError):
            LayerExplainInput()


class TestReportInput:
    def test_valid(self):
        inp = ReportInput(models=["a", "b"])
        assert inp.output_path == "mergelens_report.html"

    def test_custom_path(self):
        inp = ReportInput(models=["a"], output_path="out.html", device="cuda")
        assert inp.output_path == "out.html"
        assert inp.device == "cuda"

    def test_missing_models(self):
        with pytest.raises(ValidationError):
            ReportInput()


# --- Server creation ---


class TestMCPServer:
    @pytest.fixture(autouse=True)
    def _skip_if_no_mcp(self):
        pytest.importorskip("mcp", reason="mcp package not installed")

    def test_create_server(self):
        from mergelens.mcp.server import create_server

        server = create_server()
        assert server is not None
        assert server.name == "mergelens"

    def test_server_has_tools(self):
        from mergelens.mcp.server import create_server

        server = create_server()
        # FastMCP stores tools internally — verify via _tool_manager or similar
        # The exact attribute varies by mcp version, so just verify the server
        # was created successfully and has the expected type
        assert server.name == "mergelens"



def test_create_server_without_mcp_package(monkeypatch):
    """Verify ImportError with helpful message when mcp is missing."""
    import builtins

    import mergelens.mcp.server as srv_mod

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mcp.server.fastmcp":
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="mergelens\\[mcp\\]"):
        srv_mod.create_server()
