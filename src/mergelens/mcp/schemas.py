"""MCP tool input/output schemas.

Reference schemas for MCP tool inputs. These are not used by the server directly
(FastMCP infers schemas from function signatures), but are used in tests and
serve as documentation for the expected input shapes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompareInput(BaseModel):
    """Input for compare_models tool."""

    models: list[str] = Field(..., description="Model paths or HF repo IDs (2+)")
    base_model: str | None = Field(None, description="Base model for task vectors")
    device: str = Field("cpu", description="Torch device")
    svd_rank: int = Field(64, description="SVD rank for spectral metrics")


class DiagnoseInput(BaseModel):
    """Input for diagnose_merge tool."""

    config_yaml: str = Field(..., description="MergeKit YAML config content")
    device: str = Field("cpu", description="Torch device")


class CompatibilityInput(BaseModel):
    """Input for get_compatibility_score tool."""

    models: list[str] = Field(..., description="Model paths or HF repo IDs (2+)")
    device: str = Field("cpu", description="Torch device")


class StrategyInput(BaseModel):
    """Input for suggest_strategy tool."""

    models: list[str] = Field(..., description="Model paths or HF repo IDs (2+)")
    base_model: str | None = Field(None, description="Base model")
    device: str = Field("cpu", description="Torch device")


class LayerExplainInput(BaseModel):
    """Input for explain_layer tool."""

    layer_name: str = Field(..., description="Full layer name to explain")


class ReportInput(BaseModel):
    """Input for generate_report tool."""

    models: list[str] = Field(..., description="Model paths or HF repo IDs")
    output_path: str = Field("mergelens_report.html", description="Output file path")
    device: str = Field("cpu", description="Torch device")
