"""MergeLens MCP server for evidence-aware static checkpoint inspection.

Tools:
    compare_models — Tensor comparison, coverage, and heuristic assessment
    diagnose_merge — Analyze a MergeKit config
    get_conflict_zones — Heuristic tensor inspection priorities
    suggest_strategy — Rule-based MergeKit starting configuration
    generate_report — Create HTML report
    explain_layer — Explain a layer's role in merging
    get_compatibility_score — Unvalidated static-risk heuristic
"""

from typing import Any


def create_server():
    """Create and configure the MergeLens MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "MCP server requires the 'mcp' package. Install with: pip install mergelens[mcp]"
        )

    try:
        mcp = FastMCP("mergelens", description="Pre-merge diagnostics for LLM model merging")
    except TypeError:
        # Older mcp versions don't support description kwarg
        mcp = FastMCP("mergelens")

    @mcp.tool()
    def compare_models(
        models: list[str],
        base_model: str | None = None,
        device: str = "cpu",
        svd_rank: int = 64,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two or more checkpoints with explicit coverage and availability.

        Returns exact tensor coverage, attributable raw metrics, unavailable-signal
        reasons, and an explicitly unvalidated heuristic summary.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            base_model=base_model,
            device=device,
            svd_rank=svd_rank,
            metrics=metrics,
            show_progress=False,
        )
        return result.model_dump()

    @mcp.tool()
    def diagnose_merge(config_yaml: str, device: str = "cpu") -> dict[str, Any]:
        """Diagnose a MergeKit YAML config for potential issues before merging.

        Parses the config, loads referenced models, and identifies interference.
        """
        import tempfile
        from pathlib import Path

        from mergelens.diagnose import diagnose_config

        # Write YAML to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            result = diagnose_config(config_path, device=device)
            return result.model_dump()
        finally:
            Path(config_path).unlink(missing_ok=True)

    @mcp.tool()
    def get_conflict_zones(
        models: list[str],
        base_model: str | None = None,
        device: str = "cpu",
    ) -> list[dict[str, Any]]:
        """Return heuristic ordered-tensor inspection priorities.

        These regions are not transformer layer ranges or method prescriptions.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            base_model=base_model,
            device=device,
            show_progress=False,
            include_strategy=False,
        )
        return [z.model_dump() for z in result.conflict_zones]

    @mcp.tool()
    def suggest_strategy(
        models: list[str],
        base_model: str | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Return a rule-based MergeKit starting configuration.

        The rule and generated parameters are hypotheses for post-merge testing.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            base_model=base_model,
            device=device,
            show_progress=False,
        )
        if result.strategy:
            return result.strategy.model_dump()
        return {"error": "Could not generate strategy recommendation."}

    @mcp.tool()
    def generate_report(
        models: list[str],
        base_model: str | None = None,
        output_path: str = "mergelens_report.html",
        device: str = "cpu",
    ) -> str:
        """Generate an offline interactive HTML diagnostic report.

        Plotly JavaScript is embedded in the output file.
        """
        from pathlib import Path

        from mergelens.compare.analyzer import compare_models as _compare
        from mergelens.report.generator import generate_report as _report

        # Security: prevent path traversal from MCP clients.
        # Path.is_relative_to() was added in Python 3.9 but has a known
        # edge-case crash on Python 3.10 when the compared paths have
        # incompatible roots on some platforms.  Use relative_to() with a
        # try/except — it works identically on all Python 3.x versions.
        resolved = Path(output_path).resolve()
        cwd = Path.cwd().resolve()
        try:
            resolved.relative_to(cwd)
        except ValueError:
            raise ValueError(
                f"output_path must be within the current working directory. Resolved to: {resolved}"
            )

        result = _compare(
            model_paths=models,
            base_model=base_model,
            device=device,
            show_progress=False,
        )
        path = _report(compare_result=result, output_path=output_path)
        return f"Report saved to {path}"

    @mcp.tool()
    def explain_layer(layer_name: str) -> str:
        """Explain what a transformer layer does and its role in merging.

        Helps users understand which layers matter most for merge quality.
        """
        from mergelens.compare.loader import classify_layer
        from mergelens.models import LayerType

        layer_type = classify_layer(layer_name)

        explanations = {
            LayerType.ATTENTION_Q: "Query projection used to form self-attention query vectors.",
            LayerType.ATTENTION_K: "Key projection used to form self-attention key vectors.",
            LayerType.ATTENTION_V: "Value projection used to form self-attention value vectors.",
            LayerType.ATTENTION_O: "Output projection that maps combined attention-head output back to the hidden width.",
            LayerType.MLP_GATE: "Gate projection used by gated feed-forward blocks such as SwiGLU.",
            LayerType.MLP_UP: "Feed-forward projection from hidden width to the intermediate width.",
            LayerType.MLP_DOWN: "Feed-forward projection from intermediate width back to hidden width.",
            LayerType.NORM: "Normalization scale or bias tensor, depending on the architecture.",
            LayerType.EMBEDDING: "Token embedding tensor. Shape compatibility is reported separately because vocabulary or hidden-width differences can prevent alignment.",
            LayerType.LM_HEAD: "Output projection from hidden states to vocabulary logits; it may be tied to token embeddings.",
            LayerType.OTHER: "Tensor role was not recognized from its name; inspect the checkpoint architecture and full tensor name.",
        }

        explanation = explanations.get(layer_type, explanations[LayerType.OTHER])
        return f"**{layer_name}** (type: {layer_type.value})\n\n{explanation}"

    @mcp.tool()
    def get_compatibility_score(
        models: list[str],
        base_model: str | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Get an unvalidated static-risk heuristic for two or more models.

        The result can be suppressed for unsupported comparisons and never
        establishes downstream merged-model quality.
        """
        from mergelens.compare.analyzer import compare_models as _compare

        result = _compare(
            model_paths=models,
            base_model=base_model,
            device=device,
            show_progress=False,
            include_strategy=False,
        )
        return {
            "assessment": result.mci.model_dump(),
            "coverage": [item.model_dump() for item in result.coverage],
            "metric_availability": [item.model_dump() for item in result.metric_availability],
        }

    return mcp
