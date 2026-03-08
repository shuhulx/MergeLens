"""Export results in various formats."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from mergelens.models import CompareResult


def export_json(result: CompareResult, path: str) -> str:
    """Export results as JSON."""
    Path(path).write_text(result.model_dump_json(indent=2))
    return path


def export_csv(result: CompareResult, path: str) -> str:
    """Export layer metrics as CSV."""
    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "layer_name",
            "layer_type",
            "cosine_similarity",
            "l2_distance",
            "spectral_overlap",
            "effective_rank_ratio",
            "sign_disagreement_rate",
            "tsv_interference",
            "task_vector_energy",
            "cka_similarity",
        ]
    )

    for m in result.layer_metrics:
        writer.writerow(
            [
                m.layer_name,
                m.layer_type.value,
                m.cosine_similarity,
                m.l2_distance,
                m.spectral_overlap,
                m.effective_rank_ratio,
                m.sign_disagreement_rate,
                m.tsv_interference,
                m.task_vector_energy,
                m.cka_similarity,
            ]
        )

    Path(path).write_text(output.getvalue())
    return path


def export_markdown(result: CompareResult, path: str) -> str:
    """Export results as Markdown."""
    lines = []
    lines.append("# MergeLens Report\n")

    # MCI
    mci = result.mci
    lines.append(f"## Merge Compatibility Index: {mci.score}/100\n")
    lines.append(f"**Verdict:** {mci.verdict}")
    lines.append(
        f"**Confidence:** {mci.confidence:.0%} (Range: {mci.ci_lower:.0f}-{mci.ci_upper:.0f})\n"
    )

    # Models
    lines.append("## Models\n")
    for m in result.models:
        lines.append(f"- **{m.name}**: {m.path_or_repo}")
    lines.append("")

    # Conflict zones
    if result.conflict_zones:
        lines.append("## Conflict Zones\n")
        lines.append("| Zone | Layers | Severity | Avg Cos Sim | Recommendation |")
        lines.append("|------|--------|----------|-------------|----------------|")
        for i, z in enumerate(result.conflict_zones):
            lines.append(
                f"| {i + 1} | {z.start_layer}-{z.end_layer} | {z.severity.value} | {z.avg_cosine_sim:.4f} | {z.recommendation} |"
            )
        lines.append("")

    # Strategy
    if result.strategy:
        lines.append("## Recommended Strategy\n")
        lines.append(
            f"**Method:** {result.strategy.method.value} ({result.strategy.confidence:.0%} confidence)\n"
        )
        lines.append(result.strategy.reasoning)
        lines.append(f"\n```yaml\n{result.strategy.mergekit_yaml}```\n")

    Path(path).write_text("\n".join(lines))
    return path
