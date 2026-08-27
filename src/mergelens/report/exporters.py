"""Machine-readable and reviewer-readable result exporters."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from mergelens.models import CompareResult


def export_json(result: CompareResult, path: str) -> str:
    """Export the complete evidence model as JSON."""
    Path(path).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def export_csv(result: CompareResult, path: str) -> str:
    """Export attributable per-tensor measurements as CSV."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "comparison_id",
            "reference_model",
            "candidate_model",
            "tensor_name",
            "tensor_position",
            "transformer_block",
            "tensor_type",
            "shape",
            "parameter_count",
            "cosine_similarity",
            "l2_distance",
            "weight_distribution_divergence",
            "spectral_overlap",
            "effective_rank_ratio",
            "sign_disagreement_rate",
            "tsv_interference",
            "task_vector_energy",
            "cka_similarity",
        ]
    )
    for row in result.tensor_metrics:
        writer.writerow(
            [
                row.comparison_id,
                row.reference_model,
                row.candidate_model,
                row.tensor_name,
                row.tensor_position,
                row.transformer_block,
                row.tensor_type.value,
                "x".join(str(value) for value in row.shape),
                row.parameter_count,
                row.cosine_similarity,
                row.l2_distance,
                row.weight_distribution_divergence,
                row.spectral_overlap,
                row.effective_rank_ratio,
                row.sign_disagreement_rate,
                row.tsv_interference,
                row.task_vector_energy,
                row.cka_similarity,
            ]
        )
    Path(path).write_text(output.getvalue(), encoding="utf-8")
    return path


def export_markdown(result: CompareResult, path: str) -> str:
    """Export a compact report with explicit evidence and heuristic status."""
    assessment = result.mci
    score = "suppressed" if assessment.score is None else f"{assessment.score:.1f}/100"
    lines = [
        "# MergeLens static-checkpoint report",
        "",
        f"- Unvalidated heuristic: {score}",
        f"- Static-risk tier: `{assessment.risk_tier}`",
        f"- Evidence coverage: {assessment.evidence_coverage:.0%}",
        f"- Validation status: `{assessment.validation_status}`",
        "- This report does not establish downstream merged-model quality.",
        "",
        "## Coverage",
        "",
        "| Pair | Exact tensors | Common parameters | Reference coverage | Candidate coverage | Scoring |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for coverage in result.coverage:
        lines.append(
            f"| {coverage.reference_model} vs {coverage.candidate_model} | "
            f"{coverage.exact_shape_compatible_tensor_count} | {coverage.common_parameter_count} | "
            f"{_percentage(coverage.parameter_coverage_reference)} | "
            f"{_percentage(coverage.parameter_coverage_candidate)} | "
            f"{'supported' if coverage.scoring_supported else 'suppressed'} |"
        )

    lines.extend(["", "## Metric availability", ""])
    for item in result.metric_availability:
        suffix = f" - {item.reason}" if item.reason else ""
        lines.append(f"- `{item.metric}`: `{item.status.value}`{suffix}")

    if result.tensor_conflict_regions:
        lines.extend(["", "## Heuristic tensor inspection priorities", ""])
        for region in result.tensor_conflict_regions:
            lines.append(
                f"- `{region.comparison_id}` positions "
                f"{region.start_tensor_position}-{region.end_tensor_position}: "
                f"{region.heuristic_inspection_note}"
            )

    if result.strategy is not None:
        lines.extend(
            [
                "",
                "## Rule-based MergeKit starting configuration",
                "",
                f"Method: `{result.strategy.method.value}`; status: "
                f"`{result.strategy.config_status}`.",
                "",
                result.strategy.reasoning,
                "",
                "```yaml",
                result.strategy.mergekit_yaml.rstrip(),
                "```",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _percentage(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1%}"
