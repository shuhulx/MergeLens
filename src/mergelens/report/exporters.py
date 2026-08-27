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
    metric_names = (
        "cosine_similarity",
        "l2_distance",
        "weight_distribution_divergence",
        "spectral_overlap",
        "effective_rank_ratio",
        "sign_disagreement_rate",
        "tsv_interference",
        "task_vector_energy",
        "cka_similarity",
    )
    headers = [
        "evidence_scope",
        "comparison_id",
        "candidate_set_id",
        "reference_model",
        "candidate_model",
        "candidate_models",
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
    for metric in metric_names:
        headers.extend([f"{metric}_status", f"{metric}_reason"])
    writer.writerow(headers)
    for row in result.tensor_metrics:
        values = [
            "pair_tensor",
            row.comparison_id,
            "",
            row.reference_model,
            row.candidate_model,
            "",
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
        values.extend(_metric_status_cells(row.metric_observations, metric_names))
        writer.writerow([_spreadsheet_safe(value) for value in values])
    for candidate_row in result.candidate_set_metrics:
        values = [
            "candidate_set_tensor",
            "",
            candidate_row.candidate_set_id,
            candidate_row.base_model,
            "",
            " | ".join(candidate_row.candidate_models),
            candidate_row.tensor_name,
            candidate_row.tensor_position,
            candidate_row.transformer_block,
            candidate_row.tensor_type.value,
            "x".join(str(value) for value in candidate_row.shape),
            candidate_row.parameter_count,
            None,
            None,
            None,
            None,
            None,
            candidate_row.sign_disagreement_rate,
            candidate_row.tsv_interference,
            None,
            None,
        ]
        values.extend(_metric_status_cells(candidate_row.metric_observations, metric_names))
        writer.writerow([_spreadsheet_safe(value) for value in values])
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


def _metric_status_cells(observations, metric_names: tuple[str, ...]) -> list[str]:
    cells: list[str] = []
    for metric in metric_names:
        observation = observations.get(metric)
        cells.extend(
            [
                observation.status.value if observation is not None else "",
                observation.reason or "" if observation is not None else "",
            ]
        )
    return cells


def _spreadsheet_safe(value):
    """Neutralize strings that spreadsheet programs can interpret as formulas."""

    if not isinstance(value, str):
        return value
    candidate = value.lstrip(" \t\r\n")
    if candidate.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value
