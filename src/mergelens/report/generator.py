"""Offline HTML reporting with explicit comparison identity and evidence limits."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from mergelens.models import CompareResult, DiagnoseResult, TensorMetrics

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(
    compare_result: CompareResult | None = None,
    diagnose_result: DiagnoseResult | None = None,
    output_path: str = "mergelens_report.html",
    title: str = "MergeLens Report",
) -> str:
    """Generate one offline HTML file with embedded Plotly JavaScript.

    Report dependencies are imported lazily so the core package remains usable
    without the ``report`` extra.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from plotly.offline import get_plotlyjs  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "HTML reports require optional dependencies. Install with: pip install mergelens[report]"
        ) from exc

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    charts: dict[str, dict[str, Any]] = {}
    if compare_result is not None:
        charts = {
            "heuristic": _build_heuristic_gauge(compare_result),
            "heatmap": _build_similarity_heatmap(compare_result),
            "spectral": _build_spectral_chart(compare_result),
            "tensor_metrics": _build_tensor_metrics_chart(compare_result),
            "conflicts": _build_conflict_chart(compare_result),
        }

    rendered = env.get_template("base.html.j2").render(
        title=title,
        compare=compare_result,
        diagnose=diagnose_result,
        charts=charts,
        plotly_js=get_plotlyjs(),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return str(destination)


def _build_heuristic_gauge(result: CompareResult) -> dict[str, Any]:
    """Build a gauge without statistical-confidence terminology."""
    assessment = result.mci
    score = assessment.score
    if score is None:
        return {
            "data": [],
            "layout": {
                "title": "Static-risk heuristic suppressed: insufficient structural evidence",
                "height": 180,
            },
        }
    band = ""
    if assessment.heuristic_band_lower is not None and assessment.heuristic_band_upper is not None:
        band = (
            f"; sensitivity band {assessment.heuristic_band_lower:.0f}-"
            f"{assessment.heuristic_band_upper:.0f}"
        )
    return {
        "data": [
            {
                "type": "indicator",
                "mode": "gauge+number",
                "value": score,
                "title": {"text": "Unvalidated static-risk heuristic"},
                "gauge": {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": _score_color(score)},
                    "steps": [
                        {"range": [0, 55], "color": "#ffebee"},
                        {"range": [55, 75], "color": "#fff3e0"},
                        {"range": [75, 100], "color": "#e8f5e9"},
                    ],
                },
            }
        ],
        "layout": {
            "height": 300,
            "margin": {"t": 50, "b": 30, "l": 30, "r": 30},
            "annotations": [
                {
                    "text": f"{assessment.risk_tier}{band}; {assessment.validation_status}",
                    "x": 0.5,
                    "y": 0,
                    "showarrow": False,
                    "font": {"size": 13},
                }
            ],
        },
    }


def _rows_by_comparison(result: CompareResult) -> dict[str, list[TensorMetrics]]:
    grouped: dict[str, list[TensorMetrics]] = defaultdict(list)
    for row in result.tensor_metrics:
        grouped[row.comparison_id].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row.tensor_position, row.tensor_name))
    return dict(sorted(grouped.items()))


def _pair_label(rows: list[TensorMetrics], comparison_id: str) -> str:
    if not rows:
        return comparison_id
    return f"{rows[0].reference_model} vs {rows[0].candidate_model}"


def _build_similarity_heatmap(result: CompareResult) -> dict[str, Any]:
    """Build a pair-attributable heatmap without contiguous-row assumptions."""
    grouped = _rows_by_comparison(result)
    positions: dict[str, int] = {}
    for rows in grouped.values():
        for row in rows:
            positions[row.tensor_name] = min(
                positions.get(row.tensor_name, row.tensor_position), row.tensor_position
            )
    tensor_names = sorted(positions, key=lambda name: (positions[name], name))
    matrix: list[list[float | None]] = []
    pair_labels: list[str] = []
    for comparison_id, rows in grouped.items():
        values = {row.tensor_name: row.cosine_similarity for row in rows}
        matrix.append([values.get(name) for name in tensor_names])
        pair_labels.append(_pair_label(rows, comparison_id))
    return {
        "data": [
            {
                "type": "heatmap",
                "z": matrix,
                "x": tensor_names,
                "y": pair_labels,
                "colorscale": "RdYlGn",
                "zmin": -1,
                "zmax": 1,
                "colorbar": {"title": "Cosine"},
                "hoverongaps": False,
            }
        ],
        "layout": {
            "title": "Exact-shape tensor cosine similarity by checkpoint pair",
            "height": max(260, 70 * len(pair_labels) + 130),
            "margin": {"t": 50, "b": 130, "l": 180, "r": 30},
            "xaxis": {"tickangle": -45, "tickfont": {"size": 8}},
        },
    }


def _metric_traces(
    result: CompareResult,
    metrics: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for comparison_id, rows in _rows_by_comparison(result).items():
        for attribute, label in metrics:
            values = [getattr(row, attribute) for row in rows]
            if not any(value is not None for value in values):
                continue
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": [row.tensor_position for row in rows],
                    "y": values,
                    "text": [row.tensor_name for row in rows],
                    "name": f"{_pair_label(rows, comparison_id)} - {label}",
                    "connectgaps": False,
                }
            )
    return traces


def _build_spectral_chart(result: CompareResult) -> dict[str, Any]:
    """Build spectral traces grouped explicitly by checkpoint pair."""
    return {
        "data": _metric_traces(
            result,
            (
                ("spectral_overlap", "spectral overlap"),
                ("effective_rank_ratio", "effective-rank ratio"),
                ("task_vector_energy", "task-vector energy"),
            ),
        ),
        "layout": {
            "title": "Bounded spectral diagnostics by ordered tensor position",
            "xaxis": {"title": "Ordered tensor position"},
            "yaxis": {"title": "Measured value", "range": [0, 1.05]},
            "height": 420,
        },
    }


def _build_tensor_metrics_chart(result: CompareResult) -> dict[str, Any]:
    """Build pair L2 and separately attributed candidate-set task-vector traces."""
    traces = _metric_traces(result, (("l2_distance", "normalized L2 distance"),))
    for attribute, label in (
        ("sign_disagreement_rate", "candidate-set sign disagreement"),
        ("tsv_interference", "candidate-set TSV interference"),
    ):
        values = [getattr(row, attribute) for row in result.candidate_set_metrics]
        if any(value is not None for value in values):
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": [row.tensor_position for row in result.candidate_set_metrics],
                    "y": values,
                    "text": [row.tensor_name for row in result.candidate_set_metrics],
                    "name": f"candidate_set_0 - {label}",
                    "connectgaps": False,
                }
            )
    return {
        "data": traces,
        "layout": {
            "title": "Tensor diagnostics by checkpoint pair",
            "xaxis": {"title": "Ordered tensor position"},
            "yaxis": {"title": "Measured value"},
            "height": 420,
        },
    }


_build_layer_metrics_chart = _build_tensor_metrics_chart


def _build_conflict_chart(result: CompareResult) -> dict[str, Any]:
    """Build inspection-priority bars with pair and tensor identity."""
    regions = result.tensor_conflict_regions
    if not regions:
        return {"data": [], "layout": {"title": "No heuristic inspection regions identified"}}
    labels = [
        f"{region.comparison_id}: {region.start_tensor_position}-{region.end_tensor_position}"
        for region in regions
    ]
    return {
        "data": [
            {
                "type": "bar",
                "x": labels,
                "y": [region.avg_cosine_similarity for region in regions],
                "text": ["<br>".join(region.tensor_names) for region in regions],
                "marker": {
                    "color": [_severity_color_hex(region.severity.value) for region in regions]
                },
            }
        ],
        "layout": {
            "title": "Heuristic tensor inspection priorities",
            "yaxis": {"title": "Mean cosine similarity", "range": [-1, 1]},
            "height": 340,
        },
    }


def _score_color(score: float) -> str:
    if score >= 75:
        return "#4caf50"
    if score >= 55:
        return "#ff9800"
    return "#f44336"


def _severity_color_hex(severity: str) -> str:
    return {
        "low": "#4caf50",
        "medium": "#ff9800",
        "high": "#f44336",
        "critical": "#b71c1c",
    }.get(severity, "#9e9e9e")
