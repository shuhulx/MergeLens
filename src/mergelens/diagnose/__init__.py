"""Merge config diagnosis module."""

import logging

from mergelens.diagnose.attribution import compute_attribution
from mergelens.diagnose.config_parser import parse_mergekit_config
from mergelens.diagnose.interference import compute_interference

logger = logging.getLogger(__name__)


def diagnose_config(config_path: str, device: str = "cpu"):
    """Diagnose a MergeKit config for potential issues.

    Main entry point: parse config → load models → compute interference → generate recommendations.
    """
    from pathlib import Path

    from mergelens.compare.loader import ModelHandle
    from mergelens.models import ConflictZone, DiagnoseResult, Severity

    config = parse_mergekit_config(Path(config_path).read_text())

    # Load model handles
    handles = {}
    for model_path in config.models:
        try:
            handles[model_path] = ModelHandle(model_path, device=device)
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", model_path, exc)

    if len(handles) < 2:
        return DiagnoseResult(
            config=config,
            interference_scores=[],
            overall_interference=0.0,
            recommendations=["Could not load enough models to diagnose. Check model paths."],
        )

    # Compute interference
    handle_list = list(handles.values())
    interference_scores = compute_interference(
        handle_list, base_model=config.base_model, device=device
    )

    overall = sum(s.score for s in interference_scores) / max(len(interference_scores), 1)

    # Generate recommendations
    recommendations = _generate_recommendations(config, interference_scores, overall)

    return DiagnoseResult(
        config=config,
        interference_scores=interference_scores,
        overall_interference=round(overall, 4),
        recommendations=recommendations,
    )


def _generate_recommendations(config, interference_scores, overall):
    """Generate actionable recommendations based on diagnosis."""
    recs = []

    if overall > 0.5:
        recs.append(
            f"High overall interference ({overall:.2f}). Consider using TIES or DARE methods "
            "which handle conflicts better than linear/SLERP."
        )

    # Find hotspot layers
    hotspots = [s for s in interference_scores if s.score > 0.7]
    if hotspots:
        layer_names = [s.layer_name for s in hotspots[:5]]
        recs.append(
            f"High interference in {len(hotspots)} layers. "
            f"Hotspots: {', '.join(layer_names)}. "
            "Consider per-layer weight adjustments or passthrough for these layers."
        )

    from mergelens.models import MergeMethod

    if config.merge_method == MergeMethod.SLERP and overall > 0.3:
        recs.append(
            "SLERP may not handle the level of conflict detected. "
            "Consider TIES (for sign conflicts) or DARE (for concentrated knowledge)."
        )

    if not recs:
        recs.append("Config looks good. No major issues detected.")

    return recs


__all__ = [
    "compute_attribution",
    "compute_interference",
    "diagnose_config",
    "parse_mergekit_config",
]
