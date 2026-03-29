"""Layer-by-layer model comparison orchestrator."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
from rich.progress import track

from mergelens.compare.loader import ModelHandle, find_common_tensors, iter_aligned_tensors
from mergelens.compare.metrics import (
    centered_task_vector_energy,
    cosine_similarity,
    effective_rank_ratio,
    kl_divergence,
    l2_distance,
    merge_compatibility_index,
    sign_disagreement_rate,
    spectral_subspace_overlap,
    tsv_interference_score,
)
from mergelens.compare.strategy import recommend_strategy
from mergelens.models import (
    CompareResult,
    ConflictZone,
    LayerMetrics,
    Severity,
)
from mergelens.utils.cache import MetricCache
from mergelens.utils.tensor_ops import compute_task_vector


def compare_models(
    model_paths: list[str],
    base_model: str | None = None,
    device: str = "cpu",
    metrics: list[str] | None = None,
    svd_rank: int = 64,
    cache: MetricCache | None = None,
    show_progress: bool = True,
    include_strategy: bool = True,
    cka_scores: list[float] | None = None,
) -> CompareResult:
    """Compare two or more models layer-by-layer.

    Args:
        model_paths: Paths or HF repo IDs for models to compare.
        base_model: Optional base model for task vector computation.
            If not provided, first model is used as base.
        device: Torch device for computation ("cpu" or "cuda").
        metrics: List of metric names to compute (None = all available).
        svd_rank: Number of singular vectors for spectral metrics.
        cache: Optional metric cache instance.
        show_progress: Show rich progress bar.
        include_strategy: Whether to include merge strategy recommendation.
        cka_scores: Optional pre-computed CKA similarity scores per layer.
            Use activations.cka.compare_activations_cka() to obtain these
            and pass the resulting per-layer scores here so that the MCI
            incorporates the activation-level similarity alongside weight
            metrics.  Values must be in [0, 1].

    Returns:
        CompareResult with all metrics, conflict zones, MCI, and optional strategy.
    """
    if len(model_paths) < 2:
        raise ValueError("Need at least 2 models to compare.")

    handles = [ModelHandle(p, device=device) for p in model_paths]
    base_handle = ModelHandle(base_model, device=device) if base_model else handles[0]

    all_handles = handles if base_model is None else [base_handle, *handles]
    common_names = find_common_tensors(all_handles)

    # Warn once if sign_disagreement_rate / tsv_interference will be skipped.
    # These metrics require at least 2 task vectors (model_a - base, model_b - base),
    # so they need either 3+ models or an explicit --base separate from the compared models.
    _two_model_no_base = base_model is None and len(model_paths) == 2
    if _two_model_no_base:
        logger.info(
            "sign_disagreement_rate and tsv_interference require an explicit --base model "
            "(separate from the models being compared) so that independent task vectors can be "
            "formed for each model. With only 2 models and no base these metrics will be None. "
            "Pass base_model= to enable them."
        )

    if not common_names:
        raise ValueError("No common tensor names found between models.")

    # Number of "non-base" models — these are the models being scored against
    # the base.  When no explicit base is given, handles[0] acts as base and
    # the remaining len(handles)-1 models are scored against it.  When an
    # explicit base is provided, all len(handles) models are scored.
    n_scored = len(handles) if base_model else len(handles) - 1
    n_scored = max(n_scored, 1)  # guard for edge case

    # Per-model metric accumulators.  Indexing: per_model_cosines[i] collects
    # cosine similarities for the i-th scored model vs. base across all layers.
    # Keeping metrics separate per model prevents mixing values from models with
    # different divergence profiles into a single flat list, which makes the
    # resulting MCI average statistically meaningless.
    per_model_cosines: list[list[float]] = [[] for _ in range(n_scored)]
    per_model_spectral: list[list[float]] = [[] for _ in range(n_scored)]
    per_model_rank_ratios: list[list[float]] = [[] for _ in range(n_scored)]
    per_model_energy: list[list[float]] = [[] for _ in range(n_scored)]

    # Multi-model metrics — these are inherently cross-model, so one list suffices.
    all_sign_disagree: list[float] = []
    all_tsv: list[float] = []

    all_layer_metrics: list[LayerMetrics] = []

    iterator = iter_aligned_tensors(all_handles, common_names)
    if show_progress:
        iterator = track(list(iterator), description="Comparing layers...")

    for name, layer_type, tensors in iterator:
        base_tensor = tensors[0]
        model_tensors = tensors[1:]

        for i, model_tensor in enumerate(model_tensors):
            cos_sim = cosine_similarity(base_tensor, model_tensor)
            l2_dist = l2_distance(base_tensor, model_tensor)

            lm = LayerMetrics(
                layer_name=name,
                layer_type=layer_type,
                shape=tuple(base_tensor.shape),
                cosine_similarity=cos_sim,
                l2_distance=l2_dist,
            )

            per_model_cosines[i].append(cos_sim)

            try:
                kl_div = kl_divergence(base_tensor, model_tensor)
                lm.kl_divergence = kl_div
            except Exception:
                logger.debug("kl_divergence computation failed for %s", name, exc_info=True)

            # Spectral overlap (only for 2D+ tensors with enough elements)
            if base_tensor.numel() > 128:
                try:
                    spec_overlap = spectral_subspace_overlap(base_tensor, model_tensor, k=svd_rank)
                    lm.spectral_overlap = spec_overlap
                    per_model_spectral[i].append(spec_overlap)
                except Exception:
                    logger.debug(
                        "spectral_subspace_overlap computation failed for %s", name, exc_info=True
                    )

                try:
                    rank_r = effective_rank_ratio(base_tensor, model_tensor)
                    lm.effective_rank_ratio = rank_r
                    per_model_rank_ratios[i].append(rank_r)
                except Exception:
                    logger.debug(
                        "effective_rank_ratio computation failed for %s", name, exc_info=True
                    )

            task_vec = compute_task_vector(model_tensor, base_tensor)

            if task_vec.numel() > 64:
                try:
                    energy = centered_task_vector_energy(task_vec, k=svd_rank)
                    lm.task_vector_energy = energy
                    per_model_energy[i].append(energy)
                except Exception:
                    logger.debug(
                        "centered_task_vector_energy computation failed for %s", name, exc_info=True
                    )

            all_layer_metrics.append(lm)

        # Multi-model task vector metrics (need 2+ task vectors)
        if len(model_tensors) >= 2:
            task_vecs = [compute_task_vector(mt, base_tensor) for mt in model_tensors]

            try:
                sign_dis = sign_disagreement_rate(task_vecs)
                all_sign_disagree.append(sign_dis)
                for lm in all_layer_metrics[-len(model_tensors) :]:
                    lm.sign_disagreement_rate = sign_dis
            except Exception:
                logger.debug(
                    "sign_disagreement_rate computation failed for %s", name, exc_info=True
                )

            try:
                tsv = tsv_interference_score(task_vecs, k=svd_rank)
                all_tsv.append(tsv)
                for lm in all_layer_metrics[-len(model_tensors) :]:
                    lm.tsv_interference = tsv
            except Exception:
                logger.debug(
                    "tsv_interference_score computation failed for %s", name, exc_info=True
                )

    # Compute one MCI per model (model vs. base), then average the MCI scores
    # across models.  This prevents metrics from different models being mixed
    # into a single flat list before averaging, which is statistically invalid
    # when models diverge from the base by different amounts.
    import numpy as _np

    per_model_mcis = []
    for i in range(n_scored):
        if not per_model_cosines[i]:
            continue
        per_model_mcis.append(
            merge_compatibility_index(
                cosine_sims=per_model_cosines[i],
                spectral_overlaps=per_model_spectral[i] or None,
                rank_ratios=per_model_rank_ratios[i] or None,
                sign_disagreements=all_sign_disagree or None,
                tsv_scores=all_tsv or None,
                energy_scores=per_model_energy[i] or None,
                # CKA scores are activation-based and model-agnostic (one
                # score per shared layer), so the same list is passed for
                # every per-model MCI.  Callers obtain these via
                # activations.cka.compare_activations_cka() and supply them
                # through the cka_scores= parameter of compare_models().
                cka_scores=cka_scores or None,
            )
        )

    if not per_model_mcis:
        from mergelens.models import MergeCompatibilityIndex as _MCI

        mci = _MCI(
            score=0.0,
            confidence=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            verdict="insufficient data",
            components={},
        )
    elif len(per_model_mcis) == 1:
        mci = per_model_mcis[0]
    else:
        # Average across models: use the most conservative (lowest) score as
        # the primary value but report the mean verdict for transparency.
        avg_score = float(_np.mean([m.score for m in per_model_mcis]))
        avg_conf = float(_np.mean([m.confidence for m in per_model_mcis]))
        avg_lower = float(_np.mean([m.ci_lower for m in per_model_mcis]))
        avg_upper = float(_np.mean([m.ci_upper for m in per_model_mcis]))
        # Merge component dicts by averaging shared keys
        all_keys = set().union(*(m.components.keys() for m in per_model_mcis))
        merged_components = {
            k: float(
                _np.mean([m.components[k] for m in per_model_mcis if k in m.components])
            )
            for k in all_keys
        }
        if avg_score >= 75:
            verdict = "highly compatible"
        elif avg_score >= 55:
            verdict = "compatible"
        elif avg_score >= 35:
            verdict = "risky"
        else:
            verdict = "incompatible"
        from mergelens.models import MergeCompatibilityIndex as _MCI

        mci = _MCI(
            score=round(avg_score, 1),
            confidence=round(avg_conf, 2),
            ci_lower=round(avg_lower, 1),
            ci_upper=round(avg_upper, 1),
            verdict=verdict,
            components=merged_components,
        )

    conflict_zones = _detect_conflict_zones(all_layer_metrics)

    result = CompareResult(
        models=[h.info for h in handles],
        layer_metrics=all_layer_metrics,
        conflict_zones=conflict_zones,
        mci=mci,
    )

    if include_strategy:
        result.strategy = recommend_strategy(result)

    return result


def _detect_conflict_zones(
    layer_metrics: list[LayerMetrics],
    cos_threshold: float = 0.80,
    min_zone_size: int = 2,
) -> list[ConflictZone]:
    """Detect contiguous groups of layers with high disagreement."""
    zones: list[ConflictZone] = []
    current_zone_layers: list[tuple[int, LayerMetrics]] = []

    for i, lm in enumerate(layer_metrics):
        if lm.cosine_similarity < cos_threshold:
            current_zone_layers.append((i, lm))
        else:
            if len(current_zone_layers) >= min_zone_size:
                zones.append(_build_zone(current_zone_layers))
            current_zone_layers = []

    # Don't forget the last zone
    if len(current_zone_layers) >= min_zone_size:
        zones.append(_build_zone(current_zone_layers))

    return zones


def _build_zone(layers: list[tuple[int, LayerMetrics]]) -> ConflictZone:
    """Build a ConflictZone from a list of (index, LayerMetrics)."""
    indices = [i for i, _ in layers]
    metrics = [lm for _, lm in layers]
    avg_cos = sum(lm.cosine_similarity for lm in metrics) / len(metrics)

    sign_disagrees = [
        lm.sign_disagreement_rate for lm in metrics if lm.sign_disagreement_rate is not None
    ]
    avg_sign = sum(sign_disagrees) / len(sign_disagrees) if sign_disagrees else None

    if avg_cos < 0.5:
        severity = Severity.CRITICAL
    elif avg_cos < 0.7:
        severity = Severity.HIGH
    elif avg_cos < 0.8:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    if severity == Severity.CRITICAL:
        rec = "Critical conflict zone. Consider excluding these layers from merge or using passthrough."
    elif severity == Severity.HIGH:
        rec = "High conflict. Use TIES merging with sign resolution or reduce merge weight for these layers."
    elif severity == Severity.MEDIUM:
        rec = "Moderate conflict. SLERP with reduced t value recommended for these layers."
    else:
        rec = "Minor conflict. Standard merge parameters should work."

    return ConflictZone(
        start_layer=indices[0],
        end_layer=indices[-1],
        layer_names=[lm.layer_name for lm in metrics],
        severity=severity,
        avg_cosine_sim=round(avg_cos, 4),
        avg_sign_disagreement=round(avg_sign, 4) if avg_sign is not None else None,
        recommendation=rec,
    )
