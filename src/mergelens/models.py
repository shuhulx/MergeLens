"""MergeLens data models — the API contract for all modules."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MergeMethod(str, Enum):
    """Supported merge methods."""

    SLERP = "slerp"
    TIES = "ties"
    DARE_TIES = "dare_ties"
    DARE_LINEAR = "dare_linear"
    LINEAR = "linear"
    PASSTHROUGH = "passthrough"
    DELLA = "della"
    DELLA_LINEAR = "della_linear"
    MODEL_STOCK = "model_stock"
    BREADCRUMBS = "breadcrumbs"
    BREADCRUMBS_TIES = "breadcrumbs_ties"


class Severity(str, Enum):
    """Conflict severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LayerType(str, Enum):
    """Types of transformer layers."""

    ATTENTION_Q = "attn_q"
    ATTENTION_K = "attn_k"
    ATTENTION_V = "attn_v"
    ATTENTION_O = "attn_o"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    NORM = "norm"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    OTHER = "other"


# ── Per-Layer Results ─────────────────────────────────────────────


class LayerMetrics(BaseModel):
    """Metrics for a single layer comparison between two models."""

    layer_name: str
    layer_type: LayerType = LayerType.OTHER
    shape: tuple[int, ...] = ()
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    l2_distance: float = Field(ge=0.0)
    kl_divergence: float | None = Field(default=None, ge=0.0)
    # Optional — some require task vectors or activations
    spectral_overlap: float | None = Field(default=None, ge=0.0, le=1.0)
    effective_rank_ratio: float | None = Field(default=None, ge=0.0)
    sign_disagreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    tsv_interference: float | None = Field(default=None, ge=0.0)
    task_vector_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    cka_similarity: float | None = Field(default=None, ge=0.0, le=1.0)


# ── Conflict Zones ────────────────────────────────────────────────


class ConflictZone(BaseModel):
    """A contiguous group of layers with high disagreement."""

    start_layer: int
    end_layer: int
    layer_names: list[str]
    severity: Severity
    avg_cosine_sim: float
    avg_sign_disagreement: float | None = None
    recommendation: str


# ── Merge Compatibility Index ─────────────────────────────────────


class MergeCompatibilityIndex(BaseModel):
    """Composite 0-100 score indicating merge compatibility."""

    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    ci_lower: float = Field(ge=0.0, le=100.0)
    ci_upper: float = Field(ge=0.0, le=100.0)
    verdict: str  # "highly compatible", "compatible", "risky", "incompatible"
    components: dict[str, float] = Field(default_factory=dict)


# ── Strategy Recommendation ───────────────────────────────────────


class StrategyRecommendation(BaseModel):
    """Recommended merge strategy with generated config."""

    method: MergeMethod
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    mergekit_yaml: str  # Ready-to-use MergeKit YAML
    warnings: list[str] = Field(default_factory=list)
    per_layer_overrides: dict[str, Any] = Field(default_factory=dict)


# ── Compare Result ────────────────────────────────────────────────


class ModelInfo(BaseModel):
    """Metadata about a model being compared."""

    name: str
    path_or_repo: str
    num_parameters: int | None = None
    architecture: str | None = None
    num_layers: int | None = None


class CompareResult(BaseModel):
    """Full output of compare.models()."""

    models: list[ModelInfo]
    layer_metrics: list[LayerMetrics]
    conflict_zones: list[ConflictZone]
    mci: MergeCompatibilityIndex
    strategy: StrategyRecommendation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Diagnose Result ───────────────────────────────────────────────


class MergeConfig(BaseModel):
    """Parsed MergeKit configuration."""

    merge_method: MergeMethod
    base_model: str | None = None
    models: list[str]
    parameters: dict[str, Any] = Field(default_factory=dict)
    slices: list[dict[str, Any]] | None = None
    raw_yaml: str = ""


class InterferenceScore(BaseModel):
    """Interference measurement for a layer."""

    layer_name: str
    score: float = Field(ge=0.0, le=1.0)
    source_contributions: dict[str, float] = Field(default_factory=dict)


class DiagnoseResult(BaseModel):
    """Full output of diagnose.from_config()."""

    config: MergeConfig
    interference_scores: list[InterferenceScore]
    attribution_map: dict[str, dict[str, float]] = Field(default_factory=dict)
    conflict_zones: list[ConflictZone] = Field(default_factory=list)
    overall_interference: float = Field(ge=0.0, le=1.0)
    recommendations: list[str] = Field(default_factory=list)


# ── Audit Result ──────────────────────────────────────────────────


class ProbeResult(BaseModel):
    """Result of a single probe evaluation."""

    probe_id: str
    category: str
    prompt: str
    response: str
    score: float = Field(ge=0.0, le=1.0)
    judge_reasoning: str | None = None


class CapabilityScore(BaseModel):
    """Aggregated score for a capability category."""

    category: str
    base_score: float = Field(ge=0.0, le=1.0)
    merged_score: float = Field(ge=0.0, le=1.0)
    retention: float = Field(ge=0.0)  # merged/base ratio
    num_probes: int


class AuditResult(BaseModel):
    """Full output of audit.run()."""

    base_model: str
    merged_model: str
    capability_scores: list[CapabilityScore]
    probe_results: list[ProbeResult] = Field(default_factory=list)
    overall_retention: float = Field(ge=0.0)
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
