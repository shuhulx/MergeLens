"""Public data models for MergeLens results and configuration analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


class MergeMethod(str, Enum):
    """MergeKit methods understood by the configuration parser."""

    LINEAR = "linear"
    SLERP = "slerp"
    NUSLERP = "nuslerp"
    MULTISLERP = "multislerp"
    KARCHER = "karcher"
    TASK_ARITHMETIC = "task_arithmetic"
    TIES = "ties"
    DARE_TIES = "dare_ties"
    DARE_LINEAR = "dare_linear"
    DELLA = "della"
    DELLA_LINEAR = "della_linear"
    BREADCRUMBS = "breadcrumbs"
    BREADCRUMBS_TIES = "breadcrumbs_ties"
    SCE = "sce"
    MODEL_STOCK = "model_stock"
    NEARSWAP = "nearswap"
    ARCEE_FUSION = "arcee_fusion"
    PASSTHROUGH = "passthrough"
    RAM = "ram"
    RAMPLUS_TL = "ramplus_tl"


class Severity(str, Enum):
    """Heuristic inspection-priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LayerType(str, Enum):
    """Coarse tensor-role classifications inferred from tensor names."""

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


class ModelRole(str, Enum):
    """Role a checkpoint has in a comparison run."""

    IMPLICIT_REFERENCE = "implicit_reference"
    EXPLICIT_SHARED_BASE = "explicit_shared_base"
    CANDIDATE = "candidate"


class MetricStatus(str, Enum):
    """Why a diagnostic signal is or is not present."""

    COMPUTED = "computed"
    SKIPPED_BY_USER = "skipped_by_user"
    STRUCTURALLY_UNAVAILABLE = "structurally_unavailable"
    FAILED_NUMERICALLY = "failed_numerically"
    UNSUPPORTED_INPUT = "unsupported_input"
    RESOURCE_LIMIT_SKIPPED = "resource_limit_skipped"


class MetricObservation(BaseModel):
    """Availability of one signal for one tensor comparison."""

    status: MetricStatus
    reason: str | None = None


class MetricAvailability(BaseModel):
    """Run-level availability summary for one diagnostic signal."""

    metric: str
    status: MetricStatus
    reason: str | None = None
    computed_tensor_count: int = Field(default=0, ge=0)
    affected_tensor_count: int = Field(default=0, ge=0)


class TensorShapeMismatch(BaseModel):
    """A same-name tensor pair that cannot be compared exactly."""

    tensor_name: str
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]


class ModelInfo(BaseModel):
    """Checkpoint metadata used to establish structural comparability."""

    name: str
    path_or_repo: str
    role: ModelRole = ModelRole.CANDIDATE
    num_parameters: int | None = None
    tensor_count: int | None = None
    architecture: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    embedding_shape: tuple[int, ...] | None = None
    lm_head_shape: tuple[int, ...] | None = None


class ComparisonCoverage(BaseModel):
    """Exact tensor and parameter coverage for one checkpoint pair."""

    comparison_id: str
    reference_model: str
    candidate_model: str
    reference_role: ModelRole
    total_tensor_count_reference: int = Field(ge=0)
    total_tensor_count_candidate: int = Field(ge=0)
    total_parameter_count_reference: int = Field(ge=0)
    total_parameter_count_candidate: int = Field(ge=0)
    common_tensor_name_count: int = Field(ge=0)
    exact_shape_compatible_tensor_count: int = Field(ge=0)
    common_parameter_count: int = Field(ge=0)
    parameter_coverage_reference: float | None = Field(default=None, ge=0.0, le=1.0)
    parameter_coverage_candidate: float | None = Field(default=None, ge=0.0, le=1.0)
    tensors_missing_from_reference: list[str] = Field(default_factory=list)
    tensors_missing_from_candidate: list[str] = Field(default_factory=list)
    shape_mismatches: list[TensorShapeMismatch] = Field(default_factory=list)
    reference_architecture: str | None = None
    candidate_architecture: str | None = None
    reference_hidden_size: int | None = None
    candidate_hidden_size: int | None = None
    reference_layer_count: int | None = None
    candidate_layer_count: int | None = None
    reference_vocab_size: int | None = None
    candidate_vocab_size: int | None = None
    embedding_compatible: bool | None = None
    lm_head_compatible: bool | None = None
    explicit_shared_base: bool = False
    appears_homologous: bool = False
    scoring_supported: bool = False
    warnings: list[str] = Field(default_factory=list)
    unsupported_conditions: list[str] = Field(default_factory=list)


class TensorMetrics(BaseModel):
    """Diagnostic measurements for one exact-shape tensor pair."""

    reference_model: str = ""
    candidate_model: str = ""
    comparison_id: str = ""
    tensor_name: str = Field(validation_alias=AliasChoices("tensor_name", "layer_name"))
    tensor_position: int = Field(default=0, ge=0)
    transformer_block: int | None = Field(default=None, ge=0)
    tensor_type: LayerType = Field(
        default=LayerType.OTHER,
        validation_alias=AliasChoices("tensor_type", "layer_type"),
    )
    shape: tuple[int, ...] = ()
    parameter_count: int = Field(default=0, ge=0)
    cosine_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    l2_distance: float | None = Field(default=None, ge=0.0)
    weight_distribution_divergence: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("weight_distribution_divergence", "kl_divergence"),
    )
    spectral_overlap: float | None = Field(default=None, ge=0.0, le=1.0)
    effective_rank_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    sign_disagreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    tsv_interference: float | None = Field(default=None, ge=0.0, le=1.0)
    task_vector_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    cka_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    metric_observations: dict[str, MetricObservation] = Field(default_factory=dict)

    @property
    def layer_name(self) -> str:
        """Deprecated alias for ``tensor_name``."""

        return self.tensor_name

    @property
    def layer_type(self) -> LayerType:
        """Deprecated alias for ``tensor_type``."""

        return self.tensor_type

    @property
    def kl_divergence(self) -> float | None:
        """Deprecated alias for the descriptive directional divergence."""

        return self.weight_distribution_divergence


LayerMetrics = TensorMetrics


class TensorConflictRegion(BaseModel):
    """A contiguous ordered-tensor region prioritized for inspection."""

    comparison_id: str = ""
    reference_model: str = ""
    candidate_model: str = ""
    start_tensor_position: int = Field(
        default=0, validation_alias=AliasChoices("start_tensor_position", "start_layer")
    )
    end_tensor_position: int = Field(
        default=0, validation_alias=AliasChoices("end_tensor_position", "end_layer")
    )
    tensor_names: list[str] = Field(
        default_factory=list, validation_alias=AliasChoices("tensor_names", "layer_names")
    )
    severity: Severity
    avg_cosine_similarity: float = Field(
        validation_alias=AliasChoices("avg_cosine_similarity", "avg_cosine_sim")
    )
    avg_sign_disagreement: float | None = None
    triggering_signals: list[str] = Field(default_factory=list)
    heuristic_inspection_note: str = Field(
        validation_alias=AliasChoices("heuristic_inspection_note", "recommendation")
    )

    @property
    def start_layer(self) -> int:
        return self.start_tensor_position

    @property
    def end_layer(self) -> int:
        return self.end_tensor_position

    @property
    def layer_names(self) -> list[str]:
        return self.tensor_names

    @property
    def avg_cosine_sim(self) -> float:
        return self.avg_cosine_similarity

    @property
    def recommendation(self) -> str:
        return self.heuristic_inspection_note


ConflictZone = TensorConflictRegion


class MergeCompatibilityIndex(BaseModel):
    """Unvalidated, hand-specified summary of static diagnostic signals."""

    score: float | None = Field(default=None, ge=0.0, le=100.0)
    risk_tier: str = "insufficient_evidence"
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    available_metrics: list[str] = Field(default_factory=list)
    unavailable_metrics: list[MetricAvailability] = Field(default_factory=list)
    heuristic_band_lower: float | None = Field(default=None, ge=0.0, le=100.0)
    heuristic_band_upper: float | None = Field(default=None, ge=0.0, le=100.0)
    validation_status: Literal["heuristic_unvalidated"] = "heuristic_unvalidated"
    components: dict[str, float] = Field(default_factory=dict)
    component_weights: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_v02_names(cls, value: Any) -> Any:
        """Accept v0.2 constructor names without serializing misleading labels."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("evidence_coverage", data.pop("confidence", 0.0))
        data.setdefault("heuristic_band_lower", data.pop("ci_lower", None))
        data.setdefault("heuristic_band_upper", data.pop("ci_upper", None))
        old_verdict = data.pop("verdict", None)
        if old_verdict and "risk_tier" not in data:
            data["risk_tier"] = str(old_verdict).replace(" ", "_")
        return data

    @property
    def confidence(self) -> float:
        """Deprecated non-statistical alias for evidence coverage."""

        return self.evidence_coverage

    @property
    def ci_lower(self) -> float:
        """Deprecated alias for the non-statistical heuristic band."""

        return self.heuristic_band_lower or 0.0

    @property
    def ci_upper(self) -> float:
        """Deprecated alias for the non-statistical heuristic band."""

        return self.heuristic_band_upper or 0.0

    @property
    def verdict(self) -> str:
        """Deprecated alias for ``risk_tier``."""

        return self.risk_tier


class StrategyRecommendation(BaseModel):
    """Rule-based MergeKit starting point; not an outcome prediction."""

    method: MergeMethod
    heuristic_strength: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("heuristic_strength", "confidence"),
    )
    reasoning: str
    triggering_signals: list[str] = Field(default_factory=list)
    mergekit_yaml: str
    config_status: Literal["illustrative", "schema_validated"] = "illustrative"
    validated_against: str | None = None
    warnings: list[str] = Field(default_factory=list)
    per_tensor_overrides: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("per_tensor_overrides", "per_layer_overrides"),
    )

    @property
    def confidence(self) -> float:
        """Deprecated alias; this is heuristic rule strength, not confidence."""

        return self.heuristic_strength

    @property
    def per_layer_overrides(self) -> dict[str, Any]:
        return self.per_tensor_overrides


class CompareResult(BaseModel):
    """Complete comparison result with explicit evidence boundaries."""

    models: list[ModelInfo]
    reference_model: ModelInfo | None = None
    explicit_base: ModelInfo | None = None
    coverage: list[ComparisonCoverage] = Field(default_factory=list)
    tensor_metrics: list[TensorMetrics] = Field(
        default_factory=list,
        validation_alias=AliasChoices("tensor_metrics", "layer_metrics"),
    )
    tensor_conflict_regions: list[TensorConflictRegion] = Field(
        default_factory=list,
        validation_alias=AliasChoices("tensor_conflict_regions", "conflict_zones"),
    )
    mci: MergeCompatibilityIndex
    pair_assessments: dict[str, MergeCompatibilityIndex] = Field(default_factory=dict)
    metric_availability: list[MetricAvailability] = Field(default_factory=list)
    strategy: StrategyRecommendation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def layer_metrics(self) -> list[TensorMetrics]:
        """Deprecated alias for ``tensor_metrics``."""

        return self.tensor_metrics

    @property
    def conflict_zones(self) -> list[TensorConflictRegion]:
        """Deprecated alias for ``tensor_conflict_regions``."""

        return self.tensor_conflict_regions


class MergeConfig(BaseModel):
    """Parsed subset of a MergeKit configuration."""

    merge_method: MergeMethod
    base_model: str | None = None
    models: list[str]
    model_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    slices: list[dict[str, Any]] | None = None
    tokenizer: dict[str, Any] | None = None
    tokenizer_source: str | None = None
    chat_template: str | None = None
    dtype: str | None = None
    honored_features: list[str] = Field(default_factory=list)
    ignored_features: list[str] = Field(default_factory=list)
    unsupported_features: list[str] = Field(default_factory=list)
    raw_yaml: str = ""


class InterferenceScore(BaseModel):
    """Descriptive task-vector or equal-average conflict proxy for one tensor."""

    tensor_name: str = Field(validation_alias=AliasChoices("tensor_name", "layer_name"))
    score: float = Field(ge=0.0, le=1.0)
    source_similarity_profile: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("source_similarity_profile", "source_contributions"),
    )

    @property
    def layer_name(self) -> str:
        return self.tensor_name

    @property
    def source_contributions(self) -> dict[str, float]:
        """Deprecated alias; values are similarities, not causal contributions."""

        return self.source_similarity_profile


class DiagnoseResult(BaseModel):
    """Configuration diagnosis with an explicit simulation-scope disclosure."""

    config: MergeConfig
    interference_scores: list[InterferenceScore]
    source_similarity_profiles: dict[str, dict[str, float]] = Field(default_factory=dict)
    tensor_conflict_regions: list[TensorConflictRegion] = Field(default_factory=list)
    overall_interference: float = Field(ge=0.0, le=1.0)
    analysis_status: str = "descriptive_proxy_only"
    honored_features: list[str] = Field(default_factory=list)
    ignored_features: list[str] = Field(default_factory=list)
    unsupported_features: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    @property
    def attribution_map(self) -> dict[str, dict[str, float]]:
        """Deprecated alias for non-causal similarity profiles."""

        return self.source_similarity_profiles

    @property
    def conflict_zones(self) -> list[TensorConflictRegion]:
        return self.tensor_conflict_regions
