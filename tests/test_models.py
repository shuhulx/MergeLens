"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from mergelens.models import (
    AuditResult,
    CapabilityScore,
    CompareResult,
    ConflictZone,
    DiagnoseResult,
    InterferenceScore,
    LayerMetrics,
    LayerType,
    MergeCompatibilityIndex,
    MergeConfig,
    MergeMethod,
    ModelInfo,
    ProbeResult,
    Severity,
    StrategyRecommendation,
)


class TestEnums:
    def test_merge_method_values(self):
        assert MergeMethod.SLERP.value == "slerp"
        assert MergeMethod.TIES.value == "ties"
        assert MergeMethod.DARE_TIES.value == "dare_ties"
        assert MergeMethod.DELLA.value == "della"
        assert MergeMethod.MODEL_STOCK.value == "model_stock"
        assert MergeMethod.BREADCRUMBS.value == "breadcrumbs"

    def test_merge_method_from_value(self):
        assert MergeMethod("slerp") is MergeMethod.SLERP
        assert MergeMethod("linear") is MergeMethod.LINEAR

    def test_merge_method_is_str(self):
        assert isinstance(MergeMethod.SLERP, str)
        assert MergeMethod.SLERP == "slerp"

    def test_severity_ordering(self):
        values = [s.value for s in Severity]
        assert values == ["low", "medium", "high", "critical"]

    def test_layer_type_all_values(self):
        expected = {
            "attn_q", "attn_k", "attn_v", "attn_o",
            "mlp_gate", "mlp_up", "mlp_down",
            "norm", "embedding", "lm_head", "other",
        }
        assert {lt.value for lt in LayerType} == expected


class TestLayerMetrics:
    def test_minimal(self):
        m = LayerMetrics(layer_name="layer.0", cosine_similarity=0.9, l2_distance=0.1)
        assert m.layer_name == "layer.0"
        assert m.layer_type is LayerType.OTHER
        assert m.shape == ()
        assert m.spectral_overlap is None
        assert m.sign_disagreement_rate is None

    def test_full(self):
        m = LayerMetrics(
            layer_name="layer.0",
            layer_type=LayerType.ATTENTION_Q,
            shape=(32, 32),
            cosine_similarity=0.95,
            l2_distance=0.05,
            spectral_overlap=0.8,
            effective_rank_ratio=1.2,
            sign_disagreement_rate=0.1,
            tsv_interference=0.3,
            task_vector_energy=0.5,
            cka_similarity=0.9,
        )
        assert m.layer_type is LayerType.ATTENTION_Q
        assert m.shape == (32, 32)
        assert m.effective_rank_ratio == 1.2

    def test_cosine_bounds(self):
        with pytest.raises(ValidationError):
            LayerMetrics(layer_name="x", cosine_similarity=1.5, l2_distance=0.0)
        with pytest.raises(ValidationError):
            LayerMetrics(layer_name="x", cosine_similarity=-1.5, l2_distance=0.0)

    def test_l2_non_negative(self):
        with pytest.raises(ValidationError):
            LayerMetrics(layer_name="x", cosine_similarity=0.5, l2_distance=-0.1)

    def test_spectral_overlap_bounds(self):
        with pytest.raises(ValidationError):
            LayerMetrics(layer_name="x", cosine_similarity=0.5, l2_distance=0.1, spectral_overlap=1.5)

    def test_sign_disagreement_bounds(self):
        with pytest.raises(ValidationError):
            LayerMetrics(layer_name="x", cosine_similarity=0.5, l2_distance=0.1, sign_disagreement_rate=-0.1)

    def test_roundtrip(self):
        m = LayerMetrics(layer_name="layer.0", cosine_similarity=0.9, l2_distance=0.1)
        data = m.model_dump_json()
        m2 = LayerMetrics.model_validate_json(data)
        assert m == m2


class TestConflictZone:
    def test_creation(self):
        cz = ConflictZone(
            start_layer=0,
            end_layer=3,
            layer_names=["layer.0", "layer.1", "layer.2"],
            severity=Severity.HIGH,
            avg_cosine_sim=0.3,
            recommendation="Use TIES",
        )
        assert cz.severity is Severity.HIGH
        assert cz.avg_sign_disagreement is None
        assert len(cz.layer_names) == 3

    def test_roundtrip(self):
        cz = ConflictZone(
            start_layer=0,
            end_layer=1,
            layer_names=["l0"],
            severity=Severity.LOW,
            avg_cosine_sim=0.9,
            recommendation="ok",
        )
        cz2 = ConflictZone.model_validate_json(cz.model_dump_json())
        assert cz == cz2


class TestMergeCompatibilityIndex:
    def test_creation(self):
        mci = MergeCompatibilityIndex(
            score=85.0,
            confidence=0.9,
            ci_lower=80.0,
            ci_upper=90.0,
            verdict="highly compatible",
        )
        assert mci.components == {}

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            MergeCompatibilityIndex(
                score=101.0, confidence=0.5, ci_lower=0.0, ci_upper=100.0, verdict="x"
            )
        with pytest.raises(ValidationError):
            MergeCompatibilityIndex(
                score=-1.0, confidence=0.5, ci_lower=0.0, ci_upper=100.0, verdict="x"
            )

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            MergeCompatibilityIndex(
                score=50.0, confidence=1.5, ci_lower=0.0, ci_upper=100.0, verdict="x"
            )

    def test_roundtrip(self):
        mci = MergeCompatibilityIndex(
            score=50.0, confidence=0.8, ci_lower=40.0, ci_upper=60.0,
            verdict="compatible", components={"cos": 0.9},
        )
        mci2 = MergeCompatibilityIndex.model_validate_json(mci.model_dump_json())
        assert mci == mci2


class TestStrategyRecommendation:
    def test_creation(self):
        sr = StrategyRecommendation(
            method=MergeMethod.SLERP,
            confidence=0.85,
            reasoning="High similarity",
            mergekit_yaml="merge_method: slerp\n",
        )
        assert sr.warnings == []
        assert sr.per_layer_overrides == {}

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            StrategyRecommendation(
                method=MergeMethod.LINEAR, confidence=2.0,
                reasoning="x", mergekit_yaml="x",
            )

    def test_roundtrip(self):
        sr = StrategyRecommendation(
            method=MergeMethod.TIES,
            confidence=0.7,
            reasoning="reason",
            mergekit_yaml="yaml",
            warnings=["warn1"],
            per_layer_overrides={"layer.0": {"t": 0.3}},
        )
        sr2 = StrategyRecommendation.model_validate_json(sr.model_dump_json())
        assert sr == sr2


class TestModelInfo:
    def test_minimal(self):
        mi = ModelInfo(name="llama", path_or_repo="/tmp/llama")
        assert mi.num_parameters is None
        assert mi.architecture is None
        assert mi.num_layers is None

    def test_full(self):
        mi = ModelInfo(
            name="llama", path_or_repo="meta-llama/Llama-2",
            num_parameters=7_000_000_000, architecture="LlamaForCausalLM", num_layers=32,
        )
        assert mi.num_parameters == 7_000_000_000


class TestCompareResult:
    def test_creation(self):
        cr = CompareResult(
            models=[ModelInfo(name="a", path_or_repo="/a")],
            layer_metrics=[
                LayerMetrics(layer_name="l0", cosine_similarity=0.9, l2_distance=0.1),
            ],
            conflict_zones=[],
            mci=MergeCompatibilityIndex(
                score=80.0, confidence=0.9, ci_lower=75.0, ci_upper=85.0, verdict="compatible",
            ),
        )
        assert cr.strategy is None
        assert cr.metadata == {}

    def test_roundtrip(self):
        cr = CompareResult(
            models=[ModelInfo(name="a", path_or_repo="/a")],
            layer_metrics=[],
            conflict_zones=[],
            mci=MergeCompatibilityIndex(
                score=50.0, confidence=0.5, ci_lower=40.0, ci_upper=60.0, verdict="risky",
            ),
            metadata={"key": "val"},
        )
        cr2 = CompareResult.model_validate_json(cr.model_dump_json())
        assert cr == cr2


class TestMergeConfig:
    def test_minimal(self):
        mc = MergeConfig(merge_method=MergeMethod.SLERP, models=["a", "b"])
        assert mc.base_model is None
        assert mc.parameters == {}
        assert mc.slices is None
        assert mc.raw_yaml == ""

    def test_roundtrip(self):
        mc = MergeConfig(
            merge_method=MergeMethod.TIES,
            base_model="base",
            models=["a", "b"],
            parameters={"density": 0.5},
            raw_yaml="merge_method: ties\n",
        )
        mc2 = MergeConfig.model_validate_json(mc.model_dump_json())
        assert mc == mc2


class TestInterferenceScore:
    def test_creation(self):
        i = InterferenceScore(layer_name="layer.0", score=0.5)
        assert i.source_contributions == {}

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            InterferenceScore(layer_name="x", score=1.5)
        with pytest.raises(ValidationError):
            InterferenceScore(layer_name="x", score=-0.1)

    def test_roundtrip(self):
        i = InterferenceScore(
            layer_name="l0", score=0.3, source_contributions={"a": 0.6, "b": 0.4},
        )
        i2 = InterferenceScore.model_validate_json(i.model_dump_json())
        assert i == i2


class TestDiagnoseResult:
    def test_creation(self):
        dr = DiagnoseResult(
            config=MergeConfig(merge_method=MergeMethod.LINEAR, models=["a", "b"]),
            interference_scores=[InterferenceScore(layer_name="l0", score=0.2)],
            overall_interference=0.2,
        )
        assert dr.conflict_zones == []
        assert dr.recommendations == []
        assert dr.attribution_map == {}

    def test_overall_interference_bounds(self):
        with pytest.raises(ValidationError):
            DiagnoseResult(
                config=MergeConfig(merge_method=MergeMethod.LINEAR, models=["a"]),
                interference_scores=[],
                overall_interference=1.5,
            )


class TestProbeResult:
    def test_creation(self):
        pr = ProbeResult(
            probe_id="p1", category="reasoning", prompt="2+2?",
            response="4", score=1.0,
        )
        assert pr.judge_reasoning is None

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            ProbeResult(
                probe_id="p1", category="c", prompt="p",
                response="r", score=1.1,
            )


class TestCapabilityScore:
    def test_creation(self):
        cs = CapabilityScore(
            category="math", base_score=0.9, merged_score=0.85,
            retention=0.944, num_probes=10,
        )
        assert cs.retention == pytest.approx(0.944)

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            CapabilityScore(
                category="x", base_score=1.5, merged_score=0.5,
                retention=1.0, num_probes=1,
            )


class TestAuditResult:
    def test_creation(self):
        ar = AuditResult(
            base_model="base",
            merged_model="merged",
            capability_scores=[
                CapabilityScore(
                    category="math", base_score=0.9, merged_score=0.85,
                    retention=0.944, num_probes=5,
                ),
            ],
            overall_retention=0.944,
        )
        assert ar.probe_results == []
        assert ar.regressions == []
        assert ar.improvements == []

    def test_roundtrip(self):
        ar = AuditResult(
            base_model="b",
            merged_model="m",
            capability_scores=[],
            overall_retention=1.0,
            regressions=["math dropped"],
            improvements=["code improved"],
        )
        ar2 = AuditResult.model_validate_json(ar.model_dump_json())
        assert ar == ar2
