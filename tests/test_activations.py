"""Tests for activation-level comparison (CKA) and extraction."""

import pytest
import torch
import torch.nn as nn

from mergelens.activations.cka import compare_activations_cka
from mergelens.activations.extractor import ActivationExtractor


class TestCompareActivationsCKA:
    def test_identical_activations(self):
        acts = {"layer.0": torch.randn(50, 32), "layer.1": torch.randn(50, 64)}
        result = compare_activations_cka(acts, acts)
        assert set(result.keys()) == {"layer.0", "layer.1"}
        for score in result.values():
            assert score == pytest.approx(1.0, abs=1e-3)

    def test_random_activations_bounded(self):
        acts_a = {"layer.0": torch.randn(50, 32), "layer.1": torch.randn(50, 64)}
        acts_b = {"layer.0": torch.randn(50, 32), "layer.1": torch.randn(50, 64)}
        result = compare_activations_cka(acts_a, acts_b)
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_only_common_layers(self):
        acts_a = {"layer.0": torch.randn(50, 32), "layer.2": torch.randn(50, 32)}
        acts_b = {"layer.0": torch.randn(50, 32), "layer.1": torch.randn(50, 32)}
        result = compare_activations_cka(acts_a, acts_b)
        assert list(result.keys()) == ["layer.0"]

    def test_no_common_layers(self):
        acts_a = {"layer.0": torch.randn(50, 32)}
        acts_b = {"layer.1": torch.randn(50, 32)}
        result = compare_activations_cka(acts_a, acts_b)
        assert result == {}

    def test_mismatched_samples_truncated(self):
        acts_a = {"layer.0": torch.randn(30, 32)}
        acts_b = {"layer.0": torch.randn(50, 32)}
        result = compare_activations_cka(acts_a, acts_b)
        assert "layer.0" in result
        assert 0.0 <= result["layer.0"] <= 1.0

    def test_scores_rounded(self):
        acts = {"layer.0": torch.randn(50, 32)}
        result = compare_activations_cka(acts, acts)
        score_str = str(result["layer.0"])
        decimals = score_str.split(".")[-1] if "." in score_str else ""
        assert len(decimals) <= 4

    def test_orthogonal_activations_low(self):
        n = 50
        a = torch.zeros(n, 4)
        a[:, 0] = torch.randn(n)
        b = torch.zeros(n, 4)
        b[:, 2] = torch.randn(n)
        result = compare_activations_cka({"layer.0": a}, {"layer.0": b})
        assert result["layer.0"] < 0.3


class TestActivationExtractor:
    def _make_model(self):
        model = nn.Sequential()
        model.add_module("linear1", nn.Linear(16, 32))
        model.add_module("relu", nn.ReLU())
        model.add_module("linear2", nn.Linear(32, 8))
        return model

    def test_extracts_named_layers(self):
        model = self._make_model()
        extractor = ActivationExtractor(model, layer_names=["linear1", "linear2"])
        with extractor:
            model(torch.randn(4, 16))
        acts = extractor.get_activations()
        assert "linear1" in acts
        assert "linear2" in acts
        assert acts["linear1"].shape == (4, 32)
        assert acts["linear2"].shape == (4, 8)

    def test_context_manager_removes_hooks(self):
        model = self._make_model()
        extractor = ActivationExtractor(model, layer_names=["linear1"])
        with extractor:
            pass
        assert len(extractor._hooks) == 0

    def test_multiple_forward_passes_concatenated(self):
        model = self._make_model()
        extractor = ActivationExtractor(model, layer_names=["linear1"])
        with extractor:
            model(torch.randn(3, 16))
            model(torch.randn(5, 16))
        acts = extractor.get_activations()
        assert acts["linear1"].shape == (8, 32)

    def test_no_layers_requested(self):
        model = self._make_model()
        extractor = ActivationExtractor(model, layer_names=[])
        with extractor:
            model(torch.randn(4, 16))
        assert extractor.get_activations() == {}

    def test_nonexistent_layer_ignored(self):
        model = self._make_model()
        extractor = ActivationExtractor(model, layer_names=["nonexistent"])
        with extractor:
            model(torch.randn(4, 16))
        assert extractor.get_activations() == {}

    def test_clear(self):
        model = self._make_model()
        extractor = ActivationExtractor(model, layer_names=["linear1"])
        with extractor:
            model(torch.randn(4, 16))
        assert len(extractor._activations["linear1"]) > 0
        extractor.clear()
        assert len(extractor._activations["linear1"]) == 0

    def test_3d_output_mean_pooled(self):
        class FakeSeqModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 16)
                self.proj = nn.Linear(16, 8)

            def forward(self, x):
                return self.proj(self.embed(x))

        model = FakeSeqModel()
        extractor = ActivationExtractor(model, layer_names=["embed"])
        with extractor:
            model(torch.randint(0, 10, (2, 5)))
        acts = extractor.get_activations()
        assert acts["embed"].shape == (2, 16)

    def test_tuple_output_handled(self):
        class TupleModule(nn.Module):
            def forward(self, x):
                return (x * 2, x * 3)

        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = TupleModule()

            def forward(self, x):
                out, _ = self.block(x)
                return out

        model = TupleModel()
        extractor = ActivationExtractor(model, layer_names=["block"])
        with extractor:
            model(torch.randn(4, 8))
        acts = extractor.get_activations()
        assert acts["block"].shape == (4, 8)

    def test_activations_detached_and_cpu(self):
        model = self._make_model()
        extractor = ActivationExtractor(model, layer_names=["linear1"])
        with extractor:
            model(torch.randn(4, 16))
        acts = extractor.get_activations()
        assert not acts["linear1"].requires_grad
        assert acts["linear1"].device == torch.device("cpu")
