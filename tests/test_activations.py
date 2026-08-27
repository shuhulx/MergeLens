"""Known-answer tests for activation extraction and linear CKA provenance."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mergelens.activations.cka import compare_activations_cka
from mergelens.activations.extractor import ActivationExtractor, ActivationSet
from mergelens.compare.metrics import cka_similarity


def _set(activations, calibration_id="calibration"):
    sample_count = next(iter(activations.values())).shape[0] if activations else 0
    return ActivationSet(activations, calibration_id, sample_count)


def _numpy_linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    x = first - first.mean(axis=0, keepdims=True)
    y = second - second.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(numerator / denominator)


def test_cka_matches_independent_numpy_reference():
    generator = torch.Generator().manual_seed(7)
    first = torch.randn(31, 5, generator=generator)
    second = torch.randn(31, 9, generator=generator)
    expected = _numpy_linear_cka(first.numpy(), second.numpy())
    assert cka_similarity(first, second) == pytest.approx(expected, abs=1e-6)


def test_cka_identical_scaling_and_orthogonal_feature_invariance():
    generator = torch.Generator().manual_seed(8)
    values = torch.randn(60, 12, generator=generator)
    orthogonal, _ = torch.linalg.qr(torch.randn(12, 12, generator=generator))
    assert cka_similarity(values, values) == pytest.approx(1.0, abs=1e-5)
    assert cka_similarity(values, values * 13.0) == pytest.approx(1.0, abs=1e-5)
    assert cka_similarity(values, values @ orthogonal) == pytest.approx(1.0, abs=1e-5)
    assert cka_similarity(values * 1e-8, values * 1e-8) == pytest.approx(1.0, abs=1e-5)


def test_cka_rejects_degenerate_or_nonfinite_activations():
    with pytest.raises(ValueError, match="undefined"):
        cka_similarity(torch.ones(4, 3), torch.ones(4, 5))
    bad = torch.randn(4, 3)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        cka_similarity(bad, torch.randn(4, 5))


def test_cka_supports_different_feature_dimensions_and_random_is_lower():
    generator = torch.Generator().manual_seed(9)
    latent = torch.randn(100, 4, generator=generator)
    first = latent @ torch.randn(4, 7, generator=generator)
    second = latent @ torch.randn(4, 11, generator=generator)
    unrelated = torch.randn(100, 11, generator=generator)
    assert cka_similarity(first, second) > cka_similarity(first, unrelated)


def test_cka_rejects_sample_mismatch():
    with pytest.raises(ValueError, match="Sample count mismatch"):
        cka_similarity(torch.randn(4, 3), torch.randn(5, 9))


def test_cka_comparison_records_alignment_provenance():
    first = _set({"layer.0": torch.randn(20, 4), "layer.2": torch.randn(20, 5)})
    second = _set({"layer.0": torch.randn(20, 8), "layer.1": torch.randn(20, 5)})
    result = compare_activations_cka(first, second)
    assert result.aligned_layers == ("layer.0",)
    assert result.calibration_id == "calibration"
    assert result.sample_count == 20


def test_cka_validates_recorded_rows_and_warns_when_features_exceed_samples():
    invalid = ActivationSet({"layer.0": torch.randn(2, 3)}, "calibration", 99)
    with pytest.raises(ValueError, match="Recorded sample count mismatch"):
        compare_activations_cka(invalid, invalid)
    first = ActivationSet({"layer.0": torch.randn(4, 8)}, "calibration", 4)
    second = ActivationSet({"layer.0": torch.randn(4, 9)}, "calibration", 4)
    result = compare_activations_cka(first, second)
    assert result.warnings


def test_cka_comparison_rejects_different_calibration_text_identity():
    first = _set({"layer.0": torch.randn(20, 4)}, "first")
    second = _set({"layer.0": torch.randn(20, 4)}, "second")
    with pytest.raises(ValueError, match="Calibration identity mismatch"):
        compare_activations_cka(first, second)


class SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 3)

    def forward(self, values):
        return self.embed(values)


def test_activation_extraction_is_padding_aware():
    model = SequenceModel()
    with torch.no_grad():
        model.embed.weight.copy_(torch.arange(30, dtype=torch.float32).reshape(10, 3))
    tokens = torch.tensor([[1, 2, 9], [3, 9, 9]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    extractor = ActivationExtractor(model, ["embed"])
    extractor.set_attention_mask(mask)
    with extractor:
        model(tokens)
    expected = torch.stack([model.embed.weight[[1, 2]].mean(dim=0), model.embed.weight[3]])
    assert torch.allclose(extractor.get_activations()["embed"], expected)


def test_sequence_pooling_requires_an_attention_mask():
    model = SequenceModel()
    extractor = ActivationExtractor(model, ["embed"])
    with extractor, pytest.raises(ValueError, match="attention mask"):
        model(torch.tensor([[1, 2]]))


def test_activation_hooks_are_removed_and_outputs_are_detached():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    extractor = ActivationExtractor(model, ["0"])
    with extractor:
        model(torch.randn(2, 3))
    output = extractor.get_activations()["0"]
    assert not output.requires_grad
    assert output.device.type == "cpu"
    assert extractor._hooks == []
