"""Tests for utils modules: tensor_ops and hf_utils."""


import numpy as np
import pytest
import torch

from mergelens.utils.hf_utils import (
    ModelMetadata,
    _estimate_params_from_config,
    _get_local_metadata,
    resolve_model_path,
)
from mergelens.utils.tensor_ops import (
    MAX_ELEMENTS_FOR_SVD,
    compute_task_vector,
    effective_rank,
    flatten_to_2d,
    grassmann_distance,
    truncated_svd,
)


class TestFlattenTo2D:
    def test_1d(self):
        t = torch.randn(16)
        result = flatten_to_2d(t)
        assert result.shape == (1, 16)

    def test_2d_passthrough(self):
        t = torch.randn(8, 16)
        result = flatten_to_2d(t)
        assert result.shape == (8, 16)
        assert torch.equal(result, t)

    def test_3d(self):
        t = torch.randn(4, 8, 16)
        result = flatten_to_2d(t)
        assert result.shape == (4, 128)

    def test_4d(self):
        t = torch.randn(2, 3, 4, 5)
        result = flatten_to_2d(t)
        assert result.shape == (2, 60)

    def test_preserves_data(self):
        t = torch.randn(3, 4, 5)
        result = flatten_to_2d(t)
        assert torch.allclose(result, t.reshape(3, 20))


class TestTruncatedSVD:
    def test_shapes(self):
        m = torch.randn(16, 32)
        U, S, Vh = truncated_svd(m, k=8)
        assert U.shape == (16, 8)
        assert S.shape == (8,)
        assert Vh.shape == (8, 32)

    def test_k_clamped_to_min_dim(self):
        m = torch.randn(4, 8)
        U, S, Vh = truncated_svd(m, k=64)
        assert U.shape == (4, 4)
        assert S.shape == (4,)
        assert Vh.shape == (4, 8)

    def test_singular_values_descending(self):
        m = torch.randn(16, 16)
        _, S, _ = truncated_svd(m, k=16)
        for i in range(len(S) - 1):
            assert S[i] >= S[i + 1]

    def test_flattens_3d_input(self):
        m = torch.randn(4, 8, 16)
        U, S, Vh = truncated_svd(m, k=4)
        assert U.shape == (4, 4)
        assert S.shape == (4,)
        assert Vh.shape == (4, 128)

    def test_too_large_raises(self):
        side = int(np.sqrt(MAX_ELEMENTS_FOR_SVD)) + 1
        m = torch.randn(side, side)
        with pytest.raises(ValueError, match="too large"):
            truncated_svd(m)

    def test_1d_input(self):
        m = torch.randn(32)
        U, S, _Vh = truncated_svd(m, k=1)
        assert U.shape[1] == 1
        assert S.shape == (1,)


class TestEffectiveRank:
    def test_rank1_matrix(self):
        a = torch.randn(32, 1)
        m = a @ a.T
        rank = effective_rank(m)
        assert rank == pytest.approx(1.0, abs=0.1)

    def test_identity(self):
        m = torch.eye(16)
        rank = effective_rank(m)
        assert rank == pytest.approx(16.0, abs=0.5)

    def test_always_ge_one(self):
        m = torch.randn(8, 8)
        assert effective_rank(m) >= 1.0

    def test_zero_matrix(self):
        m = torch.zeros(8, 8)
        assert effective_rank(m) == 1.0

    def test_1d_input(self):
        v = torch.randn(32)
        rank = effective_rank(v)
        assert rank >= 1.0


class TestGrassmannDistance:
    def test_identical_subspaces(self):
        m = torch.randn(32, 32)
        U, _, _ = truncated_svd(m, k=8)
        dist = grassmann_distance(U, U)
        assert dist == pytest.approx(0.0, abs=1e-3)

    def test_bounded(self):
        m1 = torch.randn(32, 32)
        m2 = torch.randn(32, 32)
        U1, _, _ = truncated_svd(m1, k=8)
        U2, _, _ = truncated_svd(m2, k=8)
        dist = grassmann_distance(U1, U2)
        assert 0.0 <= dist <= 1.0

    def test_orthogonal_subspaces(self):
        U1 = torch.zeros(8, 2)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0
        U2 = torch.zeros(8, 2)
        U2[2, 0] = 1.0
        U2[3, 1] = 1.0
        dist = grassmann_distance(U1, U2)
        assert dist == pytest.approx(1.0, abs=1e-5)


class TestComputeTaskVector:
    def test_basic(self):
        base = torch.ones(4, 4)
        model = torch.ones(4, 4) * 3
        tv = compute_task_vector(model, base)
        assert torch.allclose(tv, torch.ones(4, 4) * 2)

    def test_zero_difference(self):
        a = torch.randn(8, 8)
        tv = compute_task_vector(a, a)
        assert torch.allclose(tv, torch.zeros(8, 8))

    def test_casts_to_float(self):
        base = torch.ones(4, dtype=torch.bfloat16)
        model = torch.ones(4, dtype=torch.bfloat16) * 2
        tv = compute_task_vector(model, base)
        assert tv.dtype == torch.float32


class TestModelMetadata:
    def test_defaults(self):
        m = ModelMetadata(repo_id="test/model")
        assert m.safetensors_files == []
        assert m.config == {}
        assert m.architecture is None
        assert m.num_parameters is None

    def test_post_init_no_overwrite(self):
        m = ModelMetadata(repo_id="x", safetensors_files=["a.safetensors"], config={"k": "v"})
        assert m.safetensors_files == ["a.safetensors"]
        assert m.config == {"k": "v"}


class TestResolveModelPath:
    def test_local_with_safetensors(self, tmp_model_path):
        resolved, is_local = resolve_model_path(tmp_model_path)
        assert is_local is True
        assert resolved == str(tmp_model_path) or resolved.endswith("model")

    def test_nonexistent_treated_as_repo(self):
        resolved, is_local = resolve_model_path("org/some-model")
        assert is_local is False
        assert resolved == "org/some-model"

    def test_dir_without_safetensors(self, tmp_path):
        d = tmp_path / "empty_dir"
        d.mkdir()
        _resolved, is_local = resolve_model_path(str(d))
        assert is_local is False


class TestGetLocalMetadata:
    def test_reads_config(self, tmp_model_path):
        meta = _get_local_metadata(tmp_model_path)
        assert meta.architecture == "llama"
        assert meta.safetensors_files == ["model.safetensors"]
        assert meta.config["hidden_size"] == 32
        assert meta.num_parameters is not None
        assert meta.num_parameters > 0

    def test_no_config(self, tmp_path):
        d = tmp_path / "bare"
        d.mkdir()
        save_file = d / "model.safetensors"
        save_file.touch()
        meta = _get_local_metadata(str(d))
        assert meta.architecture is None
        assert meta.config == {}


class TestEstimateParamsFromConfig:
    def test_full_config(self):
        cfg = {"hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 1000, "intermediate_size": 256}
        params = _estimate_params_from_config(cfg)
        assert params is not None
        assert params > 0

    def test_missing_fields(self):
        assert _estimate_params_from_config({}) is None
        assert _estimate_params_from_config({"hidden_size": 64}) is None

    def test_infers_intermediate(self):
        cfg = {"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 500}
        params = _estimate_params_from_config(cfg)
        assert params is not None
        assert params > 0
