"""Tests for model loading."""

import json

import pytest
import torch
from safetensors.torch import save_file

import mergelens.compare.loader as loader
from mergelens.compare.loader import (
    ModelHandle,
    classify_layer,
    comparison_coverage,
    find_common_tensors,
    iter_aligned_tensors,
    tensor_sort_key,
    transformer_block_index,
)
from mergelens.models import LayerType
from mergelens.utils.hf_utils import ModelMetadata


def test_model_handle_creation(tmp_model_path):
    handle = ModelHandle(tmp_model_path)
    assert len(handle.tensor_names) > 0
    assert handle.info.name is not None


def test_model_handle_tensor_access(tmp_model_path):
    handle = ModelHandle(tmp_model_path)
    for name in handle.tensor_names[:3]:
        tensor = handle.get_tensor(name)
        assert tensor is not None
        shape = handle.get_tensor_shape(name)
        assert tuple(tensor.shape) == shape


def test_find_common_tensors(tmp_models):
    h1 = ModelHandle(tmp_models[0])
    h2 = ModelHandle(tmp_models[1])
    common = find_common_tensors([h1, h2])
    assert len(common) > 0
    assert all(n in h1.tensor_names for n in common)
    assert all(n in h2.tensor_names for n in common)


def test_iter_aligned_tensors(tmp_models):
    h1 = ModelHandle(tmp_models[0])
    h2 = ModelHandle(tmp_models[1])
    count = 0
    for _name, _layer_type, tensors in iter_aligned_tensors([h1, h2]):
        assert len(tensors) == 2
        assert tensors[0].shape == tensors[1].shape
        count += 1
    assert count > 0


def test_classify_layer():
    assert classify_layer("model.layers.0.self_attn.q_proj.weight") == LayerType.ATTENTION_Q
    assert classify_layer("model.layers.0.mlp.gate_proj.weight") == LayerType.MLP_GATE
    assert classify_layer("model.layers.0.input_layernorm.weight") == LayerType.NORM
    assert classify_layer("model.embed_tokens.weight") == LayerType.EMBEDDING
    assert classify_layer("lm_head.weight") == LayerType.LM_HEAD
    assert classify_layer("some_random_tensor") == LayerType.OTHER


def test_tensor_order_has_a_complete_deterministic_tie_breaker():
    names = [
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ]
    expected = sorted(names, key=tensor_sort_key)
    assert sorted(reversed(names), key=tensor_sort_key) == expected
    assert sorted({names[2], names[0], names[3], names[1]}, key=tensor_sort_key) == expected


def test_transformer_block_is_distinct_from_tensor_position():
    assert transformer_block_index("model.layers.17.self_attn.q_proj.weight") == 17
    assert transformer_block_index("model.embed_tokens.weight") is None


def test_missing_tensor_coverage_is_explicit(tmp_models):
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    first, second = tmp_models
    second_file = Path(second) / "model.safetensors"
    tensors = load_file(str(second_file))
    removed = tensors.pop("lm_head.weight")
    save_file(tensors, str(second_file))
    coverage = comparison_coverage(
        ModelHandle(first), ModelHandle(second), "pair", explicit_shared_base=False
    )
    assert removed.numel() > 0
    assert "lm_head.weight" in coverage.tensors_missing_from_candidate
    assert coverage.parameter_coverage_reference < 1.0
    assert coverage.scoring_supported is False


def test_nonfloating_tensor_dtype_suppresses_scoring(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    save_file({"w": torch.ones(2, dtype=torch.float32)}, str(first / "model.safetensors"))
    save_file({"w": torch.ones(2, dtype=torch.int8)}, str(second / "model.safetensors"))
    coverage = comparison_coverage(
        ModelHandle(str(first)), ModelHandle(str(second)), "pair", explicit_shared_base=False
    )
    assert coverage.dtype_issues
    assert coverage.scoring_supported is False
    assert find_common_tensors([ModelHandle(str(first)), ModelHandle(str(second))]) == []


def test_local_shard_index_fails_closed_when_a_shard_is_missing(tmp_path):
    path = tmp_path / "sharded"
    path.mkdir()
    save_file({"a": torch.ones(2)}, str(path / "part-1.safetensors"))
    index = {
        "weight_map": {
            "a": "part-1.safetensors",
            "b": "part-2.safetensors",
        }
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(FileNotFoundError, match="missing shard"):
        ModelHandle(str(path))


def test_remote_shard_downloads_are_revision_pinned_and_fail_closed(tmp_path, monkeypatch):
    shard = tmp_path / "part-1.safetensors"
    save_file({"a": torch.ones(2)}, str(shard))
    metadata = ModelMetadata(
        repo_id="org/model",
        safetensors_files=["part-1.safetensors", "part-2.safetensors"],
        revision="immutable-sha",
    )
    revisions = []

    monkeypatch.setattr(loader, "resolve_model_path", lambda _value: ("org/model", False))
    monkeypatch.setattr(loader, "get_model_metadata", lambda _value: metadata)

    def download(_repo, filename, *, revision):
        revisions.append(revision)
        if filename == "part-2.safetensors":
            raise OSError("network failure")
        return str(shard)

    monkeypatch.setattr(loader, "hf_hub_download", download)
    with pytest.raises(FileNotFoundError, match="Incomplete checkpoint download"):
        ModelHandle("org/model")
    assert revisions == ["immutable-sha", "immutable-sha"]
