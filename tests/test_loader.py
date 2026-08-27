"""Tests for model loading."""

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
