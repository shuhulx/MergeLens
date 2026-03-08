"""Test fixtures — tiny safetensors models for unit tests.

Creates models with hidden_size=32, 2 layers. ~KB each, no network access.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def _create_tiny_model(path: Path, seed: int = 42, hidden: int = 32, layers: int = 2):
    """Create a tiny safetensors model for testing."""
    torch.manual_seed(seed)
    tensors = {}

    # Embedding
    tensors["model.embed_tokens.weight"] = torch.randn(100, hidden)

    for i in range(layers):
        prefix = f"model.layers.{i}"
        # Attention
        tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden, hidden)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        # MLP
        tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(hidden * 4, hidden)
        tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(hidden * 4, hidden)
        tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden, hidden * 4)
        # Norm
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden)

    # LM head
    tensors["lm_head.weight"] = torch.randn(100, hidden)
    tensors["model.norm.weight"] = torch.ones(hidden)

    save_file(tensors, str(path / "model.safetensors"))

    # Write a minimal config.json
    import json
    config = {
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "intermediate_size": hidden * 4,
        "vocab_size": 100,
        "architectures": ["LlamaForCausalLM"],
    }
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture
def tmp_models(tmp_path):
    """Create two tiny models for comparison testing."""
    model_a = tmp_path / "model_a"
    model_b = tmp_path / "model_b"
    model_a.mkdir()
    model_b.mkdir()
    _create_tiny_model(model_a, seed=42)
    _create_tiny_model(model_b, seed=123)
    return str(model_a), str(model_b)


@pytest.fixture
def tmp_identical_models(tmp_path):
    """Create two identical models."""
    model_a = tmp_path / "model_a"
    model_b = tmp_path / "model_b"
    model_a.mkdir()
    model_b.mkdir()
    _create_tiny_model(model_a, seed=42)
    _create_tiny_model(model_b, seed=42)
    return str(model_a), str(model_b)


@pytest.fixture
def tmp_model_path(tmp_path):
    """Create a single tiny model."""
    model = tmp_path / "model"
    model.mkdir()
    _create_tiny_model(model, seed=42)
    return str(model)


@pytest.fixture
def sample_mergekit_yaml():
    """Sample MergeKit YAML config."""
    return """
merge_method: slerp
slices:
  - sources:
      - model: model_a
        layer_range: [0, 2]
      - model: model_b
        layer_range: [0, 2]
parameters:
  t:
    - 0.5
dtype: bfloat16
"""
