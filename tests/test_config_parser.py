"""Tests for MergeKit config parser."""

import pytest

from mergelens.diagnose.config_parser import parse_mergekit_config
from mergelens.models import MergeMethod


def test_parse_slerp_config(sample_mergekit_yaml):
    config = parse_mergekit_config(sample_mergekit_yaml)
    assert config.merge_method == MergeMethod.SLERP
    assert len(config.models) >= 2


def test_parse_ties_config():
    yaml_content = """
merge_method: ties
base_model: base_model
models:
  - model: model_a
  - model: model_b
parameters:
  density: 0.5
  weight:
    - 0.5
    - 0.5
"""
    config = parse_mergekit_config(yaml_content)
    assert config.merge_method == MergeMethod.TIES
    assert config.base_model == "base_model"
    assert "model_a" in config.models


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_mergekit_config("- just a list\n- not a mapping")


def test_unknown_merge_method_is_not_coerced_to_linear():
    with pytest.raises(ValueError, match="Unsupported MergeKit method 'future_method'"):
        parse_mergekit_config("merge_method: future_method\nmodels:\n  - model: a\n  - model: b\n")


def test_unsupported_and_ignored_config_features_are_disclosed():
    config = parse_mergekit_config(
        """
merge_method: linear
slices:
  - sources:
      - model: model_a
        layer_range: [0, 1]
        parameters:
          weight: [0.0, 1.0]
      - model: model_b
        layer_range: [0, 1]
tokenizer_source: base
chat_template: auto
parameters:
  normalize: true
"""
    )
    disclosure = " ".join(config.ignored_features)
    assert "slice layer ranges" in disclosure
    assert "tokenizer" in disclosure
    assert "chat-template" in disclosure
    assert "merge-method-specific" in disclosure


def test_models_slices_and_modules_are_mutually_exclusive():
    with pytest.raises(ValueError, match="Exactly one"):
        parse_mergekit_config("merge_method: linear\nmodels: [a, b]\nslices: [{sources: [a, b]}]\n")
