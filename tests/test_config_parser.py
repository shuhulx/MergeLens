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
