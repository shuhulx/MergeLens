"""Tests for the CLI."""

from typer.testing import CliRunner

from mergelens.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MergeLens" in result.stdout or "mergelens" in result.stdout


def test_compare_help():
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "compare" in result.stdout.lower()


def test_compare_basic(tmp_models):
    result = runner.invoke(app, ["compare", tmp_models[0], tmp_models[1]])
    assert result.exit_code == 0
    assert "Merge Compatibility Index" in result.stdout


def test_compare_with_json(tmp_models, tmp_path):
    json_path = str(tmp_path / "result.json")
    result = runner.invoke(app, ["compare", tmp_models[0], tmp_models[1], "--json", json_path])
    assert result.exit_code == 0
    from pathlib import Path
    assert Path(json_path).exists()
