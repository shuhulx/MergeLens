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
    assert "Unvalidated static-risk heuristic" in result.stdout
    assert "Exact comparison coverage" in result.stdout
    assert "Available diagnostic signals" in result.stdout


def test_removed_public_options_and_commands_are_absent():
    help_result = runner.invoke(app, ["--help"])
    compare_help = runner.invoke(app, ["compare", "--help"])
    diagnose_help = runner.invoke(app, ["diagnose", "--help"])
    assert "audit" not in help_result.stdout.lower()
    assert "--no-cache" not in compare_help.stdout
    assert "--report" not in diagnose_help.stdout


def test_compare_with_json(tmp_models, tmp_path):
    json_path = str(tmp_path / "result.json")
    result = runner.invoke(app, ["compare", tmp_models[0], tmp_models[1], "--json", json_path])
    assert result.exit_code == 0
    from pathlib import Path

    assert Path(json_path).exists()
