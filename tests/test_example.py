"""The documented synthetic example runs through its public artifact path."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_synthetic_example_generates_valid_artifacts(tmp_path):
    repository = Path(__file__).parents[1]
    output = tmp_path / "artifacts"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "examples" / "synthetic_demo.py"),
            "--output-dir",
            str(output),
        ],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "not scientific outcome validation" in completed.stdout
    data = json.loads((output / "comparison.json").read_text())
    assert data["mci"]["validation_status"] == "heuristic_unvalidated"
    assert "Pairwise comparison coverage" in (output / "report.html").read_text()
    assert "merge_method" in (output / "starting-config.yaml").read_text()
