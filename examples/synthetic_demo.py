"""Tiny synthetic software demonstration; not merge-outcome validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from mergelens import compare_models


def _write_checkpoint(path: Path, offset: float) -> None:
    path.mkdir()
    tensors = {
        "model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4) + offset,
        "model.layers.0.self_attn.q_proj.weight": torch.eye(4) + offset,
        "lm_head.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4) + offset,
    }
    save_file(tensors, str(path / "model.safetensors"))
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "vocab_size": 6,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def main() -> None:
    """Create two fixtures, run the public API, and print evidence boundaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Write JSON, HTML, and YAML artifacts.")
    arguments = parser.parse_args()
    with TemporaryDirectory(prefix="mergelens-demo-") as directory:
        root = Path(directory)
        reference = root / "reference"
        candidate = root / "candidate"
        _write_checkpoint(reference, 0.0)
        _write_checkpoint(candidate, 0.05)
        result = compare_models(
            [str(reference), str(candidate)],
            metrics=["cosine_similarity", "l2_distance"],
            show_progress=False,
        )

    if arguments.output_dir is not None:
        from mergelens.report.exporters import export_json
        from mergelens.report.generator import generate_report

        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        export_json(result, str(arguments.output_dir / "comparison.json"))
        generate_report(
            compare_result=result,
            output_path=str(arguments.output_dir / "report.html"),
            title="MergeLens synthetic software demonstration",
        )
        if result.strategy is not None:
            (arguments.output_dir / "starting-config.yaml").write_text(
                result.strategy.mergekit_yaml,
                encoding="utf-8",
            )

    coverage = result.coverage[0]
    available = [
        item.metric for item in result.metric_availability if item.status.value == "computed"
    ]
    unavailable = [
        f"{item.metric} ({item.status.value})"
        for item in result.metric_availability
        if item.status.value != "computed"
    ]
    first_row = result.tensor_metrics[0]
    print("Synthetic software demonstration - not scientific outcome validation")
    print(
        "Coverage: "
        f"{coverage.exact_shape_compatible_tensor_count}/{coverage.total_tensor_count_reference} "
        f"tensors; reference parameters {coverage.parameter_coverage_reference:.1%}; "
        f"candidate parameters {coverage.parameter_coverage_candidate:.1%}"
    )
    print(f"Available diagnostic signals: {len(available)}/9 - {', '.join(available)}")
    print("Unavailable or skipped: " + ", ".join(unavailable))
    print(
        f"First tensor: {first_row.tensor_name}; cosine={first_row.cosine_similarity:.6f}; "
        f"normalized_l2={first_row.l2_distance:.6f}"
    )
    print(
        f"Heuristic risk tier: {result.mci.risk_tier}; score={result.mci.score}; "
        f"validation={result.mci.validation_status}"
    )
    print("Limitation: post-merge behavioural evaluation is still required.")
    if arguments.output_dir is not None:
        print(f"Artifacts written to: {arguments.output_dir}")


if __name__ == "__main__":
    main()
