"""MergeLens command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="mergelens",
    help="Experimental static inspection for homologous LLM checkpoints.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def compare(
    models: list[str] = typer.Argument(..., help="Model paths or Hugging Face repo IDs (2+)."),
    base_model: str | None = typer.Option(
        None, "--base", "-b", help="Explicit shared base for task-vector signals."
    ),
    device: str = typer.Option("cpu", "--device", "-d", help="Torch device."),
    svd_rank: int = typer.Option(64, "--svd-rank", "-k", help="Retained SVD output rank."),
    metric: list[str] | None = typer.Option(
        None, "--metric", "-m", help="Repeat to select diagnostic signals explicitly."
    ),
    report: Path | None = typer.Option(None, "--report", "-r", help="Save an HTML report."),
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Save JSON results."),
    no_strategy: bool = typer.Option(
        False, "--no-strategy", help="Skip the rule-based starting configuration."
    ),
) -> None:
    """Compare checkpoints and expose coverage, raw signals, and unavailable metrics."""

    from mergelens.compare.analyzer import compare_models

    if len(models) < 2:
        console.print("[red]Error: Need at least 2 models to compare.[/red]")
        raise typer.Exit(1)
    console.print(
        Panel(
            f"Models: {', '.join(models)}\n"
            f"Reference: {base_model or models[0]} "
            f"({'explicit shared base' if base_model else 'implicit comparison reference'})\n"
            f"Device: {device} | retained SVD rank: {svd_rank}",
            title="[cyan]MergeLens static inspection[/cyan]",
            border_style="cyan",
        )
    )
    try:
        result = compare_models(
            model_paths=models,
            base_model=base_model,
            device=device,
            metrics=metric,
            svd_rank=svd_rank,
            include_strategy=not no_strategy,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    _print_coverage(result)
    _print_availability(result)
    _print_assessment(result)
    _print_tensor_metrics(result)
    _print_conflict_regions(result)
    _print_strategy(result)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(result.model_dump_json(indent=2))
        console.print(f"\n[green]JSON saved to {output_json}[/green]")
    if report:
        try:
            from mergelens.report.generator import generate_report
        except ImportError as exc:
            console.print("[yellow]HTML reports require: pip install mergelens[report][/yellow]")
            raise typer.Exit(1) from exc
        generate_report(compare_result=result, output_path=str(report))
        console.print(f"\n[green]HTML report saved to {report}[/green]")


def _print_coverage(result) -> None:
    table = Table(title="Exact comparison coverage", box=box.SIMPLE)
    table.add_column("Pair")
    table.add_column("Tensors", justify="right")
    table.add_column("Reference params", justify="right")
    table.add_column("Candidate params", justify="right")
    table.add_column("Scoring")
    for coverage in result.coverage:
        reference_pct = (
            f"{coverage.parameter_coverage_reference:.1%}"
            if coverage.parameter_coverage_reference is not None
            else "unknown"
        )
        candidate_pct = (
            f"{coverage.parameter_coverage_candidate:.1%}"
            if coverage.parameter_coverage_candidate is not None
            else "unknown"
        )
        table.add_row(
            f"{coverage.reference_model} → {coverage.candidate_model}",
            f"{coverage.exact_shape_compatible_tensor_count}/{coverage.common_tensor_name_count} exact",
            reference_pct,
            candidate_pct,
            "supported" if coverage.scoring_supported else "suppressed",
        )
    console.print(table)
    for coverage in result.coverage:
        for warning in coverage.warnings:
            console.print(
                f"[yellow]Coverage warning ({coverage.comparison_id}): {warning}[/yellow]"
            )
        for condition in coverage.unsupported_conditions:
            console.print(f"[red]Unsupported ({coverage.comparison_id}): {condition}[/red]")


def _print_availability(result) -> None:
    available = [item for item in result.metric_availability if item.status.value == "computed"]
    console.print(
        f"\n[bold]Available diagnostic signals: {len(available)}/{len(result.metric_availability)}[/bold]"
    )
    unavailable = [item for item in result.metric_availability if item.status.value != "computed"]
    if unavailable:
        table = Table(title="Unavailable or skipped signals", box=box.SIMPLE)
        table.add_column("Signal", style="cyan")
        table.add_column("Status")
        table.add_column("Reason")
        for item in unavailable:
            table.add_row(item.metric, item.status.value, item.reason or "—")
        console.print(table)


def _print_assessment(result) -> None:
    assessment = result.mci
    if assessment.score is None:
        body = (
            "[bold]Score suppressed[/bold]\n"
            f"Risk tier: {assessment.risk_tier}\n"
            "Raw coverage and diagnostic values remain available."
        )
    else:
        body = (
            f"[bold]{assessment.score:.1f}[/bold] / 100\n"
            f"Risk tier: {assessment.risk_tier}\n"
            f"Evidence coverage: {assessment.evidence_coverage:.0%}\n"
            f"Heuristic sensitivity band: "
            f"{assessment.heuristic_band_lower:.1f}-{assessment.heuristic_band_upper:.1f}\n"
            "Validation status: heuristic_unvalidated"
        )
    console.print(
        Panel(
            body,
            title="[bold]Unvalidated static-risk heuristic[/bold]",
            border_style="yellow",
        )
    )


def _format_optional(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def _print_tensor_metrics(result) -> None:
    table = Table(title="Tensor metrics", box=box.ROUNDED)
    table.add_column("Reference", max_width=20)
    table.add_column("Candidate", max_width=20)
    table.add_column("Tensor", max_width=45)
    table.add_column("Block", justify="right")
    table.add_column("Cosine", justify="right")
    table.add_column("L2", justify="right")
    table.add_column("Spectral", justify="right")
    for row in result.tensor_metrics:
        table.add_row(
            row.reference_model,
            row.candidate_model,
            row.tensor_name,
            str(row.transformer_block) if row.transformer_block is not None else "—",
            _format_optional(row.cosine_similarity),
            _format_optional(row.l2_distance),
            _format_optional(row.spectral_overlap),
        )
    console.print(table)


def _print_conflict_regions(result) -> None:
    if not result.tensor_conflict_regions:
        return
    console.print("\n[bold]Heuristic tensor inspection priorities[/bold]")
    for region in result.tensor_conflict_regions:
        console.print(
            Panel(
                f"Pair: {region.reference_model} → {region.candidate_model}\n"
                f"Ordered tensor positions: {region.start_tensor_position}-{region.end_tensor_position}\n"
                f"Tensors: {', '.join(region.tensor_names)}\n"
                f"Triggers: {'; '.join(region.triggering_signals)}\n"
                f"{region.heuristic_inspection_note}",
                title=f"{region.severity.value} inspection priority",
                border_style="yellow",
            )
        )


def _print_strategy(result) -> None:
    if result.strategy is None:
        return
    strategy = result.strategy
    console.print(
        Panel(
            f"Method to test: {strategy.method.value}\n"
            f"Rule strength (non-statistical): {strategy.heuristic_strength:.0%}\n"
            f"Configuration status: {strategy.config_status}\n\n"
            f"{strategy.reasoning}\n\n"
            f"Illustrative MergeKit starting configuration:\n{strategy.mergekit_yaml}",
            title="[bold]Rule-based starting point[/bold]",
            border_style="cyan",
        )
    )
    for warning in strategy.warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


@app.command()
def diagnose(
    config: Path = typer.Argument(..., help="Path to a MergeKit YAML config."),
    device: str = typer.Option("cpu", "--device", "-d", help="Torch device."),
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Save JSON results."),
) -> None:
    """Describe the supported static subset of a MergeKit configuration."""

    from mergelens.diagnose import diagnose_config

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)
    try:
        result = diagnose_config(str(config), device=device)
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[bold]Analysis status:[/bold] {result.analysis_status}")
    console.print(f"[bold]Static proxy:[/bold] {result.overall_interference:.4f}")
    console.print(f"[bold]Honored:[/bold] {', '.join(result.honored_features) or 'none'}")
    console.print(f"[bold]Ignored:[/bold] {', '.join(result.ignored_features) or 'none'}")
    console.print(f"[bold]Unsupported:[/bold] {', '.join(result.unsupported_features) or 'none'}")
    for note in result.recommendations:
        console.print(f"  • {note}")
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(result.model_dump_json(indent=2))
        console.print(f"[green]JSON saved to {output_json}[/green]")


@app.command()
def serve(
    transport: str = typer.Option("stdio", "--transport", "-t", help="MCP transport."),
) -> None:
    """Start the MergeLens MCP server."""

    try:
        from mergelens.mcp.server import create_server
    except ImportError as exc:
        console.print("[yellow]MCP support requires: pip install mergelens[mcp][/yellow]")
        raise typer.Exit(1) from exc
    create_server().run(transport=transport)


def version_callback(value: bool) -> None:
    if value:
        from mergelens import __version__

        typer.echo(f"mergelens {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """MergeLens — experimental static checkpoint inspection."""


if __name__ == "__main__":
    app()
