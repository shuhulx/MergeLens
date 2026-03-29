"""MergeLens CLI — pre-merge diagnostics for LLM model merging."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="mergelens",
    help="Pre-merge diagnostic framework for LLM model merging.",
    no_args_is_help=True,
)
console = Console()


def _severity_color(severity: str) -> str:
    return {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(
        severity, "white"
    )


def _verdict_color(verdict: str) -> str:
    return {
        "highly compatible": "bold green",
        "compatible": "green",
        "risky": "yellow",
        "incompatible": "bold red",
    }.get(verdict, "white")


@app.command()
def compare(
    models: list[str] = typer.Argument(..., help="Model paths or HF repo IDs to compare (2+)."),
    base_model: str | None = typer.Option(
        None, "--base", "-b", help="Base model for task vectors."
    ),
    device: str = typer.Option("cpu", "--device", "-d", help="Torch device (cpu/cuda)."),
    svd_rank: int = typer.Option(64, "--svd-rank", "-k", help="SVD rank for spectral metrics."),
    report: Path | None = typer.Option(None, "--report", "-r", help="Save HTML report to path."),
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Save results as JSON."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable metric caching."),
    no_strategy: bool = typer.Option(False, "--no-strategy", help="Skip strategy recommendation."),
):
    """Compare two or more models layer-by-layer with rich diagnostics."""
    from mergelens.compare.analyzer import compare_models
    from mergelens.utils.cache import MetricCache

    if len(models) < 2:
        console.print("[red]Error: Need at least 2 models to compare.[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]MergeLens Compare[/bold]\n"
            f"Models: {', '.join(models)}\n"
            f"Device: {device} | SVD rank: {svd_rank}",
            title="[cyan]MergeLens[/cyan]",
            border_style="cyan",
        )
    )

    cache = MetricCache(enabled=not no_cache)

    try:
        try:
            result = compare_models(
                model_paths=models,
                base_model=base_model,
                device=device,
                svd_rank=svd_rank,
                cache=cache,
                include_strategy=not no_strategy,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        mci = result.mci
        color = _verdict_color(mci.verdict)
        console.print()
        console.print(
            Panel(
                f"[bold]{mci.score:.1f}[/bold] / 100  [{color}]{mci.verdict.upper()}[/{color}]\n"
                f"Confidence: {mci.confidence:.0%}  |  Range: [{mci.ci_lower:.1f}, {mci.ci_upper:.1f}]",
                title="[bold]Merge Compatibility Index[/bold]",
                border_style=color.replace("bold ", ""),
            )
        )

        if mci.components:
            comp_table = Table(title="MCI Components", box=box.SIMPLE)
            comp_table.add_column("Component", style="cyan")
            comp_table.add_column("Score", justify="right")
            for name, score in sorted(mci.components.items()):
                comp_table.add_row(name, f"{score:.4f}")
            console.print(comp_table)

        table = Table(title="Layer Metrics Summary", box=box.ROUNDED)
        table.add_column("Layer", style="dim", max_width=50)
        table.add_column("Type", style="cyan")
        table.add_column("Cosine Sim", justify="right")
        table.add_column("L2 Dist", justify="right")
        table.add_column("Spec. Overlap", justify="right")
        table.add_column("Rank Ratio", justify="right")

        for lm in result.layer_metrics:
            cos_color = (
                "green"
                if lm.cosine_similarity > 0.9
                else "yellow"
                if lm.cosine_similarity > 0.7
                else "red"
            )
            table.add_row(
                lm.layer_name[:50],
                lm.layer_type.value,
                f"[{cos_color}]{lm.cosine_similarity:.4f}[/{cos_color}]",
                f"{lm.l2_distance:.4f}",
                f"{lm.spectral_overlap:.4f}" if lm.spectral_overlap is not None else "—",
                f"{lm.effective_rank_ratio:.4f}" if lm.effective_rank_ratio is not None else "—",
            )

        console.print(table)

        if result.conflict_zones:
            console.print()
            console.print("[bold]Conflict Zones[/bold]")
            for zone in result.conflict_zones:
                color = _severity_color(zone.severity.value)
                console.print(
                    Panel(
                        f"Layers {zone.start_layer}-{zone.end_layer} "
                        f"({len(zone.layer_names)} layers)\n"
                        f"Avg cosine sim: {zone.avg_cosine_sim:.4f}\n"
                        f"[{color}]{zone.recommendation}[/{color}]",
                        title=f"[{color}]{zone.severity.value.upper()}[/{color}]",
                        border_style=color.replace("bold ", ""),
                    )
                )

        if result.strategy:
            console.print()
            console.print(
                Panel(
                    f"[bold]Method:[/bold] {result.strategy.method.value}\n"
                    f"[bold]Confidence:[/bold] {result.strategy.confidence:.0%}\n\n"
                    f"{result.strategy.reasoning}\n\n"
                    f"[bold]MergeKit Config:[/bold]\n```yaml\n{result.strategy.mergekit_yaml}```",
                    title="[bold green]Strategy Recommendation[/bold green]",
                    border_style="green",
                )
            )
            if result.strategy.warnings:
                for w in result.strategy.warnings:
                    console.print(f"  [yellow]⚠ {w}[/yellow]")

        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(result.model_dump_json(indent=2))
            console.print(f"\n[green]Results saved to {output_json}[/green]")

        if report:
            try:
                from mergelens.report.generator import generate_report

                generate_report(result, output_path=str(report))
                console.print(f"\n[green]HTML report saved to {report}[/green]")
            except ImportError:
                console.print("[yellow]Report generation requires jinja2 and plotly.[/yellow]")
    finally:
        cache.close()


@app.command()
def diagnose(
    config: Path = typer.Argument(..., help="Path to MergeKit YAML config."),
    device: str = typer.Option("cpu", "--device", "-d", help="Torch device."),
    report: Path | None = typer.Option(None, "--report", "-r", help="Save HTML report."),
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Save results as JSON."),
):
    """Diagnose a MergeKit config for potential issues before merging."""
    from mergelens.diagnose import diagnose_config

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]MergeLens Diagnose[/bold]\nConfig: {config}",
            title="[cyan]MergeLens[/cyan]",
            border_style="cyan",
        )
    )

    result = diagnose_config(str(config), device=device)

    console.print(f"\n[bold]Overall Interference:[/bold] {result.overall_interference:.4f}")
    console.print(f"[bold]Merge Method:[/bold] {result.config.merge_method.value}")

    if result.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in result.recommendations:
            console.print(f"  • {rec}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(result.model_dump_json(indent=2))
        console.print(f"\n[green]Results saved to {output_json}[/green]")


@app.command()
def audit(
    base_model: str = typer.Argument(..., help="Base model path or HF repo ID."),
    merged_model: str = typer.Argument(..., help="Merged model path or HF repo ID."),
    categories: list[str] | None = typer.Option(
        None, "--category", "-c", help="Probe categories to run."
    ),
    device: str = typer.Option("cpu", "--device", "-d", help="Torch device."),
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Save results as JSON."),
):
    """Audit a merged model's capabilities against its base."""
    console.print(
        Panel(
            f"[bold]MergeLens Audit[/bold]\nBase: {base_model}\nMerged: {merged_model}",
            title="[cyan]MergeLens[/cyan]",
            border_style="cyan",
        )
    )
    console.print("[yellow]The audit command is not yet implemented.[/yellow]")
    console.print(
        "Capability auditing (probe-based regression testing against a base model) "
        "is planned for a future release."
    )
    raise typer.Exit(1)


@app.command()
def serve(
    transport: str = typer.Option("stdio", "--transport", "-t", help="MCP transport (stdio/sse)."),
):
    """Start the MergeLens MCP server for AI assistant integration."""
    try:
        from mergelens.mcp.server import create_server

        server = create_server()
        console.print("[green]Starting MergeLens MCP server...[/green]")
        server.run(transport=transport)
    except ImportError:
        console.print("[yellow]MCP server requires mergelens[mcp] extra.[/yellow]")
        console.print("Install with: pip install mergelens[mcp]")
        raise typer.Exit(1)


def version_callback(value: bool):
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
):
    """MergeLens — Pre-merge diagnostics for LLM model merging."""


if __name__ == "__main__":
    app()
