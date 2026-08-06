"""Lang-PINN CLI — solve differential equations from natural language.

Usage::

    uv run lang-pinn solve "u'' + 2u' + 6400u = 0, u(0)=1, u'(0)=0 on [0,1]"
    uv run lang-pinn solve "Burgers equation" --mode library --save-code
    uv run lang-pinn solve "heat equation on [0,1]x[0,1]" --execute
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .orchestrator import Orchestrator

app = typer.Typer(
    name="lang-pinn",
    help="Lang-PINN: solve differential equations with LLM-guided PINNs.",
    no_args_is_help=True,
)

console = Console()


def _print_spec_table(result) -> None:
    """Print parsed PDE specification as a rich table."""
    spec = result.spec
    table = Table(title="Parsed PDE Specification", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Name", spec.name)
    table.add_row("Equation", spec.equation)
    table.add_row("Variables", " → ".join(spec.independent_vars) + f" → {spec.dependent_var}")
    table.add_row("Order", str(spec.order))
    table.add_row("Spatial dim", str(spec.spatial_dim))
    table.add_row("Domain", str(spec.domain))
    if spec.initial_conditions:
        table.add_row("ICs", ", ".join(spec.initial_conditions))
    if spec.boundary_conditions:
        table.add_row("BCs", ", ".join(spec.boundary_conditions))
    if spec.parameters:
        table.add_row("Parameters", ", ".join(f"{k}={v}" for k, v in spec.parameters.items()))

    features = []
    if spec.has_high_frequency:
        features.append("high-frequency")
    if spec.has_sharp_gradients:
        features.append("sharp-gradients")
    if spec.has_periodic_bc:
        features.append("periodic-BC")
    if not spec.is_linear:
        features.append("nonlinear")
    if features:
        table.add_row("Features", ", ".join(features))

    console.print(table)


def _print_arch_table(result) -> None:
    """Print architecture recommendation as a rich table."""
    arch = result.architecture
    table = Table(title="Architecture Recommendation", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Network", f"{arch.hidden_layers}×{arch.hidden_neurons} {arch.activation}")
    table.add_row("Input → Output", f"{arch.input_dim} → {arch.output_dim}")
    table.add_row("Learning rate", str(arch.learning_rate))
    table.add_row("Epochs", str(arch.epochs))
    table.add_row("Collocation points", str(arch.n_collocation))
    table.add_row("Loss weights", str(arch.loss_weights))
    if arch.use_ansatz:
        table.add_row("Ansatz", arch.ansatz_type or "yes")
    table.add_row("Reasoning", arch.reasoning)

    console.print(table)


@app.command()
def solve(
    description: str = typer.Argument(..., help="Natural language PDE description."),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m",
        help="Operating mode: library, code-agent, or hybrid.",
    ),
    execute: bool = typer.Option(
        False, "--execute", "-x",
        help="Execute the generated code after generation.",
    ),
    max_iterations: int = typer.Option(
        3, "--max-iter",
        help="Max refinement iterations (hybrid mode only).",
    ),
    save_code: bool = typer.Option(
        False, "--save-code", "-s",
        help="Save generated code to a .py file.",
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Directory to save artifacts (default: outputs/lang_pinn/<timestamp>).",
    ),
    show_code: bool = typer.Option(
        True, "--show-code/--no-code",
        help="Display the generated code.",
    ),
    verify: bool = typer.Option(
        True, "--verify/--no-verify",
        help="Verify PDE parse with SymPy (if available).",
    ),
):
    """Parse a PDE from natural language, recommend architecture, generate code."""
    from pinn import setup_logging

    setup_logging()

    console.print(Panel(
        f"[bold]Lang-PINN[/bold] — mode: [cyan]{mode}[/cyan]",
        subtitle=f"[dim]{description[:80]}{'...' if len(description) > 80 else ''}[/dim]",
    ))

    # Build orchestrator and solve
    orch = Orchestrator(mode=mode)
    result = orch.solve(
        description,
        execute=execute,
        max_iterations=max_iterations,
    )

    # Display parsed PDE
    _print_spec_table(result)

    # SymPy verification
    if verify:
        try:
            from .sympy_verify import verify_spec
            issues = verify_spec(result.spec)
            if issues:
                console.print(Panel(
                    "\n".join(f"[yellow]⚠[/yellow] {issue}" for issue in issues),
                    title="[yellow]SymPy Verification Warnings[/yellow]",
                ))
            else:
                console.print("[green]✓[/green] SymPy verification passed")
        except ImportError:
            logger.debug("SymPy not available, skipping verification")
        except Exception as e:
            logger.warning("SymPy verification failed: {}", e)

    # Display architecture
    _print_arch_table(result)

    # Display generated code
    if show_code:
        syntax = Syntax(result.code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated Code"))

    # Execution results
    if result.executed:
        if result.error:
            console.print(Panel(
                f"[red]{result.error}[/red]",
                title="[red]Execution Error[/red]",
            ))
        else:
            parts = []
            if result.quality_score is not None:
                parts.append(f"Quality score: [green]{result.quality_score:.3f}[/green]")
            if result.iterations > 1:
                parts.append(f"Iterations: {result.iterations}")
            console.print(Panel(
                "\n".join(parts) if parts else "[green]Execution completed[/green]",
                title="[green]Execution Results[/green]",
            ))

    # Save artifacts
    if save_code or output_dir:
        out = _resolve_output_dir(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save code
        code_path = out / "generated_experiment.py"
        code_path.write_text(result.code)
        console.print(f"Code saved to: [cyan]{code_path}[/cyan]")

        # Save spec as JSON
        spec_path = out / "pde_spec.json"
        spec_data = {
            "name": result.spec.name,
            "equation": result.spec.equation,
            "independent_vars": result.spec.independent_vars,
            "dependent_var": result.spec.dependent_var,
            "order": result.spec.order,
            "spatial_dim": result.spec.spatial_dim,
            "domain": {k: list(v) for k, v in result.spec.domain.items()},
            "initial_conditions": result.spec.initial_conditions,
            "boundary_conditions": result.spec.boundary_conditions,
            "parameters": result.spec.parameters,
        }
        spec_path.write_text(json.dumps(spec_data, indent=2))

        # Save architecture
        arch_path = out / "architecture.json"
        arch_data = {
            "input_dim": result.architecture.input_dim,
            "output_dim": result.architecture.output_dim,
            "hidden_layers": result.architecture.hidden_layers,
            "hidden_neurons": result.architecture.hidden_neurons,
            "activation": result.architecture.activation,
            "learning_rate": result.architecture.learning_rate,
            "epochs": result.architecture.epochs,
            "use_ansatz": result.architecture.use_ansatz,
            "ansatz_type": result.architecture.ansatz_type,
            "loss_weights": result.architecture.loss_weights,
            "n_collocation": result.architecture.n_collocation,
            "reasoning": result.architecture.reasoning,
        }
        arch_path.write_text(json.dumps(arch_data, indent=2))

        console.print(f"Spec saved to: [cyan]{spec_path}[/cyan]")
        console.print(f"Architecture saved to: [cyan]{arch_path}[/cyan]")


@app.command()
def parse(
    description: str = typer.Argument(..., help="Natural language PDE description."),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="SymPy verification."),
):
    """Parse a PDE description without generating code (PDE Agent only)."""
    from pinn import setup_logging

    setup_logging()

    from .agents.pde_agent import PDEAgent

    agent = PDEAgent()
    spec = agent.parse(description)

    # Reuse the table printer
    from .orchestrator import SolveResult
    from .schemas import ArchitectureRec
    dummy = SolveResult(
        spec=spec,
        architecture=ArchitectureRec(input_dim=0, output_dim=0, hidden_layers=0, hidden_neurons=0),
        code="", mode="parse-only",
    )
    _print_spec_table(dummy)

    if verify:
        try:
            from .sympy_verify import verify_spec
            issues = verify_spec(spec)
            if issues:
                for issue in issues:
                    console.print(f"[yellow]⚠[/yellow] {issue}")
            else:
                console.print("[green]✓[/green] SymPy verification passed")
        except Exception as e:
            logger.warning("SymPy verification: {}", e)


@app.command()
def recommend(
    description: str = typer.Argument(..., help="Natural language PDE description."),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for architecture advice."),
):
    """Parse PDE and recommend architecture (PDE + PINN Agents)."""
    from pinn import setup_logging

    setup_logging()

    from .agents.pde_agent import PDEAgent
    from .agents.pinn_agent import PINNAgent

    pde_agent = PDEAgent()
    spec = pde_agent.parse(description)

    pinn_agent = PINNAgent()
    arch = pinn_agent.recommend(spec, use_llm=use_llm)

    from .orchestrator import SolveResult
    result = SolveResult(spec=spec, architecture=arch, code="", mode="recommend-only")
    _print_spec_table(result)
    _print_arch_table(result)


def _resolve_output_dir(output_dir: str | None) -> Path:
    """Resolve output directory, creating timestamped default if needed."""
    if output_dir:
        return Path(output_dir)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "lang_pinn" / timestamp


if __name__ == "__main__":
    app()
