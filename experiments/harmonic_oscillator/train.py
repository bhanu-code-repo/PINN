#!/usr/bin/env python3
"""
Harmonic Oscillator PINN Training Script

This script trains a Physics-Informed Neural Network (PINN) to solve the
damped harmonic oscillator ODE:
    u'' + mu*u' + k*u = 0

It uses an Ansatz formulation to handle high-frequency oscillations effectively.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from pyfiglet import Figlet
from pathlib import Path
from typing import Optional

# Import from your local library
from pinn.core.network import PINN
from pinn.trainer.trainer import PINNTrainer

# Initialize Rich console
console = Console()
app = typer.Typer(help="Train a PINN for the Damped Harmonic Oscillator.")

def show_banner():
    """Display a startup banner using pyfiglet and rich."""
    f = Figlet(font="slant")
    banner_text = f.renderText("PINN")
    console.print(f"[bold cyan]{banner_text}[/bold cyan]")
    console.print("[bold yellow]Damped Harmonic Oscillator PINN Solver[/bold yellow]")
    console.print("=" * 50)

def solve_harmonic_oscillator(
    epochs: int = 15000,
    lr: float = 1e-3,
    hidden_neurons: int = 32,
    hidden_layers: int = 3,
    w0: float = 80.0,
    d: float = 2.0,
    save_plot: bool = False,
    plot_path: Optional[str] = None,
):
    """
    Trains a PINN to solve the damped harmonic oscillator.
    
    Args:
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        hidden_neurons: Number of neurons per hidden layer.
        hidden_layers: Number of hidden layers.
        w0: Natural frequency of the oscillator.
        d: Damping coefficient.
        save_plot: Whether to save the final plot to disk.
        plot_path: Path to save the plot (if save_plot is True).
    """
    
    # 1. Problem Setup
    mu, k = 2*d, w0**2
    t_domain = (0.0, 1.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Define the PDE Residual
    def pde_residual(model, t):
        u = model(t)
        u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_tt = autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
        return u_tt + mu * u_t + k * u

    # 3. Define Loss Functions
    t_ic = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    t_physics = torch.linspace(t_domain[0], t_domain[1], 100).view(-1, 1).to(device).requires_grad_(True)

    def ic_loss(model):
        u = model(t_ic)
        u_t = autograd.grad(u, t_ic, torch.ones_like(u), create_graph=True)[0]
        return (u - 1.0)**2 + (u_t - 0.0)**2

    def physics_loss(model):
        return torch.mean(pde_residual(model, t_physics)**2)

    # 4. Create Model & Ansatz
    console.print("[bold cyan]Building Neural Network...[/bold cyan]")
    pinn = PINN(input_dim=1, hidden_layers=hidden_layers, hidden_neurons=hidden_neurons)
    
    class Ansatz(nn.Module):
        def __init__(self, pinn):
            super().__init__()
            self.pinn = pinn
            self.a = nn.Parameter(torch.tensor(70.0, device=device, requires_grad=True))
            self.b = nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))
        def forward(self, t):
            return self.pinn(t) * torch.sin(self.a * t + self.b)
    
    model = Ansatz(pinn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)
    
    # 5. Train the Model
    console.print(f"[bold green]Starting Training for {epochs} epochs...[/bold green]")
    console.print(f"    - Hidden Layers: {hidden_layers}")
    console.print(f"    - Neurons per Layer: {hidden_neurons}")
    console.print(f"    - Learning Rate: {lr}")
    console.print("=" * 50)
    
    trainer.train(n_epochs=epochs, optimizer=optimizer,
                  loss_functions={'ic': ic_loss, 'physics': physics_loss},
                  weights={'ic': 0.1, 'physics': 1e-4})
    
    console.print("[bold green]Training Complete![/bold green]")

    # 6. Visualize Results
    t_test = torch.linspace(t_domain[0], t_domain[1], 300).view(-1, 1).to(device)
    
    with torch.no_grad():
        u_pinn = model(t_test).cpu().numpy()
    
    def exact_solution(d, w0, t):
        w = np.sqrt(w0**2 - d**2)
        phi = np.arctan(-d/w)
        A = 1/(2*np.cos(phi))
        return np.exp(-d*t) * 2 * A * np.cos(phi + w*t)
    
    t_test_np = t_test.cpu().numpy()
    u_exact = exact_solution(d, w0, t_test_np)

    plt.figure(figsize=(10, 6))
    plt.plot(t_test_np, u_pinn, 'r-', label='PINN Solution', linewidth=2, alpha=0.5)
    plt.plot(t_test_np, u_exact, 'k--', label='Exact Solution', linewidth=2, alpha=0.9)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title(f'High-Frequency Damped Harmonic Oscillator (w0={w0})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plot and plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]Plot saved to: {plot_path}[/bold green]")
    
    plt.show()

    # 7. Print a summary table
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Final Loss", f"{trainer.loss_history[-1]['total']:.4e}")
    table.add_row("Learned 'a' (Frequency)", f"{model.a.item():.4f}")
    table.add_row("Learned 'b' (Phase)", f"{model.b.item():.4f}")
    table.add_row("Epochs Run", str(len(trainer.loss_history)))
    console.print(table)

@app.command()
def train(
    epochs: int = typer.Option(15000, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(32, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(3, "--layers", "-l", help="Number of hidden layers."),
    w0: float = typer.Option(80.0, "--w0", help="Natural frequency of the oscillator."),
    damping: float = typer.Option(2.0, "--damping", "-d", help="Damping coefficient."),
    save_plot: bool = typer.Option(False, "--save-plot", help="Save the final plot to disk."),
    plot_path: Optional[str] = typer.Option(None, "--plot-path", help="Path to save the plot."),
):
    """
    Train a PINN to solve the damped harmonic oscillator.
    """
    show_banner()
    solve_harmonic_oscillator(
        epochs=epochs,
        lr=lr,
        hidden_neurons=neurons,
        hidden_layers=layers,
        w0=w0,
        d=damping,
        save_plot=save_plot,
        plot_path=plot_path,
    )

if __name__ == "__main__":
    app()