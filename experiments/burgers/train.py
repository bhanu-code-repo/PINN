#!/usr/bin/env python3
"""
Burgers' Equation PINN Training Script

This script trains a Physics-Informed Neural Network (PINN) to solve the
1D Burgers' equation:
    u_t + u*u_x - nu*u_xx = 0

This equation is a classic example of a hyperbolic conservation law that forms
sharp shocks, which are notoriously difficult for traditional numerical methods.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import typer
from rich.console import Console
from rich.table import Table
from pyfiglet import Figlet
from typing import Optional

# Import from your local library
from pinn.core.network import PINN
from pinn.trainer.trainer import PINNTrainer

# Initialize Rich console
console = Console()
app = typer.Typer(help="Train a PINN for Burgers' Equation.")

def show_banner():
    """Display a startup banner using pyfiglet and rich."""
    f = Figlet(font="slant")
    banner_text = f.renderText("BURGERS")
    console.print(f"[bold cyan]{banner_text}[/bold cyan]")
    console.print("[bold yellow]1D Burgers' Equation PINN Solver[/bold yellow]")
    console.print("=" * 50)

def solve_burgers_equation(
    epochs: int = 30000,
    lr: float = 1e-3,
    hidden_neurons: int = 50,
    hidden_layers: int = 5,
    nu: float = 0.01 / np.pi,
    save_plot: bool = False,
    plot_path: Optional[str] = None,
):
    """
    Trains a PINN to solve the 1D Burgers' equation.
    
    Args:
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        hidden_neurons: Number of neurons per hidden layer.
        hidden_layers: Number of hidden layers.
        nu: Viscosity coefficient.
        save_plot: Whether to save the final plot to disk.
        plot_path: Path to save the plot (if save_plot is True).
    """
    
    # 1. Problem Setup
    x_domain = (-1.0, 1.0)
    t_domain = (0.0, 1.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Define the PDE Residual
    def pde_residual(model, x, t):
        xt = torch.cat([x, t], dim=1)
        u = model(xt)
        
        u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        
        return u_t + u * u_x - nu * u_xx

    # 3. Define Loss Functions
    x_ic = torch.linspace(x_domain[0], x_domain[1], 100).view(-1, 1).to(device).requires_grad_(True)
    t_bc = torch.linspace(t_domain[0], t_domain[1], 50).view(-1, 1).to(device).requires_grad_(True)
    
    x_physics = torch.rand(5000, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
    t_physics = torch.rand(5000, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
    x_physics = x_physics.to(device).requires_grad_(True)
    t_physics = t_physics.to(device).requires_grad_(True)

    def ic_loss(model):
        xt = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
        u = model(xt)
        u_exact = -torch.sin(np.pi * x_ic)
        return torch.mean((u - u_exact)**2)

    def bc_loss(model):
        xt_left = torch.cat([-torch.ones_like(t_bc), t_bc], dim=1)
        u_left = model(xt_left)
        xt_right = torch.cat([torch.ones_like(t_bc), t_bc], dim=1)
        u_right = model(xt_right)
        return torch.mean(u_left**2 + u_right**2)

    def physics_loss(model):
        return torch.mean(pde_residual(model, x_physics, t_physics)**2)

    # 4. Create Model and Trainer
    console.print("[bold cyan]Building Neural Network...[/bold cyan]")
    console.print(f"    - Input Dim: 2 (x, t)")
    console.print(f"    - Hidden Layers: {hidden_layers}")
    console.print(f"    - Neurons per Layer: {hidden_neurons}")
    
    model = PINN(input_dim=2, hidden_layers=hidden_layers, hidden_neurons=hidden_neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)
    
    # 5. Train the Model
    console.print(f"[bold green]Starting Training for {epochs} epochs...[/bold green]")
    
    trainer.train(n_epochs=epochs, optimizer=optimizer,
                  loss_functions={'ic': ic_loss, 'bc': bc_loss, 'physics': physics_loss},
                  weights={'ic': 1.0, 'bc': 1.0, 'physics': 1.0})
    
    console.print("[bold green]Training Complete![/bold green]")

    # 6. Visualize Results (Contour Plot)
    x_test = torch.linspace(x_domain[0], x_domain[1], 200).view(-1, 1).to(device)
    t_test = torch.linspace(t_domain[0], t_domain[1], 200).view(-1, 1).to(device)
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    
    with torch.no_grad():
        xt_test = torch.stack([X.flatten(), T.flatten()], dim=1)
        u_pred = model(xt_test).cpu().numpy().reshape(200, 200)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(T.cpu().numpy(), X.cpu().numpy(), u_pred, 20, cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title("PINN Solution for Burgers' Equation")
    
    if save_plot and plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]Contour plot saved to: {plot_path}[/bold green]")
    
    plt.show()

    # 7. Validation Test (Snapshots at t=0 and t=1)
    console.print("\n[bold cyan]Running Validation Tests...[/bold cyan]")
    
    with torch.no_grad():
        xt_0 = torch.cat([x_test, torch.zeros_like(x_test)], dim=1)
        xt_1 = torch.cat([x_test, torch.ones_like(x_test)], dim=1)
        u_pinn_0 = model(xt_0).cpu().numpy()
        u_pinn_1 = model(xt_1).cpu().numpy()

    u_exact_0 = -np.sin(np.pi * x_test.cpu().numpy())

    plt.figure(figsize=(12, 5))

    # Plot Initial Condition (t=0)
    plt.subplot(1, 2, 1)
    plt.plot(x_test.cpu().numpy(), u_exact_0, 'k-', label='Exact (t=0)', linewidth=2)
    plt.plot(x_test.cpu().numpy(), u_pinn_0, 'r--', label='PINN (t=0)', linewidth=2)
    plt.title("Snapshot at t = 0")
    plt.xlabel('x')
    plt.ylabel('u(0, x)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Final State (t=1) - Shock formation
    plt.subplot(1, 2, 2)
    plt.plot(x_test.cpu().numpy(), u_pinn_1, 'r-', label='PINN (t=1)', linewidth=2)
    plt.title("Snapshot at t = 1 (Steep Shock Formed)")
    plt.xlabel('x')
    plt.ylabel('u(1, x)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_plot and plot_path:
        # Save the validation plot with a suffix
        val_path = plot_path.replace('.png', '_validation.png')
        plt.savefig(val_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]Validation plot saved to: {val_path}[/bold green]")
    
    plt.show()

    # 8. Print a summary table
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Final Total Loss", f"{trainer.loss_history[-1]['total']:.4e}")
    table.add_row("Final IC Loss", f"{trainer.loss_history[-1]['ic']:.4e}")
    table.add_row("Final BC Loss", f"{trainer.loss_history[-1]['bc']:.4e}")
    table.add_row("Final Physics Loss", f"{trainer.loss_history[-1]['physics']:.4e}")
    table.add_row("Epochs Run", str(len(trainer.loss_history)))
    console.print(table)

@app.command()
def train(
    epochs: int = typer.Option(30000, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(50, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(5, "--layers", "-l", help="Number of hidden layers."),
    nu: float = typer.Option(0.01 / np.pi, "--nu", help="Viscosity coefficient."),
    save_plot: bool = typer.Option(False, "--save-plot", help="Save the final plots to disk."),
    plot_path: Optional[str] = typer.Option(None, "--plot-path", help="Path to save the plots."),
):
    """
    Train a PINN to solve the 1D Burgers' equation.
    """
    show_banner()
    solve_burgers_equation(
        epochs=epochs,
        lr=lr,
        hidden_neurons=neurons,
        hidden_layers=layers,
        nu=nu,
        save_plot=save_plot,
        plot_path=plot_path,
    )

if __name__ == "__main__":
    app()