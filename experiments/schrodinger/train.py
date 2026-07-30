#!/usr/bin/env python3
"""
Schrödinger Equation PINN Training Script

This script trains a Physics-Informed Neural Network (PINN) to solve the
1D nonlinear Schrödinger equation:
    i*h_t + 0.5*h_xx + |h|^2*h = 0

This equation describes wave propagation in nonlinear media, such as optical
fibers and Bose-Einstein condensates. The PINN handles complex-valued solutions
and periodic boundary conditions.
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
app = typer.Typer(help="Train a PINN for the Schrödinger Equation.")

def show_banner():
    """Display a startup banner using pyfiglet and rich."""
    f = Figlet(font="slant")
    banner_text = f.renderText("SCHRODINGER")
    console.print(f"[bold cyan]{banner_text}[/bold cyan]")
    console.print("[bold yellow]1D Nonlinear Schrödinger Equation PINN Solver[/bold yellow]")
    console.print("=" * 50)

def solve_schrodinger_equation(
    epochs: int = 25000,
    lr: float = 5e-4,
    hidden_neurons: int = 100,
    hidden_layers: int = 4,
    save_plot: bool = False,
    plot_path: Optional[str] = None,
):
    """
    Trains a PINN to solve the 1D nonlinear Schrödinger equation.
    
    Args:
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        hidden_neurons: Number of neurons per hidden layer.
        hidden_layers: Number of hidden layers.
        save_plot: Whether to save the final plot to disk.
        plot_path: Path to save the plot (if save_plot is True).
    """
    
    # 1. Problem Setup
    x_domain = (-5.0, 5.0)
    t_domain = (0.0, np.pi/2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Define the PDE Residual (Complex)
    def pde_residual(model, x, t):
        u, v = model(x, t) # u and v are the real and imag parts
        h = u + 1j * v
        h_conj = u - 1j * v
        
        # Gradients on real and imag parts
        u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        v_t = autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        v_x = autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        v_xx = autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        
        # Combine into complex derivatives
        h_t = u_t + 1j * v_t
        h_xx = u_xx + 1j * v_xx
        
        # Residual: i*h_t + 0.5*h_xx + |h|^2*h = 0
        f = 1j * h_t + 0.5 * h_xx + (h * h_conj) * h
        return torch.mean(torch.abs(f)**2)

    # 3. Define Loss Functions
    x_ic = torch.linspace(x_domain[0], x_domain[1], 100).view(-1, 1).to(device).requires_grad_(True)
    t_bc = torch.linspace(t_domain[0], t_domain[1], 50).view(-1, 1).to(device)
    
    # IMPROVEMENT: Increased collocation points from 2000 to 5000
    x_physics = torch.rand(5000, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
    t_physics = torch.rand(5000, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
    x_physics = x_physics.to(device).requires_grad_(True)
    t_physics = t_physics.to(device).requires_grad_(True)

    def ic_loss(model):
        u, v = model(x_ic, torch.zeros_like(x_ic))
        # IC: h(0, x) = 2 * sech(x)
        h_exact = 2 / torch.cosh(x_ic)
        return torch.mean((u - h_exact)**2 + v**2)

    def bc_loss(model):
        # Periodic BC: h(t, -5) = h(t, 5) and h_x(t, -5) = h_x(t, 5)
        x_boundary_val_l = -5 * torch.ones_like(t_bc).requires_grad_(True)
        x_boundary_val_r = 5 * torch.ones_like(t_bc).requires_grad_(True)

        u_l, v_l = model(x_boundary_val_l, t_bc)
        u_r, v_r = model(x_boundary_val_r, t_bc)
        
        u_l_x = autograd.grad(u_l, x_boundary_val_l, torch.ones_like(u_l), create_graph=True)[0]
        v_l_x = autograd.grad(v_l, x_boundary_val_l, torch.ones_like(v_l), create_graph=True)[0]
        u_r_x = autograd.grad(u_r, x_boundary_val_r, torch.ones_like(u_r), create_graph=True)[0]
        v_r_x = autograd.grad(v_r, x_boundary_val_r, torch.ones_like(v_r), create_graph=True)[0]
        
        loss_period = torch.mean((u_l - u_r)**2 + (v_l - v_r)**2)
        loss_period_x = torch.mean((u_l_x - u_r_x)**2 + (v_l_x - v_r_x)**2)
        return loss_period + loss_period_x

    def physics_loss(model):
        return pde_residual(model, x_physics, t_physics)

    # 4. Create Model
    class ComplexPINN(nn.Module):
        def __init__(self, input_dim, hidden_layers, hidden_neurons):
            super().__init__()
            self.network = PINN(input_dim, hidden_layers, hidden_neurons, output_dim=2)
        def forward(self, x, t):
            xt = torch.cat([x, t], dim=1)
            out = self.network(xt)
            return out[:, 0:1], out[:, 1:2]

    console.print("[bold cyan]Building Complex Neural Network...[/bold cyan]")
    console.print(f"    - Input Dim: 2 (x, t)")
    console.print(f"    - Hidden Layers: {hidden_layers}")
    console.print(f"    - Neurons per Layer: {hidden_neurons}")
    
    model = ComplexPINN(input_dim=2, hidden_layers=hidden_layers, hidden_neurons=hidden_neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)
    
    # 5. Train
    console.print(f"[bold green]Starting Training for {epochs} epochs...[/bold green]")
    
    trainer.train(n_epochs=epochs, optimizer=optimizer,
                  loss_functions={'ic': ic_loss, 'bc': bc_loss, 'physics': physics_loss},
                  weights={'ic': 1.0, 'bc': 1.0, 'physics': 1.0})
    
    console.print("[bold green]Training Complete![/bold green]")

    # 6. Visualize Results
    x_test = torch.linspace(x_domain[0], x_domain[1], 200).view(-1, 1).to(device)
    t_test = torch.linspace(t_domain[0], t_domain[1], 100).view(-1, 1).to(device)
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    
    with torch.no_grad():
        u_pred, v_pred = model(X.flatten().unsqueeze(1), T.flatten().unsqueeze(1))
        h_mag = torch.sqrt(u_pred**2 + v_pred**2).cpu().numpy().reshape(200, 100)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(T.cpu().numpy(), X.cpu().numpy(), h_mag, 20, cmap='viridis')
    plt.colorbar(label='|h(t,x)|')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title("PINN Solution Magnitude for Schrödinger Equation")
    
    if save_plot and plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]Contour plot saved to: {plot_path}[/bold green]")
    
    plt.show()

    # 7. Validation Test (Snapshot at t=0)
    console.print("\n[bold cyan]Running Validation Tests...[/bold cyan]")
    
    with torch.no_grad():
        u_pred_0, v_pred_0 = model(x_test, torch.zeros_like(x_test))
        h_mag_0 = torch.sqrt(u_pred_0**2 + v_pred_0**2).cpu().numpy()
        
    h_exact_0 = (2 / torch.cosh(x_test)).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x_test.cpu().numpy(), h_exact_0, 'k-', label='Exact (t=0)', linewidth=2)
    plt.plot(x_test.cpu().numpy(), h_mag_0, 'r--', label='PINN (t=0)', linewidth=2)
    plt.title("Comparison at t = 0 (Initial Condition)")
    plt.xlabel('x')
    plt.ylabel('|h(0,x)|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plot and plot_path:
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
    epochs: int = typer.Option(25000, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(5e-4, "--lr", help="Learning rate."),
    neurons: int = typer.Option(100, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(4, "--layers", "-l", help="Number of hidden layers."),
    save_plot: bool = typer.Option(False, "--save-plot", help="Save the final plots to disk."),
    plot_path: Optional[str] = typer.Option(None, "--plot-path", help="Path to save the plots."),
):
    """
    Train a PINN to solve the 1D nonlinear Schrödinger equation.
    """
    show_banner()
    solve_schrodinger_equation(
        epochs=epochs,
        lr=lr,
        hidden_neurons=neurons,
        hidden_layers=layers,
        save_plot=save_plot,
        plot_path=plot_path,
    )

if __name__ == "__main__":
    app()