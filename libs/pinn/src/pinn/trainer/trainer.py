import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, List
from tqdm import tqdm

class PINNTrainer:
    """
    A generic trainer for PINN problems with real-time monitoring and early stopping.
    """
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = model.to(self.device)
        self.loss_history: List[Dict[str, float]] = []
        self._epoch = 0

    def train(self, 
              n_epochs: int, 
              optimizer: torch.optim.Optimizer, 
              loss_functions: Dict[str, Callable], 
              weights: Optional[Dict[str, float]] = None,
              verbose: bool = True,
              plot_every: int = 1000,
              debug_every: int = 5000,
              early_stop_patience: Optional[int] = None,
              early_stop_threshold: float = 1e-8,
              grad_clip: Optional[float] = None):
        """
        Args:
            n_epochs: Number of training epochs.
            optimizer: The PyTorch optimizer.
            loss_functions: Dictionary of loss functions.
            weights: Dictionary of loss weights.
            verbose: Print epoch summaries.
            plot_every: Update the live loss plot every N epochs.
            debug_every: Print gradient info every N epochs.
            early_stop_patience: Stop if total loss doesn't improve for N epochs.
            early_stop_threshold: Minimum improvement to count as improvement.
            grad_clip: Clip gradients to this norm to prevent exploding.
        """
        if weights is None:
            weights = {key: 1.0 for key in loss_functions.keys()}
            
        best_loss = float('inf')
        early_stop_counter = 0
        best_epoch = 0
        
        # Setup live plot efficiently
        if plot_every > 0:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.lines = {}  # Store line objects for faster updates
        
        pbar = tqdm(range(n_epochs), desc="Training", unit="epoch", disable=not verbose)
        
        for epoch in pbar:
            self._epoch = epoch
            optimizer.zero_grad()
            
            total_loss = 0.0
            epoch_losses = {}
            
            # 1. Compute losses
            for name, loss_fn in loss_functions.items():
                loss_value = loss_fn(self.model)
                weighted_loss = weights.get(name, 1.0) * loss_value
                total_loss += weighted_loss
                epoch_losses[name] = loss_value.item()
            
            # 2. Backward pass
            total_loss.backward()
            
            # 3. Gradient Clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            # 4. Debug
            if debug_every > 0 and epoch % debug_every == 0 and epoch > 0:
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"\n[DEBUG] Epoch {epoch} | Grad Norm: {total_grad_norm:.4e}")
            
            optimizer.step()
            
            # 5. Log history
            epoch_losses['total'] = total_loss.item()
            self.loss_history.append(epoch_losses)
            
            # 6. Early Stopping
            if early_stop_patience is not None:
                if total_loss.item() < best_loss - early_stop_threshold:
                    best_loss = total_loss.item()
                    early_stop_counter = 0
                    best_epoch = epoch
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs. Stopping at epoch {epoch}.")
                        print(f"Best loss was {best_loss:.4e} at epoch {best_epoch}.")
                        break
            
            # 7. Update progress bar
            if verbose:
                pbar.set_postfix({"Loss": f"{total_loss.item():.4e}"})
            
            # 8. Live Plot (Optimized)
            if plot_every > 0 and epoch % plot_every == 0 and epoch > 0:
                loss_keys = [k for k in self.loss_history[0].keys() if k != 'total']
                
                # Clear and reset if first update
                if not self.lines:
                    self.ax.clear()
                    for key in loss_keys:
                        values = [epoch_loss[key] for epoch_loss in self.loss_history]
                        line, = self.ax.plot(values, label=key, linewidth=1.5, alpha=0.7)
                        self.lines[key] = line
                    self.ax.set_yscale('log')
                    self.ax.set_xlabel('Epoch')
                    self.ax.set_ylabel('Loss')
                    self.ax.set_title(f'Training Loss History (Epoch {epoch}/{n_epochs})')
                    self.ax.legend()
                    self.ax.grid(True, alpha=0.3)
                else:
                    # Update existing lines
                    for key, line in self.lines.items():
                        values = [epoch_loss[key] for epoch_loss in self.loss_history]
                        line.set_data(range(len(values)), values)
                    self.ax.relim()
                    self.ax.autoscale_view()
                
                plt.pause(0.001)
        
        if plot_every > 0:
            plt.ioff()
            plt.show()

    def plot_loss_history(self, show_total: bool = False, save_path: Optional[str] = None):
        """
        Plots the loss history after training completes.
        
        Args:
            show_total: Whether to include the total loss in the plot.
            save_path: If provided, saves the plot to this file path (e.g., 'loss_plot.png').
        """
        plt.figure(figsize=(10, 6))
        
        loss_keys = [k for k in self.loss_history[0].keys() if k != 'total']
        if show_total:
            loss_keys = ['total'] + loss_keys
        
        for key in loss_keys:
            values = [epoch_loss[key] for epoch_loss in self.loss_history]
            plt.plot(values, label=key, linewidth=1.5, alpha=0.8)
            
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
        plt.show()