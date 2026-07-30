from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

# Signature of an epoch-end callback: (epoch, epoch_losses) -> None
EpochCallback = Callable[[int, dict[str, float]], None]


class PINNTrainer:
    """Generic trainer for PINN problems with weighted multi-term losses.

    The trainer is problem-agnostic: physics enters exclusively through the
    ``loss_functions`` dictionary passed to :meth:`train`. Each loss callable
    receives the model and returns a scalar tensor; collocation points are
    typically closed over by the callables themselves.

    Responsibilities:

    - full-batch optimisation loop over named, weighted loss terms
    - per-epoch loss history (``self.loss_history``)
    - progress bar (tqdm) and structured logging (loguru)
    - optional early stopping and gradient clipping
    - optional user callbacks per epoch (e.g. custom monitoring)
    - checkpoint save/load (model + optimizer + history)

    Plotting is deliberately *not* part of the training loop — use
    :meth:`plot_loss_history` after training, or a callback during it.

    Args:
        model: The model to train (any ``nn.Module``; moved to ``device``).
        device: Target device. Defaults to CUDA when available, else CPU.
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.loss_history: list[dict[str, float]] = []
        self._epoch = 0

    def train(
        self,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_functions: dict[str, Callable[[nn.Module], torch.Tensor]],
        weights: dict[str, float] | None = None,
        verbose: bool = True,
        log_every: int = 1000,
        early_stop_patience: int | None = None,
        early_stop_threshold: float = 1e-8,
        grad_clip: float | None = None,
        callbacks: list[EpochCallback] | None = None,
    ) -> list[dict[str, float]]:
        """Run the full-batch training loop.

        Args:
            n_epochs: Number of training epochs.
            optimizer: The PyTorch optimizer (already bound to model parameters).
            loss_functions: Mapping ``name -> fn(model) -> scalar tensor``. The
                total loss is ``sum(weights[name] * fn(model))``.
            weights: Per-term weights. Missing entries default to ``1.0``.
            verbose: Show a tqdm progress bar with the live total loss.
            log_every: Log an epoch summary (all loss terms + grad norm) every N
                epochs. ``0`` disables periodic logging.
            early_stop_patience: Stop if the total loss has not improved for N
                consecutive epochs. ``None`` disables early stopping.
            early_stop_threshold: Minimum decrease that counts as improvement.
            grad_clip: If set, clip the global gradient norm to this value.
            callbacks: Optional list of ``fn(epoch, epoch_losses)`` called at
                the end of every epoch (after the optimizer step).

        Returns:
            The loss history: one ``{name: value, ..., 'total': value}`` dict
            per epoch (also stored on ``self.loss_history``).
        """
        if weights is None:
            weights = {key: 1.0 for key in loss_functions}

        logger.info(
            "Starting training: {} epochs, loss terms {}, weights {}, device {}",
            n_epochs, list(loss_functions), weights, self.device,
        )

        best_loss = float("inf")
        early_stop_counter = 0
        best_epoch = 0

        pbar = tqdm(range(n_epochs), desc="Training", unit="epoch", disable=not verbose)

        for epoch in pbar:
            self._epoch = epoch
            optimizer.zero_grad()

            total_loss = torch.zeros((), device=self.device)
            epoch_losses: dict[str, float] = {}

            # 1. Compute weighted losses
            for name, loss_fn in loss_functions.items():
                loss_value = loss_fn(self.model)
                total_loss = total_loss + weights.get(name, 1.0) * loss_value
                epoch_losses[name] = loss_value.item()

            # 2. Backward pass
            total_loss.backward()

            # 3. Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()

            # 4. Log history
            epoch_losses["total"] = total_loss.item()
            self.loss_history.append(epoch_losses)

            # 5. Periodic epoch summary
            if log_every > 0 and epoch % log_every == 0:
                grad_norm = self._grad_norm()
                terms = ", ".join(f"{k}={v:.4e}" for k, v in epoch_losses.items())
                logger.debug("epoch {}/{} | {} | grad_norm={:.4e}",
                             epoch, n_epochs, terms, grad_norm)

            # 6. Early stopping
            if early_stop_patience is not None:
                if epoch_losses["total"] < best_loss - early_stop_threshold:
                    best_loss = epoch_losses["total"]
                    early_stop_counter = 0
                    best_epoch = epoch
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        logger.info(
                            "Early stop at epoch {}: no improvement for {} epochs "
                            "(best {:.4e} at epoch {})",
                            epoch, early_stop_patience, best_loss, best_epoch,
                        )
                        break

            # 7. Progress bar
            if verbose:
                pbar.set_postfix({"Loss": f"{epoch_losses['total']:.4e}"})

            # 8. User callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, epoch_losses)

        logger.info(
            "Training finished after {} epochs | final total loss {:.4e}",
            len(self.loss_history), self.loss_history[-1]["total"],
        )
        return self.loss_history

    def _grad_norm(self) -> float:
        """Global L2 norm of all parameter gradients (0.0 if no grads)."""
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total**0.5

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        metadata: dict | None = None,
    ) -> Path:
        """Save model weights, optimizer state, and loss history to ``path``.

        Args:
            path: Destination file (parent directories are created).
            optimizer: If given, its state is saved so training can resume.
            metadata: Arbitrary JSON-serialisable info (hyperparameters, seed,
                problem configuration, ...) stored alongside the states.

        Returns:
            The path the checkpoint was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "loss_history": self.loss_history,
            "epoch": self._epoch,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to {}", path)
        return path

    def load_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict:
        """Restore model (and optionally optimizer) state from a checkpoint.

        Args:
            path: Checkpoint file written by :meth:`save_checkpoint`.
            optimizer: If given and the checkpoint contains optimizer state,
                that state is restored into it.

        Returns:
            The checkpoint's ``metadata`` dict.
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.loss_history = checkpoint.get("loss_history", [])
        self._epoch = checkpoint.get("epoch", 0)
        logger.info("Checkpoint loaded from {} (epoch {})", path, self._epoch)
        return checkpoint.get("metadata", {})

    # --------------------------------------------------------------- plotting

    def plot_loss_history(
        self,
        show_total: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot the recorded loss history on a log scale (post-training).

        Args:
            show_total: Include the weighted total loss curve.
            save_path: If provided, save the figure there (300 dpi).
            show: Call ``plt.show()``. Set ``False`` for headless runs.
        """
        if not self.loss_history:
            logger.warning("No loss history to plot — has train() been called?")
            return

        plt.figure(figsize=(10, 6))

        loss_keys = [k for k in self.loss_history[0] if k != "total"]
        if show_total:
            loss_keys = ["total", *loss_keys]

        for key in loss_keys:
            values = [epoch_loss[key] for epoch_loss in self.loss_history]
            plt.plot(values, label=key, linewidth=1.5, alpha=0.8)

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Loss plot saved to {}", save_path)

        if show:
            plt.show()
        else:
            plt.close()
