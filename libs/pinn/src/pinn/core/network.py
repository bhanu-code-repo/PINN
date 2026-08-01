"""PINN network backbone — fully-connected MLP with smooth activations.

Copyright 2026 Bhanu Thakur. All rights reserved.
"""

import torch
import torch.nn as nn

# Supported activation functions. All are smooth (infinitely differentiable),
# which is required for PINN losses that differentiate the output w.r.t.
# inputs via autograd — often twice (second-order PDE residuals).
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class PINN(nn.Module):
    """Fully-connected multilayer perceptron backbone for Physics-Informed Neural Networks.

    Architecture::

        Linear(input_dim -> hidden_neurons) -> Activation
        -> [Linear(hidden_neurons -> hidden_neurons) -> Activation] x (hidden_layers - 1)
        -> Linear(hidden_neurons -> output_dim)

    ``tanh`` is the default activation and is used deliberately: PINN losses
    differentiate the network output with respect to its *inputs* (often twice),
    so the activation must be smooth and infinitely differentiable. Piecewise-linear
    activations such as ReLU have a zero second derivative and break second-order
    residuals.

    Other supported activations: ``silu``, ``gelu``.

    Args:
        input_dim: Number of input coordinates, e.g. ``1`` for ``u(t)`` or
            ``2`` for ``u(x, t)``.
        hidden_layers: Number of hidden layers (must be >= 1).
        hidden_neurons: Width of each hidden layer (must be >= 1).
        output_dim: Number of output channels. Use ``1`` for scalar fields, or
            ``2`` to represent e.g. the real/imaginary parts of a complex field.
        activation: Activation function name. One of ``'tanh'``, ``'silu'``,
            ``'gelu'``. Default: ``'tanh'``.

    Raises:
        ValueError: If any dimension argument is < 1 or activation is unknown.

    Example:
        >>> model = PINN(input_dim=2, hidden_layers=4, hidden_neurons=64)
        >>> xt = torch.rand(100, 2, requires_grad=True)
        >>> u = model(xt)          # shape: (100, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int,
        hidden_neurons: int,
        output_dim: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()

        # --- Input validation ---
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")
        if hidden_neurons < 1:
            raise ValueError(f"hidden_neurons must be >= 1, got {hidden_neurons}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")

        activation_lower = activation.lower()
        if activation_lower not in _ACTIVATIONS:
            valid = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(
                f"Unknown activation {activation!r}. Choose from: {valid}"
            )
        act_cls = _ACTIVATIONS[activation_lower]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(act_cls())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden_neurons, output_dim))
        self.network = nn.Sequential(*layers)

        # Xavier uniform initialisation — good default for tanh/sigmoid,
        # reasonable for other smooth activations.
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all Linear layers."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Args:
            x: Input tensor of shape ``(N, input_dim)``. Set ``requires_grad=True``
                on inputs that appear in differential-operator losses.

        Returns:
            Output tensor of shape ``(N, output_dim)``.
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        linears = [m for m in self.network if isinstance(m, nn.Linear)]
        acts = [m for m in self.network if not isinstance(m, nn.Linear)]
        act_name = type(acts[0]).__name__ if acts else "None"
        dims = " -> ".join(
            [str(linears[0].in_features)]
            + [str(layer.out_features) for layer in linears]
        )
        n_params = self.count_parameters()
        return (
            f"PINN({dims}, activation={act_name}, "
            f"params={n_params:,})"
        )
