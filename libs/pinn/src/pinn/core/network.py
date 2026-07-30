import torch
import torch.nn as nn


class PINN(nn.Module):
    """Fully-connected multilayer perceptron backbone for Physics-Informed Neural Networks.

    Architecture::

        Linear(input_dim -> hidden_neurons) -> Tanh
        -> [Linear(hidden_neurons -> hidden_neurons) -> Tanh] x (hidden_layers - 1)
        -> Linear(hidden_neurons -> output_dim)

    ``tanh`` is used deliberately: PINN losses differentiate the network output with
    respect to its *inputs* (often twice), so the activation must be smooth and
    infinitely differentiable. Piecewise-linear activations such as ReLU have a zero
    second derivative and break second-order residuals.

    Args:
        input_dim: Number of input coordinates, e.g. ``1`` for ``u(t)`` or
            ``2`` for ``u(x, t)``.
        hidden_layers: Number of hidden layers (must be >= 1).
        hidden_neurons: Width of each hidden layer.
        output_dim: Number of output channels. Use ``1`` for scalar fields, or
            ``2`` to represent e.g. the real/imaginary parts of a complex field.

    Example:
        >>> model = PINN(input_dim=2, hidden_layers=4, hidden_neurons=64)
        >>> xt = torch.rand(100, 2, requires_grad=True)
        >>> u = model(xt)          # shape: (100, 1)
    """

    def __init__(self, input_dim: int, hidden_layers: int, hidden_neurons: int, output_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Args:
            x: Input tensor of shape ``(N, input_dim)``. Set ``requires_grad=True``
                on inputs that appear in differential-operator losses.

        Returns:
            Output tensor of shape ``(N, output_dim)``.
        """
        return self.network(x)
