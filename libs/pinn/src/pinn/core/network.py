import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: int, hidden_neurons: int, output_dim: int = 1):
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)