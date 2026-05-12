"""Generic configurable MLP module."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Builds ``[Linear -> activation -> dropout] x N -> Linear(last_hidden, output_dim)``.

    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output.
        hidden_dims: List of hidden layer sizes. E.g. ``[128, 64]`` creates
            ``Linear(input_dim, 128) → act → drop → Linear(128, 64) → act → drop → Linear(64, output_dim)``.
        activation: Activation function class. Default: ``nn.GELU``.
        dropout: Dropout probability applied after each activation. Default: 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
