"""Task head construction utilities.

Builds per-task MLP heads from a task configuration dictionary.
"""

import torch.nn as nn

from michelangelo.lib.foundation_model.model.mlp import MLP


def build_task_heads(
    task_config: dict,
    d_model: int,
    dropout: float = 0.1,
) -> nn.ModuleDict:
    """Build per-task MLP heads from configuration.

    Each task entry in ``task_config`` requires:
        - ``num_classes`` (int): Output dimension.
        - ``hidden_dims`` (list[int]): Hidden layer sizes.

    Optional per-task overrides:
        - ``activation`` (str): Activation name, e.g. ``"GELU"``, ``"ReLU"``.
          Default: ``"GELU"``.
        - ``dropout`` (float): Dropout probability. Default: top-level ``dropout``.

    Args:
        task_config: Mapping of task name to task configuration dict.
        d_model: Input dimension (transformer output size).
        dropout: Default dropout probability for all heads.

    Returns:
        ``nn.ModuleDict`` mapping task names to ``MLP`` heads.
    """
    heads: dict[str, nn.Module] = {}
    for name, cfg in task_config.items():
        act_name = cfg.get("activation", "GELU")
        activation = getattr(nn, act_name)
        heads[name] = MLP(
            input_dim=d_model,
            output_dim=cfg["num_classes"],
            hidden_dims=cfg["hidden_dims"],
            activation=activation,
            dropout=cfg.get("dropout", dropout),
        )
    return nn.ModuleDict(heads)
