"""Base multi-task Lightning module with mask derivation.

Provides reusable optimizer setup, masked loss computation,
mask type registry, and multi-task train/val step logic.
Subclasses must implement ``forward()``.
"""

from typing import ClassVar

import pytorch_lightning as pl
import torch
import torch.nn as nn


class BaseSequenceLightning(pl.LightningModule):
    """Base class for multi-task sequence models.

    Provides reusable optimizer setup, masked loss computation,
    mask type registry, and multi-task train/val step logic.
    Subclasses must implement ``forward()``.

    Mask types:
        - ``"padding"``: 1 for real events (pos < seq_len), 0 for padding.
          Used for tasks where all real positions have valid targets.
        - ``"token"``: 1 for non-last real events (pos < seq_len - 1).
          Used for next-event prediction where the last position has no target.
        - ``"sequence"``: 1 only at the last real position.
          Used for sequence-level targets like churn.
    """

    MASK_TYPES: ClassVar[dict] = {
        "padding": lambda pos, sl: (pos < sl.unsqueeze(1)).long(),
        "token": lambda pos, sl: (pos < (sl - 1).unsqueeze(1)).long(),
        "sequence": lambda pos, sl: (pos == (sl - 1).unsqueeze(1)).long(),
    }
    TASK_TYPES: ClassVar[dict] = {
        "classification": lambda: nn.CrossEntropyLoss(reduction="none"),
        "regression": lambda: nn.MSELoss(reduction="none"),
    }

    def __init__(
        self,
        task_config: dict,
        max_len: int = 100,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        **kwargs,  # noqa: ARG002
    ):
        """Initialize the base multi-task sequence model.

        Args:
            task_config: Dict mapping task name to a config dict with keys:
                - ``"task_type"`` (str): ``"classification"`` or ``"regression"``.
                - ``"mask_type"`` (str): One of ``"padding"``, ``"token"``, ``"sequence"``.
                - ``"target_key"`` (str): Key in batch dict containing targets of shape ``(B, S)``.
                - ``"weight"`` (float, optional): Task loss weight. Defaults to ``1.0``.
            max_len: Maximum sequence length.
            learning_rate: Peak learning rate for AdamW.
            weight_decay: Weight decay for AdamW.
            warmup_steps: Number of linear warmup steps.
            **kwargs: Extra keyword arguments (ignored, for subclass compatibility).
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.task_config = task_config

        unknown = {cfg["mask_type"] for cfg in task_config.values()} - set(self.MASK_TYPES)
        if unknown:
            raise ValueError(f"Unknown mask types: {unknown}")

        unknown_types = {cfg["task_type"] for cfg in task_config.values()} - set(self.TASK_TYPES)
        if unknown_types:
            raise ValueError(f"Unknown task types: {unknown_types}")

        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))  # (1, max_len)

        self.task_weights = {}
        self.task_types = {}
        self.task_loss_fns = nn.ModuleDict()
        for task_name, cfg in task_config.items():
            self.task_weights[task_name] = cfg.get("weight", 1.0)
            self.task_types[task_name] = cfg["task_type"]
            self.task_loss_fns[task_name] = self.TASK_TYPES[cfg["task_type"]]()

    @torch.jit.unused
    def _derive_masks(self, sequence_length: torch.Tensor) -> dict[str, torch.Tensor]:
        """Derive masks for loss computation.

        Args:
            sequence_length: (B,) tensor of sequence lengths.

        Returns:
            Dict mapping mask type name to (B, max_len) mask tensor.
        """
        seq_lens = sequence_length.long().clamp(max=self.positions.size(1))
        needed = {cfg["mask_type"] for cfg in self.task_config.values()}
        return {mt: self.MASK_TYPES[mt](self.positions, seq_lens) for mt in needed}

    def _compute_attention_mask(self, sequence_length: torch.Tensor) -> torch.Tensor:
        """Derive attention padding mask (True = padding, for PyTorch convention).

        Args:
            sequence_length: (B,) tensor of sequence lengths.

        Returns:
            (B, max_len) bool tensor where True = position to ignore.
        """
        max_len = sequence_length.new_tensor(self.positions.size(1)).long()
        seq_lens = torch.min(sequence_length.long(), max_len)
        return self.positions >= seq_lens.unsqueeze(1)

    @torch.jit.unused
    def _compute_masked_task_loss(
        self,
        task_name: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked loss for a single task, averaged over valid positions.

        Args:
            task_name: Task name (selects loss function and type).
            logits: ``(B, S, C)`` for classification or ``(B, S, 1)`` for regression.
            targets: ``(B, S)`` target values.
            mask: ``(B, S)`` with 1 for valid positions, 0 for masked.
        """
        B, S = mask.shape
        task_type = self.task_types[task_name]

        if task_type == "classification":
            logits_flat = logits.reshape(B * S, -1)
            targets_flat = targets.reshape(B * S).long()
        else:
            logits_flat = logits.reshape(B * S, 1)
            targets_flat = targets.reshape(B * S, 1).float()

        mask_flat = mask.reshape(B * S).float()
        loss_per_pos = self.task_loss_fns[task_name](logits_flat, targets_flat)
        loss_per_pos = loss_per_pos.squeeze(-1)

        num_valid = mask_flat.sum()
        masked_loss = loss_per_pos * mask_flat
        return masked_loss.sum() / num_valid.clamp(min=1)

    @torch.jit.unused
    def _step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        outputs = self.forward(batch)
        masks = self._derive_masks(batch["derived_sequence_length"])
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_name, cfg in self.task_config.items():
            logits = outputs[task_name]
            targets = batch[cfg["target_key"]]
            mask = masks[cfg["mask_type"]]

            task_loss = self._compute_masked_task_loss(task_name, logits, targets, mask)
            weighted_loss = self.task_weights[task_name] * task_loss

            total_loss = total_loss + weighted_loss

            sync_dist = prefix == "val"
            self.log(f"{prefix}_{task_name}_loss", task_loss, on_step=False, on_epoch=True, sync_dist=sync_dist)

        sync_dist = prefix == "val"
        self.log(f"{prefix}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=sync_dist)
        return total_loss

    @torch.jit.unused
    def on_train_epoch_end(self) -> None:
        """Fail fast if the epoch loss is NaN — avoids wasting remaining epochs."""
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None and not torch.isfinite(train_loss):
            raise ValueError(
                f"NaN/Inf train_loss at end of epoch {self.current_epoch} — stopping training. "
                "Common causes: warmup_steps too small, learning_rate too high, NaN inputs."
            )

    @torch.jit.unused
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        return self._step(batch, "train")

    @torch.jit.unused
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        return self._step(batch, "val")

    @torch.jit.unused
    def configure_optimizers(self):
        """AdamW with linear warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def warmup_schedule(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
