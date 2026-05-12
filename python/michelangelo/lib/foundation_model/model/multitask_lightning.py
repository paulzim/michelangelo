"""Multitask Lightning module for the foundation model.

Composes MultiModalEncoder + DecoderOnlyTransformer + per-task heads
into a single PyTorch Lightning module for multi-task sequence training.
"""

from __future__ import annotations

import importlib
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from michelangelo.lib.model_manager.interface.custom_model import Model
from michelangelo.lib.sequence.lightning.base import BaseSequenceLightning
from michelangelo.lib.foundation_model.model.encoders import MultiModalEncoder
from michelangelo.lib.foundation_model.model.task_heads import build_task_heads
from michelangelo.lib.foundation_model.model.transformers.decoder_only import DecoderOnlyTransformer

logger = logging.getLogger(__name__)

# Default forward() output field order (alphabetical to match SDK schema sorting).
_DEFAULT_FORWARD_OUTPUT_FIELDS: tuple[str, ...] = (
    "pred_churn_logits",
    "pred_embedding",
    "pred_next_event_type_indexed_logits",
    "pred_time_to_next_event_bucket_logits",
)
_TASK_NAME_TO_PRED_OUTPUT: dict[str, str] = {
    "churn": "pred_churn_logits",
    "next_event_type": "pred_next_event_type_indexed_logits",
    "time_to_next_event": "pred_time_to_next_event_bucket_logits",
}
_PRED_OUTPUT_TO_TASK: dict[str, str] = {v: k for k, v in _TASK_NAME_TO_PRED_OUTPUT.items()}


class MultitaskSequenceLightning(BaseSequenceLightning):
    """Multi-task transformer lightning module.

    Composes:
    - ``MultiModalEncoder``: encodes multiple feature types into d_model
    - ``DecoderOnlyTransformer``: causal transformer backbone
    - Task heads: per-task classification heads with masked loss

    Mask derivation is handled by the base class via ``_derive_masks()``
    using the ``mask_type`` field in each task's config.

    Args:
        embedding_config: Feature encoder config for ``MultiModalEncoder``.
        architecture_config: Keys: ``d_model``, ``n_heads``, ``n_layers``,
            ``d_ff``, and optionally ``max_len``, ``dropout``, ``pos_encoding``.
        task_config: Per-task dict with ``num_classes``, ``hidden_dims``,
            ``task_type``, ``mask_type``, ``target_key``, and optionally
            ``weight``.
        eval_callback_fn: Optional dotted path to an evaluation function
            called at validation epoch end with ``(model=self, trainer=trainer)``.
        eval_callback_every_n_epochs: How often to run the eval callback.
        forward_output_fields: Ordered tuple of output field names. Must be
            alphabetically sorted and must include ``"pred_embedding"``.
    """

    def __init__(
        self,
        embedding_config: dict,
        architecture_config: dict,
        task_config: dict,
        eval_callback_fn: str | None = None,
        eval_callback_every_n_epochs: int = 1,
        forward_output_fields: list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ):
        d_model = architecture_config["d_model"]
        max_len = architecture_config.get("max_len", 100)
        dropout = architecture_config.get("dropout", 0.1)

        if forward_output_fields is None:
            forward_output_fields = kwargs.pop("forward_output_fields", None)
        else:
            kwargs.pop("forward_output_fields", None)

        order_src = forward_output_fields or list(_DEFAULT_FORWARD_OUTPUT_FIELDS)
        self._forward_output_fields = tuple(order_src)

        if "pred_embedding" not in self._forward_output_fields:
            raise ValueError("forward_output_fields must include 'pred_embedding'.")

        unknown = [
            n for n in self._forward_output_fields
            if n != "pred_embedding" and n not in _PRED_OUTPUT_TO_TASK
        ]
        if unknown:
            raise ValueError(f"forward_output_fields contains unknown names: {unknown}")

        expected_order = sorted(self._forward_output_fields)
        if list(self._forward_output_fields) != expected_order:
            raise ValueError(
                "forward_output_fields must be alphabetically sorted to match the "
                f"SDK assembler's output_schema sorting. Got: {list(self._forward_output_fields)}, "
                f"expected: {expected_order}"
            )

        super().__init__(task_config=task_config, max_len=max_len, **kwargs)

        self._eval_fn = None
        if eval_callback_fn:
            module_path, _, attr_name = eval_callback_fn.rpartition(".")
            self._eval_fn = getattr(importlib.import_module(module_path), attr_name)
        self._eval_every_n_epochs = eval_callback_every_n_epochs
        self._should_eval_this_epoch = False

        self.encoder = MultiModalEncoder(
            embedding_config=embedding_config,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            pos_encoding=architecture_config.get("pos_encoding", "sinusoidal"),
        )

        self.transformer = DecoderOnlyTransformer(
            d_model=d_model,
            n_heads=architecture_config["n_heads"],
            n_layers=architecture_config["n_layers"],
            d_ff=architecture_config["d_ff"],
            dropout=dropout,
        )

        self.task_heads = build_task_heads(task_config, d_model, dropout)

        unknown_tasks = set(task_config) - set(_TASK_NAME_TO_PRED_OUTPUT)
        if unknown_tasks:
            raise ValueError(
                f"task_config contains keys not in _TASK_NAME_TO_PRED_OUTPUT: {sorted(unknown_tasks)}. "
                "Update the mapping in multitask_lightning.py."
            )

    def _forward_output_names(self) -> list[str]:
        return [
            name
            for name in self._forward_output_fields
            if name == "pred_embedding"
            or (name in _PRED_OUTPUT_TO_TASK and _PRED_OUTPUT_TO_TASK[name] in self.task_config)
        ]

    def forward(self, batch: dict[str, torch.Tensor] | None = None, **kwargs) -> tuple[torch.Tensor, ...]:
        if batch is None:
            batch = kwargs

        encoded = self.encoder(batch)
        S = encoded.size(1)
        src_key_padding_mask = self._compute_attention_mask(batch["derived_sequence_length"].flatten())[:, :S]
        transformer_out = self.transformer(encoded, src_key_padding_mask=src_key_padding_mask)

        # Extract last non-padded token embedding per sequence
        seq_lengths = batch["derived_sequence_length"].flatten().long()
        S = transformer_out.size(1)
        capped = torch.minimum(seq_lengths, seq_lengths.new_tensor(S))
        last_positions = (capped - 1).clamp(min=0)
        last_positions_idx = last_positions.view(-1, 1, 1).expand(-1, 1, transformer_out.size(2))
        last_token_embedding = transformer_out.gather(dim=1, index=last_positions_idx).squeeze(1)

        forward_names = self._forward_output_names()
        tensors: list[torch.Tensor] = []
        for name in forward_names:
            if name == "pred_embedding":
                tensors.append(last_token_embedding)
            else:
                task_key = _PRED_OUTPUT_TO_TASK[name]
                tensors.append(self.task_heads[task_key](transformer_out))
        return tuple(tensors)

    @torch.jit.unused
    def _get_bad_keys(self, batch: dict[str, torch.Tensor]) -> list[str]:
        return [
            key for key, tensor in batch.items()
            if tensor.is_floating_point() and not torch.isfinite(tensor).all()
        ]

    @torch.jit.unused
    def _log_bad_batch(self, batch: dict[str, torch.Tensor], bad_keys: list[str], prefix: str) -> None:
        for key in bad_keys:
            t = batch[key].detach().float()
            total = t.numel()
            nan_count = int(torch.isnan(t).sum())
            inf_count = int(torch.isinf(t).sum())
            bad_pct = (nan_count + inf_count) / total * 100
            finite_vals = t[torch.isfinite(t)]
            finite_range = (
                f"finite_min={finite_vals.min():.4f}, finite_max={finite_vals.max():.4f}"
                if finite_vals.numel() > 0 else "no finite values"
            )
            logger.warning(
                f"[{prefix}_step] BAD FEATURE '{key}': nan={nan_count}, inf={inf_count}, "
                f"total={total} ({bad_pct:.2f}% bad) | {finite_range} | shape={list(t.shape)} — skipping batch."
            )
        self.log(f"{prefix}_bad_batch_count", 1.0, on_step=True, on_epoch=True, reduce_fx="sum")

    @torch.jit.unused
    def on_validation_epoch_start(self) -> None:
        self._should_eval_this_epoch = (
            self._eval_fn is not None
            and (
                self.trainer.current_epoch == 0
                or (self.trainer.current_epoch + 1) % self._eval_every_n_epochs == 0
                or self.trainer.current_epoch == self.trainer.max_epochs - 1
            )
        )
        if self._should_eval_this_epoch:
            self._val_batches: list[dict[str, torch.Tensor]] = []
            self._val_outputs: list[dict[str, torch.Tensor]] = []

    @torch.jit.unused
    def on_validation_epoch_end(self) -> None:
        if self._should_eval_this_epoch:
            self._eval_fn(model=self, trainer=self.trainer)
            self._val_batches = []
            self._val_outputs = []

    @torch.jit.unused
    def _step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        bad_keys = self._get_bad_keys(batch)
        if bad_keys:
            self._log_bad_batch(batch, bad_keys, prefix)
            trainable = [p for p in self.parameters() if p.requires_grad]
            return torch.stack([p.sum() * 0.0 for p in trainable]).sum()

        forward_outputs = self.forward(batch)
        names = self._forward_output_names()
        outputs = {
            _PRED_OUTPUT_TO_TASK[n]: forward_outputs[i]
            for i, n in enumerate(names)
            if n != "pred_embedding"
        }

        _BATCH_KEYS = ("derived_event_type_indexed", "derived_sequence_length", "response_next_event_type_indexed", "response_churned")
        _OUTPUT_KEYS = ("next_event_type", "churn")
        if prefix == "val" and self._should_eval_this_epoch:
            self._val_batches.append({k: v.detach().cpu() for k, v in batch.items() if k in _BATCH_KEYS})
            self._val_outputs.append({k: v.detach().cpu() for k, v in outputs.items() if k in _OUTPUT_KEYS})

        masks = self._derive_masks(batch["derived_sequence_length"])
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_name, cfg in self.task_config.items():
            logits = outputs[task_name]
            targets = batch[cfg["target_key"]]
            mask = masks[cfg["mask_type"]]

            if cfg["mask_type"] == "sequence":
                mask = mask * (targets >= 0).long()
                targets = targets.clamp(min=0)

            task_loss = self._compute_masked_task_loss(task_name, logits, targets, mask)
            weighted_loss = self.task_weights[task_name] * task_loss
            total_loss = total_loss + weighted_loss

            sync_dist = prefix == "val"
            self.log(f"{prefix}_{task_name}_loss", task_loss, on_step=True, on_epoch=True, sync_dist=sync_dist)

        sync_dist = prefix == "val"
        self.log(f"{prefix}_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=sync_dist)
        return total_loss


class FoundationModel(Model):
    """Wraps MultitaskSequenceLightning for Triton serving via the Model interface.

    save() / load() use torch state_dict so the package is pickle-free.
    predict() runs a batched forward pass and returns named numpy outputs.
    """

    _CKPT_FILENAME = "model.ckpt"
    _META_FILENAME = "meta.json"

    def __init__(self, lightning_model: MultitaskSequenceLightning, embedding_config, architecture_config, task_config, forward_output_fields):
        self._model = lightning_model
        self._embedding_config = embedding_config
        self._architecture_config = architecture_config
        self._task_config = task_config
        self._forward_output_fields = forward_output_fields

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(path, self._CKPT_FILENAME))
        meta = {
            "embedding_config": self._embedding_config,
            "architecture_config": self._architecture_config,
            "task_config": self._task_config,
            "forward_output_fields": self._forward_output_fields,
        }
        with open(os.path.join(path, self._META_FILENAME), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FoundationModel":
        with open(os.path.join(path, cls._META_FILENAME)) as f:
            meta = json.load(f)
        model = MultitaskSequenceLightning(
            embedding_config=meta["embedding_config"],
            architecture_config=meta["architecture_config"],
            task_config=meta["task_config"],
            forward_output_fields=meta["forward_output_fields"],
        )
        state_dict = torch.load(os.path.join(path, cls._CKPT_FILENAME), map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model, meta["embedding_config"], meta["architecture_config"], meta["task_config"], meta["forward_output_fields"])

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        batch = {k: torch.as_tensor(v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.forward(batch)
        names = self._model._forward_output_names()
        return {name: outputs[i].cpu().numpy() for i, name in enumerate(names)}
