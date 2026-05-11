"""Churn prediction evaluation callback for multi-task training.

Computes churn PR-AUC and ROC-AUC on the validation set at epoch end
and logs scalar metrics + PR/ROC curve figures.

Wire via ``eval_callback_fn`` in the model config::

    eval_callback_fn = "michelangelo.lib.foundation_model.callbacks.churn_auc.churn_eval"
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

mpl.use("Agg")
logger = logging.getLogger(__name__)


def _extract_churn_predictions(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
) -> tuple[list[int], list[float]]:
    logits = outputs.get("churn")
    targets = batch.get("response_churned")
    if logits is None or targets is None:
        return [], []

    if logits.dim() == 3:
        seq_lengths = batch.get("derived_sequence_length")
        if seq_lengths is not None:
            last_pos = (seq_lengths.flatten().long() - 1).clamp(min=0)
            idx = last_pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            logits = logits.gather(1, idx).squeeze(1)
        else:
            logits = logits[:, -1, :]

    probs = torch.softmax(logits.float(), dim=-1)
    if targets.dim() == 2:
        targets = targets[:, 0]
    targets_flat = targets.flatten().long()
    valid = targets_flat >= 0
    if not valid.any():
        return [], []
    return targets_flat[valid].tolist(), probs[valid, 1].tolist()


def churn_eval(*, model: Any, trainer: Any) -> None:
    """Evaluate churn prediction metrics across all validation batches."""
    all_y_true: list[int] = []
    all_y_scores: list[float] = []
    for batch, outputs in zip(getattr(model, "_val_batches", []), getattr(model, "_val_outputs", [])):
        yt, ys = _extract_churn_predictions(batch, outputs)
        all_y_true.extend(yt)
        all_y_scores.extend(ys)

    if len(all_y_true) < 10:
        logger.info("[churn_eval] Only %d earners with churn labels, skipping (need >= 10)", len(all_y_true))
        return

    y_true = np.array(all_y_true)
    y_scores = np.array(all_y_scores)
    if len(np.unique(y_true)) < 2:
        logger.warning("[churn_eval] Only one class present, skipping")
        return

    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc_val = roc_auc_score(y_true, y_scores)
    churn_rate = float(y_true.mean())

    logger.info(
        "[churn_eval] epoch=%d  earners=%d  churn_rate=%.2f%%  ROC-AUC=%.4f  PR-AUC=%.4f",
        trainer.current_epoch, len(y_true), churn_rate * 100, roc_auc_val, pr_auc,
    )
    model.log("val_churn_roc_auc", roc_auc_val, sync_dist=True)
    model.log("val_churn_pr_auc", pr_auc, sync_dist=True)
    model.log("val_churn_rate", churn_rate, sync_dist=True)

    experiment = getattr(getattr(trainer, "logger", None), "experiment", None)
    if experiment is None:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, linewidth=2, label=f"PR (AUC={pr_auc:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Churn Prediction PR Curve — Epoch {trainer.current_epoch}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    experiment.log_figure(figure_name="churn_pr_curve", figure=fig, step=trainer.current_epoch)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Churn Prediction ROC Curve — Epoch {trainer.current_epoch}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    experiment.log_figure(figure_name="churn_roc_curve", figure=fig, step=trainer.current_epoch)
    plt.close(fig)
