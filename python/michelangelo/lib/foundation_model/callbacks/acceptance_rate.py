"""Acceptance-rate evaluation callback for multi-task training.

Computes offer-acceptance PR-AUC and ROC-AUC on the validation set at
validation epoch end and logs scalar metrics + PR/ROC curve figures.

The function conditions on positions where the current event is
``offer_acknowledged`` and evaluates whether the model predicts
accepted vs rejected outcomes correctly.

Wire via ``eval_callback_fn`` in the model config::

    eval_callback_fn = "michelangelo.lib.foundation_model.callbacks.acceptance_rate.acceptance_rate_eval"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

mpl.use("Agg")
logger = logging.getLogger(__name__)

ACCEPTED_KEYWORDS = ("finalized", "accepted", "pickup", "destination", "delivery")
REJECTED_KEYWORDS = ("rejected", "ignored", "expired", "canceled")


def _resolve_event_indices(vocab: dict[int, str]) -> tuple[int | None, set[int], set[int]]:
    offer_ack_idx: int | None = None
    accepted_ids: set[int] = set()
    rejected_ids: set[int] = set()
    for idx, name in vocab.items():
        lower = name.lower()
        if "acknowledged" in lower and "offer" in lower:
            offer_ack_idx = idx
        if any(kw in lower for kw in ACCEPTED_KEYWORDS):
            accepted_ids.add(idx)
        if any(kw in lower for kw in REJECTED_KEYWORDS):
            rejected_ids.add(idx)
    return offer_ack_idx, accepted_ids, rejected_ids


def _extract_predictions(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    offer_ack_idx: int,
    accepted_ids: set[int],
    target_ids: set[int],
) -> tuple[list[int], list[float]]:
    event_logits = outputs.get("next_event_type")
    if event_logits is None:
        return [], []

    seq_events = batch.get("derived_event_type_indexed")
    seq_lengths = batch.get("derived_sequence_length")
    targets = batch.get("response_next_event_type_indexed")
    if seq_events is None or targets is None or seq_lengths is None:
        return [], []

    event_probs = torch.softmax(event_logits.float(), dim=-1)
    seq_lengths_flat = seq_lengths.flatten().long()
    num_classes = event_probs.size(-1)

    y_true: list[int] = []
    y_scores: list[float] = []
    for i in range(event_probs.size(0)):
        sl = seq_lengths_flat[i].item()
        events_i = seq_events[i, :sl].long()
        targets_i = targets[i, :sl].long()
        for pos in (events_i == offer_ack_idx).nonzero(as_tuple=False).flatten():
            p = pos.item()
            if p + 1 >= sl:
                continue
            next_event = targets_i[p].item()
            if next_event not in target_ids:
                continue
            probs_at_pos = event_probs[i, p]
            prob_accepted = sum(probs_at_pos[eid].item() for eid in accepted_ids if eid < num_classes)
            y_true.append(1 if next_event in accepted_ids else 0)
            y_scores.append(prob_accepted)
    return y_true, y_scores


def acceptance_rate_eval(*, model: Any, trainer: Any) -> None:
    """Evaluate offer-acceptance metrics across all validation batches."""
    collate_fn = getattr(model, "_collate_fn", None)
    vocab = getattr(collate_fn, "vocab", None) if collate_fn is not None else None
    if vocab is None:
        logger.info("[acceptance_rate_eval] No event_type vocab cached, skipping")
        return

    offer_ack_idx, accepted_ids, rejected_ids = _resolve_event_indices(vocab)
    if offer_ack_idx is None:
        logger.warning("[acceptance_rate_eval] offer_acknowledged not found in vocab")
        return

    target_ids = accepted_ids | rejected_ids
    all_y_true: list[int] = []
    all_y_scores: list[float] = []

    for batch, outputs in zip(getattr(model, "_val_batches", []), getattr(model, "_val_outputs", [])):
        yt, ys = _extract_predictions(batch, outputs, offer_ack_idx, accepted_ids, target_ids)
        all_y_true.extend(yt)
        all_y_scores.extend(ys)

    if len(all_y_true) < 10:
        logger.info("[acceptance_rate_eval] Only %d samples, skipping (need >= 10)", len(all_y_true))
        return

    y_true = np.array(all_y_true)
    y_scores = np.array(all_y_scores)
    if len(np.unique(y_true)) < 2:
        logger.warning("[acceptance_rate_eval] Only one class present, skipping")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc_val = auc(fpr, tpr)
    acceptance_rate = float(y_true.mean())

    logger.info(
        "[acceptance_rate_eval] epoch=%d  samples=%d  acceptance_rate=%.2f%%  ROC-AUC=%.4f  PR-AUC=%.4f",
        trainer.current_epoch, len(y_true), acceptance_rate * 100, roc_auc_val, pr_auc,
    )
    model.log("val_acceptance_roc_auc", roc_auc_val, sync_dist=True)
    model.log("val_acceptance_pr_auc", pr_auc, sync_dist=True)
    model.log("val_acceptance_rate", acceptance_rate, sync_dist=True)

    experiment = getattr(getattr(trainer, "logger", None), "experiment", None)
    if experiment is None:
        return

    fig = plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, linewidth=2, label=f"PR (AUC={pr_auc:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Offer Acceptance PR Curve — Epoch {trainer.current_epoch}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    experiment.log_figure(figure_name="acceptance_pr_curve", figure=fig, step=trainer.current_epoch)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Offer Acceptance ROC Curve — Epoch {trainer.current_epoch}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    experiment.log_figure(figure_name="acceptance_roc_curve", figure=fig, step=trainer.current_epoch)
    plt.close(fig)
