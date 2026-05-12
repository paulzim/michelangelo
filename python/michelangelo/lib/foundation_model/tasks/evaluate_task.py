"""Post-training evaluation task for the earner foundation model.

Loads the trained checkpoint, runs inference on the validation dataset,
and computes binary classification metrics (AUC-ROC, AUC-PR, accuracy,
precision, recall, F1) for each task head.

Mirrors the evaluate_predictions_task from the production inference pipeline:
  uber/product/earner/earner_access_ml/earner_foundation/tasks/inference/evaluate/task.py
"""

import logging
from typing import Any

import numpy as np
import torch

from michelangelo.uniflow.core.decorator import task
from michelangelo.uniflow.plugins.ray.task import RayTask
from michelangelo.workflow.variables import DatasetVariable
from michelangelo.lib.foundation_model.model.multitask_lightning import MultitaskSequenceLightning, _PRED_OUTPUT_TO_TASK

logger = logging.getLogger(__name__)


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray, task_name: str) -> dict[str, Any]:
    """Compute classification metrics for one task head.

    Binary tasks (churn): AUC-ROC, AUC-PR, accuracy, precision, recall, F1.
    Multi-class tasks (next_event_type, time_to_next_event): accuracy, top-3 accuracy.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    if len(np.unique(y_true)) < 2:
        logger.warning("[eval] %s: only one class present, skipping", task_name)
        return {}

    num_classes = y_score.shape[-1] if y_score.ndim > 1 else 2
    y_pred = y_score.argmax(axis=-1) if y_score.ndim > 1 else (y_score >= 0.5).astype(int)

    metrics: dict[str, Any] = {
        "num_samples": int(len(y_true)),
        "positive_rate": float((y_true > 0).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if num_classes == 2:
        # Binary classification — full suite of metrics
        probs = y_score if y_score.ndim == 1 else y_score[:, 1]
        metrics["auc_roc"] = float(roc_auc_score(y_true, probs))
        metrics["auc_pr"] = float(average_precision_score(y_true, probs))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        # Multi-class — report accuracy and top-3 accuracy
        if y_score.ndim > 1:
            top3 = np.argsort(y_score, axis=-1)[:, -3:]
            top3_acc = float(np.mean([y_true[i] in top3[i] for i in range(len(y_true))]))
            metrics["top3_accuracy"] = top3_acc

    logger.info("=" * 60)
    logger.info("METRICS: %s", task_name.upper())
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    return metrics


@task(config=RayTask())
def evaluate_task(
    config: dict[str, Any],
    train_result: dict[str, Any],
    val_data: DatasetVariable,
) -> dict[str, Any]:
    """Evaluate trained model on the validation dataset.

    Loads the checkpoint produced by train_task, runs a forward pass over
    the val dataset, and computes per-task binary classification metrics.

    Args:
        config: Must contain ``model_config`` (embedding_config, architecture_config,
            task_config, forward_output_fields) matching the trained model.
        train_result: Output dict from train_task (contains checkpoint info).
        val_data: Validation dataset in post-native-transform schema.

    Returns:
        Dict mapping task name → metrics dict with auc_roc, auc_pr, accuracy,
        precision, recall, f1_score.
    """
    from ray.train import Checkpoint

    logger.info("=" * 60)
    logger.info("EARNER FOUNDATION MODEL EVALUATION")
    logger.info("=" * 60)

    checkpoint_path = train_result.get("local_checkpoint_path")

    # Load checkpoint — instantiate fresh then load weights to avoid
    # hyperparameter conflicts from save_hyperparameters() in the base class.
    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = Checkpoint(path=checkpoint_path)

    with checkpoint.as_directory() as ckpt_dir:
        import os
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])

        model = MultitaskSequenceLightning(
            embedding_config=config["embedding_config"],
            architecture_config=config["architecture_config"],
            task_config=config["task_config"],
            forward_output_fields=config["forward_output_fields"],
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state_dict)

    model.eval()

    val_data.load_ray_dataset()

    # Collect predictions and targets over entire val set
    task_predictions: dict[str, list[np.ndarray]] = {t: [] for t in config["task_config"]}
    task_targets: dict[str, list[np.ndarray]] = {t: [] for t in config["task_config"]}

    # Reuse __getitem__ conversion logic by iterating over batches directly
    for batch_np in val_data.value.iter_batches(batch_format="numpy", batch_size=256):
        # Convert numpy batch to tensors (mirrors RayTorchDataset.__getitem__)
        batch = {}
        for k, v in batch_np.items():
            if isinstance(v, np.ndarray):
                if v.dtype == object:
                    try:
                        v = np.stack(v.tolist())
                    except Exception:
                        continue
                if v.dtype.kind in ("U", "S"):
                    continue
                if v.dtype.kind == "f":
                    batch[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    batch[k] = torch.tensor(v, dtype=torch.long)
            elif isinstance(v, np.integer):
                batch[k] = torch.tensor(int(v), dtype=torch.long)
            elif isinstance(v, np.floating):
                batch[k] = torch.tensor(float(v), dtype=torch.float32)

        if "derived_sequence_length" not in batch:
            continue

        with torch.no_grad():
            forward_names = model._forward_output_names()
            outputs_tuple = model.forward(batch)
            outputs = {
                _PRED_OUTPUT_TO_TASK[n]: outputs_tuple[i]
                for i, n in enumerate(forward_names)
                if n != "pred_embedding"
            }

        for task_name, cfg in config["task_config"].items():
            logits = outputs.get(task_name)
            targets = batch.get(cfg["target_key"])
            if logits is None or targets is None:
                continue

            mask_type = cfg["mask_type"]
            num_classes = cfg["num_classes"]
            seq_lengths = batch["derived_sequence_length"].flatten().long()
            probs_full = torch.softmax(logits.float(), dim=-1)  # (B, S, C)

            if mask_type == "sequence":
                # Last real token per sequence
                last_pos = (seq_lengths - 1).clamp(min=0)
                idx = last_pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
                probs_seq = probs_full.gather(1, idx).squeeze(1)  # (B, C)
                if targets.dim() == 2:
                    tgt = targets[torch.arange(len(last_pos)), last_pos]
                else:
                    tgt = targets
                valid = tgt >= 0
                task_predictions[task_name].append(probs_seq[valid].cpu().numpy())
                task_targets[task_name].append(tgt[valid].cpu().numpy())

            elif mask_type in ("token", "padding"):
                # All real positions → flatten
                B, S = logits.shape[:2]
                pos = torch.arange(S, device=seq_lengths.device).unsqueeze(0)
                if mask_type == "token":
                    mask = pos < (seq_lengths - 1).unsqueeze(1)
                else:
                    mask = pos < seq_lengths.unsqueeze(1)
                # Keep full probs (B, S, C) and flatten valid positions
                valid = mask & (targets >= 0)  # (B, S)
                task_predictions[task_name].append(probs_full[valid].cpu().numpy())  # (N_valid, C)
                task_targets[task_name].append(targets[valid].cpu().numpy())

    results: dict[str, Any] = {}
    for task_name in config["task_config"]:
        preds = task_predictions[task_name]
        tgts = task_targets[task_name]
        if not preds:
            logger.warning("[eval] %s: no predictions collected", task_name)
            continue
        y_score = np.concatenate(preds)
        y_true = np.concatenate(tgts).astype(int)
        results[task_name] = _compute_metrics(y_true, y_score, task_name)

    logger.info("Evaluation complete.")
    return results
