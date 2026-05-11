"""Training Ray task for the foundation model.

Loads vocabularies, builds MultitaskSequenceLightning, trains with
LightningTrainer on a Ray cluster, and returns the best checkpoint path.
"""

import json
import logging
import math
import os
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from ray.train import CheckpointConfig, RunConfig, ScalingConfig

from michelangelo.lib.sequence.collate.sequence_collate import SequenceCollateFn
from michelangelo.lib.trainer.torch.pytorch_lightning.lightning_trainer import (
    LightningTrainer,
    LightningTrainerParam,
    create_run_config,
    create_scaling_config,
)
from michelangelo.uniflow.core.decorator import task
from michelangelo.uniflow.plugins.ray.task import RayTask
from michelangelo.workflow.variables import DatasetVariable
from michelangelo.lib.foundation_model.models.multitask_lightning import MultitaskSequenceLightning

logger = logging.getLogger(__name__)


def sanitize_nan(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_vocabularies(vocab_path: str) -> tuple[dict, dict, dict]:
    """Load vocabulary parquet and return (vocabularies, vocab_sizes, embedding_config_dict)."""
    vocab_pdf = pd.read_parquet(vocab_path)
    vocabularies, vocab_sizes, embedding_config_dict = {}, {}, {}
    for _, row in vocab_pdf.iterrows():
        name = row["feature_name"]
        if row.get("feature_type", "categorical") == "categorical":
            vocabularies[name] = json.loads(row["vocab_json"])
            vocab_sizes[name] = row["vocab_size"]
        else:
            embedding_config_dict[name] = json.loads(row["vocab_json"])
    return vocabularies, vocab_sizes, embedding_config_dict


def build_embedding_config(vocab_sizes: dict, embedding_config_dict: dict) -> dict:
    """Build embedding_config dict for MultiModalEncoder from vocabulary data."""
    categoricals = [
        [name, vocab_size, embedding_config_dict.get(name, {}).get("embedding_dim") or max(8, vocab_size // 50)]
        for name, vocab_size in vocab_sizes.items()
    ]

    config: dict = {"categoricals": categoricals}

    if "numerical" in embedding_config_dict:
        num_cfg = embedding_config_dict["numerical"]
        config["numerical"] = [[
            "numerical_features",
            num_cfg["hidden_dim"],
            num_cfg["output_dim"],
            num_cfg.get("num_features"),
        ]]

    if "geo" in embedding_config_dict:
        geo_cfg = embedding_config_dict["geo"]
        config["geo"] = [[
            "geo_features",
            geo_cfg["hidden_dim"],
            geo_cfg["output_dim"],
            geo_cfg.get("num_features"),
        ]]

    return config


@task(config=RayTask())
def train_task(
    config,
    vocab_data: DatasetVariable,
    train_data: DatasetVariable,
    val_data: DatasetVariable,
) -> dict[str, Any]:
    """Ray task: load vocab, build model, run LightningTrainer, return checkpoint path."""
    logger.info("=" * 60)
    logger.info("FOUNDATION MODEL TRAINING")
    logger.info("=" * 60)

    vocabularies, vocab_sizes, embedding_config_dict = load_vocabularies(vocab_data.path)
    embedding_config = build_embedding_config(vocab_sizes, embedding_config_dict)

    # Build batch transform: stacks individual feature columns into the tensors the
    # MultiModalEncoder expects (numerical_features, geo_features).
    _num_names: list[str] = embedding_config_dict.get("numerical", {}).get("feature_names", [])
    _geo_names: list[str] = embedding_config_dict.get("geo", {}).get("feature_names", [])
    _max_len: int = config.transformer_config.max_len

    def _batch_transform(item: dict) -> dict:
        if _num_names:
            cols = [item.pop(n) for n in _num_names if n in item]
            if cols:
                item["numerical_features"] = torch.stack(cols, dim=-1).float()
        if _geo_names:
            cols = [item.pop(n) for n in _geo_names if n in item]
            if cols:
                item["geo_features"] = torch.stack(cols, dim=-1).float()
        # churn_label is stored as a scalar per earner; expand to (max_len,) so
        # it can be stacked into (B, max_len) by the DataLoader collate and
        # broadcast correctly against the sequence mask.
        if "churn_label" in item and item["churn_label"].dim() == 0:
            item["churn_label"] = item["churn_label"].expand(_max_len).clone()
        return item

    train_data.load_ray_dataset()
    val_data.load_ray_dataset()

    transformer_cfg = config.transformer_config
    task_cfg = config.task_config
    train_params = config.train_params
    save_cfg = config.save_model_config

    architecture_config = {
        "d_model": transformer_cfg.d_model,
        "n_heads": transformer_cfg.n_heads,
        "n_layers": transformer_cfg.n_layers,
        "d_ff": transformer_cfg.d_ff,
        "dropout": transformer_cfg.dropout,
        "max_len": transformer_cfg.max_len,
        "pos_encoding": transformer_cfg.pos_encoding,
    }

    collate_fn = SequenceCollateFn(vocab_key="_event_type_vocab_json")

    def create_model() -> pl.LightningModule:
        model = MultitaskSequenceLightning(
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            task_config=task_cfg,
            eval_callback_fn=config.eval_callback_fn,
            eval_callback_every_n_epochs=config.eval_callback_every_n_epochs,
            learning_rate=train_params.learning_rate,
            weight_decay=train_params.weight_decay,
            warmup_steps=train_params.warmup_steps,
        )
        # Attach collate_fn so callbacks can access the vocab
        model._collate_fn = collate_fn
        return model

    lightning_trainer_kwargs: dict[str, Any] = {
        "check_val_every_n_epoch": 1,
        "num_sanity_val_steps": 0,
        "precision": "bf16-mixed" if torch.cuda.is_available() else "32-true",
        "gradient_clip_val": train_params.gradient_clip,
    }

    if train_params.early_stopping_patience:
        lightning_trainer_kwargs["callbacks"] = [
            EarlyStopping(
                monitor="val_loss",
                patience=train_params.early_stopping_patience,
                mode="min",
                strict=False,
                check_on_train_epoch_end=False,
            )
        ]

    trainer_param = LightningTrainerParam(
        create_model=create_model,
        model_kwargs={},
        train_data=train_data.value,
        validation_data=val_data.value,
        batch_size=train_params.batch_size,
        num_epochs=train_params.num_epochs,
        batch_transform=_batch_transform,
        lightning_trainer_kwargs=lightning_trainer_kwargs,
    )

    run_config = create_run_config(
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
    use_gpu = torch.cuda.is_available()
    num_workers = 4 if use_gpu else 1
    scaling_config = create_scaling_config(cpu_per_worker=2, use_gpu=use_gpu, num_workers=num_workers)

    trainer = LightningTrainer(trainer_param)
    trained_res = trainer.train(run_config, scaling_config)

    checkpoint_path = f"{save_cfg.model_dir}/{save_cfg.project_name}/{save_cfg.experiment_name}"
    run_dir = os.environ.get("MA_PIPELINE_RUN_NAME", "")
    if run_dir:
        checkpoint_path = f"{checkpoint_path}/{run_dir}"

    return sanitize_nan({
        "status": "success",
        "training_output": trained_res,
        "checkpoint_path": checkpoint_path,
        "local_checkpoint_path": getattr(trained_res, "checkpoint_path", checkpoint_path),
    })
