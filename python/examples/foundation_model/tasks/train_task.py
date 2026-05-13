"""Training Ray task for the earner foundation model.

Trains MultitaskSequenceLightning with LightningTrainer on a Ray cluster.
Input datasets are expected to already be in the post-native-transform schema
(derived_numerical_stacked, derived_geo_stacked, etc.) as produced by the
tabular_native_transform step in the production pipeline.
"""

import logging
import math
import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from ray.train import CheckpointConfig

from michelangelo.lib.trainer.torch.pytorch_lightning.lightning_trainer import (
    LightningTrainer,
    LightningTrainerParam,
    create_run_config,
    create_scaling_config,
)
from michelangelo.uniflow.core.decorator import task
from michelangelo.uniflow.plugins.ray.task import RayTask
from michelangelo.workflow.variables import DatasetVariable
from michelangelo.lib.foundation_model.model.multitask_lightning import MultitaskSequenceLightning

logger = logging.getLogger(__name__)


def sanitize_nan(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@task(config=RayTask())
def train_task(
    config,
    train_data: DatasetVariable,
    val_data: DatasetVariable,
) -> dict[str, Any]:
    """Ray task: train MultitaskSequenceLightning and return checkpoint path.

    Expects train_data and val_data to already be in the post-native-transform
    schema with columns matching the embedding_config feature names.
    """
    logger.info("=" * 60)
    logger.info("EARNER FOUNDATION MODEL TRAINING")
    logger.info("=" * 60)

    train_data.load_ray_dataset()
    val_data.load_ray_dataset()

    train_params = config.train_params
    save_cfg = config.save_model_config

    def create_model() -> pl.LightningModule:
        return MultitaskSequenceLightning(
            embedding_config=config.embedding_config,
            architecture_config=config.architecture_config,
            task_config=config.task_config,
            forward_output_fields=config.forward_output_fields,
            eval_callback_fn=config.eval_callback_fn,
            eval_callback_every_n_epochs=config.eval_callback_every_n_epochs,
            learning_rate=train_params.learning_rate,
            weight_decay=train_params.weight_decay,
            warmup_steps=train_params.warmup_steps,
        )

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
        lightning_trainer_kwargs=lightning_trainer_kwargs,
    )

    run_config = create_run_config(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
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

    # Resolve the actual local checkpoint path from the Ray result.
    local_checkpoint_path = (
        trained_res.checkpoint.path
        if trained_res.checkpoint is not None
        else checkpoint_path
    )

    return sanitize_nan({
        "status": "success",
        "training_output": trained_res,
        "checkpoint_path": checkpoint_path,
        "local_checkpoint_path": local_checkpoint_path,
    })
