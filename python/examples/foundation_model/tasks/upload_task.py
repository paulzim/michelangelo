"""Upload task: packages and saves the trained model for Triton serving.

Loads the trained checkpoint, wraps it in FoundationModel (Model interface),
builds a ModelSchema from the trainer config, and uses CustomTritonPackager
to produce a deployable Triton model package — matching the tabular_assembler
+ pusher steps from the production pipeline.
"""

import logging
import os

import torch

from michelangelo.lib.model_manager.constants import StorageType
from michelangelo.lib.model_manager.packager.custom_triton import CustomTritonPackager
from michelangelo.lib.model_manager.schema import ModelSchema
from michelangelo.lib.model_manager.schema.data_type import DataType
from michelangelo.lib.model_manager.schema.model_schema_item import ModelSchemaItem
from michelangelo.lib.model_manager._private.utils.asset_utils import upload_assets
from michelangelo.uniflow.core.decorator import task
from michelangelo.uniflow.plugins.ray.task import RayTask
from michelangelo.lib.foundation_model.model.multitask_lightning import FoundationModel, MultitaskSequenceLightning

logger = logging.getLogger(__name__)


def _build_model_schema(train_config) -> ModelSchema:
    """Build ModelSchema from TRAIN_CONFIG matching pipeline_conf.yaml input/output columns."""
    max_len = train_config.architecture_config["max_len"]

    input_items = []

    for name, _bucket, _dim in train_config.embedding_config.get("hash_categoricals", []):
        input_items.append(ModelSchemaItem(name=name, data_type=DataType.LONG, shape=[max_len]))

    for name, _vocab, _dim in train_config.embedding_config.get("categoricals", []):
        input_items.append(ModelSchemaItem(name=name, data_type=DataType.LONG, shape=[max_len]))

    for name, _hidden, _out, num_features in train_config.embedding_config.get("numerical", []):
        input_items.append(ModelSchemaItem(name=name, data_type=DataType.FLOAT, shape=[max_len, num_features]))

    for name, _hidden, _out, num_features in train_config.embedding_config.get("geo", []):
        input_items.append(ModelSchemaItem(name=name, data_type=DataType.FLOAT, shape=[max_len, num_features]))

    input_items.append(ModelSchemaItem(name="derived_sequence_length", data_type=DataType.LONG, shape=[1]))

    output_shape_map = {
        "pred_churn_logits": [max_len, train_config.task_config["churn"]["num_classes"]],
        "pred_embedding": [train_config.architecture_config["d_model"]],
        "pred_next_event_type_indexed_logits": [max_len, train_config.task_config["next_event_type"]["num_classes"]],
        "pred_time_to_next_event_bucket_logits": [max_len, train_config.task_config["time_to_next_event"]["num_classes"]],
    }
    output_items = [
        ModelSchemaItem(name=name, data_type=DataType.FLOAT, shape=output_shape_map.get(name, [-1]))
        for name in train_config.forward_output_fields
    ]

    return ModelSchema(input_schema=input_items, output_schema=output_items)


@task(config=RayTask())
def upload_task(config, train_result: dict) -> dict:
    """Package the trained model with CustomTritonPackager and upload.

    1. Loads checkpoint from train_result["local_checkpoint_path"].
    2. Wraps it in FoundationModel (implements the Model interface).
    3. Packages it into a Triton-compatible bundle via CustomTritonPackager.
    4. Uploads the package to train_result["checkpoint_path"].

    Args:
        config: Must contain ``train_config`` (the TrainConfig object).
        train_result: Output dict from train_task.

    Returns:
        Dict with ``status`` and ``checkpoint_path``.
    """
    import tempfile
    from ray.train import Checkpoint

    train_config = config["train_config"]
    local_checkpoint_path = train_result["local_checkpoint_path"]
    dest_path = train_result["checkpoint_path"]

    logger.info("=" * 60)
    logger.info("FOUNDATION MODEL PACKAGING")
    logger.info("=" * 60)

    logger.info("Loading checkpoint from %s", local_checkpoint_path)
    checkpoint = Checkpoint(path=local_checkpoint_path)

    with checkpoint.as_directory() as ckpt_dir:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])

        lightning_model = MultitaskSequenceLightning(
            embedding_config=train_config.embedding_config,
            architecture_config=train_config.architecture_config,
            task_config=train_config.task_config,
            forward_output_fields=train_config.forward_output_fields,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        lightning_model.load_state_dict(state_dict)
        lightning_model.eval()

    foundation_model = FoundationModel(
        lightning_model,
        embedding_config=train_config.embedding_config,
        architecture_config=train_config.architecture_config,
        task_config=train_config.task_config,
        forward_output_fields=train_config.forward_output_fields,
    )

    staging_dir = tempfile.mkdtemp(prefix="efm_upload_")
    model_artifacts_dir = os.path.join(staging_dir, "model_artifacts")
    foundation_model.save(model_artifacts_dir)

    model_schema = _build_model_schema(train_config)
    logger.info("Built ModelSchema: %d inputs, %d outputs",
                len(model_schema.input_schema), len(model_schema.output_schema))

    packager = CustomTritonPackager(custom_batch_processing=True)
    package_dir = os.path.join(staging_dir, "triton_package")
    packager.create_model_package(
        model_path=model_artifacts_dir,
        model_class="michelangelo.lib.foundation_model.model.multitask_lightning.FoundationModel",
        model_schema=model_schema,
        model_name="earner-foundation-model",
        dest_model_path=package_dir,
        model_path_source_type=StorageType.LOCAL,
        include_import_prefixes=["michelangelo"],
    )
    logger.info("Triton package created at %s", package_dir)

    storage_type = StorageType.S3 if dest_path.startswith("s3://") else StorageType.LOCAL
    upload_assets(package_dir, dest_path, source_type=storage_type)
    logger.info("Model package uploaded to %s", dest_path)

    return {"status": "uploaded", "checkpoint_path": dest_path}
