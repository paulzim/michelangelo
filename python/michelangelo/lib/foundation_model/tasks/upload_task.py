"""Upload task: saves the trained model checkpoint to S3.

Runs as a Spark driver-only task with elevated disk to handle large
checkpoint files. Uses the OSS model manager's upload_assets utility
with S3 storage backend.
"""

import logging
import os
import uuid

from michelangelo.lib.model_manager._private.utils.asset_utils import upload_assets
from michelangelo.lib.model_manager.constants import StorageType
from michelangelo.uniflow.core.decorator import task
from michelangelo.uniflow.plugins.spark.task import SparkTask

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint.ckpt"


@task(config=SparkTask(driver_disk="100G", executor_instances=0))
def upload_task(config, train_result: dict) -> dict:
    """Upload trained model checkpoint to S3.

    Args:
        config: Upload configuration (unused; reserved for future auth overrides).
        train_result: Output dict from ``train_task``, must contain:
            - ``local_checkpoint_path``: local path to the checkpoint directory.
            - ``checkpoint_path``: destination path (local or ``s3://bucket/key``).

    Returns:
        Dict with ``status`` and ``checkpoint_path``.
    """
    local_path = train_result["local_checkpoint_path"]
    dest_path = train_result["checkpoint_path"]

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {local_path}. "
            "Ensure train_task completed successfully and the path is accessible."
        )

    source_type = StorageType.S3 if dest_path.startswith("s3://") else StorageType.LOCAL

    logger.info("Uploading checkpoint %s → %s (storage=%s)", local_path, dest_path, source_type)
    upload_assets(local_path, dest_path, source_type=source_type)
    logger.info("Checkpoint successfully uploaded to %s", dest_path)

    return {"status": "uploaded", "checkpoint_path": dest_path}
