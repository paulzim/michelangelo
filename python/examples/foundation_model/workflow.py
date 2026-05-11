"""Foundation model training workflow.

Wires the three tasks into a uniflow DAG:
  1. feature_prep_task  — Spark: vocab + encode + sequences + train/val split
  2. train_task         — Ray:   train MultitaskSequenceLightning
  3. upload_task        — Spark: upload checkpoint to S3

## Running locally (full Spark + Ray on your machine)

    cd python/
    python examples/foundation_model/workflow.py local-run

This uses uniflow's local-run mode: SparkTask starts a local Spark session,
RayTask starts a single-node Ray cluster. Requires PySpark and Ray installed.

The workflow generates synthetic event data automatically in local-run mode
(see ``__main__`` block below) so you don't need a real dataset to start.

## Running on a cluster

    cd python/
    python examples/foundation_model/workflow.py remote-run \\
        --storage-url s3://your-bucket/uf_storage \\
        --image your-registry/foundation-model:latest

Or via mactl::

    mactl submit examples/foundation_model/workflow.py \\
        --storage-url s3://your-bucket/uf_storage

## Testing the model without Spark/Ray

    python examples/foundation_model/test_model.py

Runs MultitaskSequenceLightning on synthetic tensors in a single process —
no Spark, no Ray, no dataset needed. Useful for rapid iteration on model code.
"""

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.core.decorator import workflow
from michelangelo.workflow.variables import DatasetVariable
from michelangelo.lib.foundation_model.tasks.feature_prep_task import feature_prep_task
from michelangelo.lib.foundation_model.tasks.train_task import train_task
from michelangelo.lib.foundation_model.tasks.upload_task import upload_task

from examples.foundation_model.config import PREP_CONFIG, TRAIN_CONFIG


@workflow()
def train_workflow(event_data: DatasetVariable):
    """End-to-end foundation model training pipeline.

    Args:
        event_data: Raw event-level dataset (one row per earner event).
            Must contain at minimum: ``earner_uuid``, ``event_type``,
            ``event_timestamp``, and optionally ``churned``.
    """
    # Step 1: Feature preparation (Spark)
    prep_result = feature_prep_task(
        config=PREP_CONFIG,
        event_data=event_data,
    )

    # Step 2: Model training (Ray)
    trained = train_task(
        config=TRAIN_CONFIG,
        vocab_data=prep_result["vocabularies"],
        train_data=prep_result["train_dataset"],
        val_data=prep_result["val_dataset"],
    )

    # Step 3: Checkpoint upload (Spark driver)
    upload_task(config={}, train_result=trained)


def _make_synthetic_event_data() -> DatasetVariable:
    """Create a tiny synthetic event DataFrame for local-run testing."""
    from datetime import datetime, timedelta

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.master("local[*]").appName("foundation_model_local").getOrCreate()

    import random
    random.seed(42)

    earners = [f"earner_{i:04d}" for i in range(50)]
    event_types = ["supply_online", "offer_assigned", "offer_acknowledged", "job_to_pickup",
                   "job_to_dropoff", "supply_offline"]
    rows = []
    base_ts = datetime(2024, 1, 1)
    for earner in earners:
        n_events = random.randint(10, 30)
        for j in range(n_events):
            rows.append({
                "earner_uuid": earner,
                "event_type": random.choice(event_types),
                "event_timestamp": base_ts + timedelta(hours=j * random.uniform(0.5, 6)),
                "city_id": str(random.randint(1, 10)),
                "vehicle_type_id": str(random.randint(1, 3)),
                "time_since_last_seconds": float(j * 1800),
                "event_number_norm": j / max(n_events - 1, 1),
                "time_to_next_seconds": float(random.randint(600, 7200)),
                "pickup_lat": 37.7749 + random.gauss(0, 0.05),
                "pickup_lng": -122.4194 + random.gauss(0, 0.05),
                "churned": random.choice([0, 1, -1]),
            })

    df = spark.createDataFrame(rows)
    dv = DatasetVariable.create(df)
    dv.save_spark_dataframe()
    return dv


if __name__ == "__main__":
    import os

    import tempfile

    ctx = uniflow.create_context()

    ctx.environ["MA_NAMESPACE"] = "local"
    ctx.environ["S3_ALLOW_BUCKET_CREATION"] = "True"

    _local_storage = os.path.join(tempfile.gettempdir(), "uf_storage_foundation_model")
    os.environ["UF_STORAGE_URL"] = f"file://{_local_storage}"
    os.makedirs(_local_storage, exist_ok=True)

    if ctx.is_local_run():
        # Override save path to local for local-run
        TRAIN_CONFIG.save_model_config.model_dir = os.path.expanduser("~/foundation_model_checkpoints")
        event_data = _make_synthetic_event_data()
    else:
        # For remote-run, expect event_data to be injected via workflow args
        # or modify this block to load from your data source
        raise RuntimeError(
            "Remote-run requires event_data. Pass it as a workflow argument or "
            "modify this __main__ block to load from your data source."
        )

    ctx.run(train_workflow, event_data=event_data)
