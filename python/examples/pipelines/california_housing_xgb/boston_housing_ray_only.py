"""XGBoost regression workflow for Boston Housing — Ray only, no Spark.

Simplified version of the boston_housing_xgb example that uses only Ray tasks,
avoiding the Java/PySpark requirement. Good for local first-run testing.
"""

import logging

import numpy as np
import ray
import ray.data
from ray.train import RunConfig, ScalingConfig
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.variables import DatasetVariable

log = logging.getLogger(__name__)


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="4Gi",
        worker_cpu=1,
        worker_memory="4Gi",
        worker_instances=0,
    ),
    cache_enabled=True,
)
def feature_prep(
    columns: list[str],
    test_size: float = 0.25,
    seed: int = 1,
) -> tuple[DatasetVariable, DatasetVariable]:
    """Generate synthetic housing data and split into train/validation sets."""
    rng = np.random.default_rng(seed)
    n_samples = 506
    feature_names = columns[:-1]
    X = rng.random((n_samples, len(feature_names)))  # noqa: N806
    y = rng.random(n_samples) * 50  # synthetic target in [0, 50]
    dataset = [
        dict(zip(feature_names, features), target=target)
        for features, target in zip(X, y)
    ]
    data = ray.data.from_items(dataset).select_columns(columns)
    train_data, validation_data = data.train_test_split(
        test_size=test_size, shuffle=True, seed=seed
    )

    train_dv = DatasetVariable.create(train_data)
    train_dv.save_ray_dataset()
    validation_dv = DatasetVariable.create(validation_data)
    validation_dv.save_ray_dataset()

    return train_dv, validation_dv


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="4Gi",
        worker_cpu=1,
        worker_memory="4Gi",
        worker_instances=0,
    ),
)
def train(
    train_dv: DatasetVariable,
    validation_dv: DatasetVariable,
    params: dict,
) -> str:
    """Train XGBoost model using Ray."""
    import xgboost

    train_dv.load_ray_dataset()
    train_data: ray.data.Dataset = train_dv.value

    validation_dv.load_ray_dataset()
    validation_data: ray.data.Dataset = validation_dv.value

    label_column = train_data.schema().names[-1]

    scaling_config = ScalingConfig(num_workers=1, use_gpu=False)
    run_config = RunConfig(storage_path="/tmp/ray_results")

    def train_loop_per_worker():
        train_shard = ray.train.get_dataset_shard("train")
        validation_shard = ray.train.get_dataset_shard("validation")

        train_df = train_shard.materialize().to_pandas()
        validation_df = validation_shard.materialize().to_pandas()

        feature_cols = [c for c in train_df.columns if c != label_column]
        dtrain = xgboost.DMatrix(train_df[feature_cols], label=train_df[label_column])
        dval = xgboost.DMatrix(
            validation_df[feature_cols], label=validation_df[label_column]
        )

        xgboost.train(
            params,
            dtrain=dtrain,
            evals=[(dval, "validation")],
            num_boost_round=10,
            callbacks=[RayTrainReportCallback()],
        )

    trainer = XGBoostTrainer(
        train_loop_per_worker,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_data, "validation": validation_data},
    )
    result = trainer.fit()
    if result.error:
        raise result.error

    log.info("Training complete. Model path: %s", result.path)
    return result.path


@uniflow.workflow()
def train_workflow(
    dataset_cols: str = "CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,target",
):
    """feature_prep → train, Ray only."""
    columns = dataset_cols.split(",")
    train_dv, validation_dv = feature_prep(columns=columns)
    result_path = train(
        train_dv=train_dv,
        validation_dv=validation_dv,
        params={
            "objective": "reg:linear",
            "max_depth": 5,
            "learning_rate": 0.1,
            "colsample_bytree": 0.3,
            "alpha": 10,
        },
    )
    print("Model saved at:", result_path)
    return result_path


if __name__ == "__main__":
    ctx = uniflow.create_context()
    ctx.run(
        train_workflow,
        dataset_cols="CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,target",
    )
