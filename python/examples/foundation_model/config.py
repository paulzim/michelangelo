"""Concrete configuration for the foundation model training pipeline.

Edit this file to tune the pipeline for your data and compute environment.
All values are passed to the tasks via the workflow — nothing is hardcoded
in the library code.
"""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config dataclasses (no Uber deps — plain Pydantic)
# ---------------------------------------------------------------------------


class GeoFeaturesConfig(BaseModel):
    features: list[str]


class EmbeddingDimsConfig(BaseModel):
    numerical_hidden: int = 128
    numerical_output: int = 128
    geo_hidden: int = 64
    geo_output: int = 64


class FeaturePrepConfig(BaseModel):
    categorical_features: list[str]
    numerical_features: list[str]
    geo_features: dict[str, GeoFeaturesConfig]
    earner_column: str = "earner_uuid"
    timestamp_column: str = "event_timestamp"
    min_seq_length: int = 5
    max_seq_length: int = 100
    max_events_per_earner: int = 500
    special_tokens: dict[str, int] = {"PAD": 0, "UNK": 1, "MASK": 2}
    embedding_dims: EmbeddingDimsConfig = EmbeddingDimsConfig()
    train_split_ratio: float = 0.8


class TransformerConfig(BaseModel):
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 100
    pos_encoding: str = "sinusoidal"


class TrainParams(BaseModel):
    batch_size: int = 4096
    num_epochs: int = 30
    learning_rate: float = 1.5e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    early_stopping_patience: int = 5


class SaveModelConfig(BaseModel):
    model_dir: str
    project_name: str
    experiment_name: str


class TrainConfig(BaseModel):
    transformer_config: TransformerConfig
    task_config: dict
    train_params: TrainParams
    save_model_config: SaveModelConfig
    eval_callback_fn: str | None = None
    eval_callback_every_n_epochs: int = 1


# ---------------------------------------------------------------------------
# Concrete values — edit these for your environment
# ---------------------------------------------------------------------------

PREP_CONFIG = FeaturePrepConfig(
    categorical_features=[
        "event_type",
        "city_id",
        "vehicle_type_id",
    ],
    numerical_features=[
        "time_since_last_seconds",
        "event_number_norm",
        "time_to_next_seconds",
    ],
    geo_features={
        "pickup": GeoFeaturesConfig(features=["pickup_lat", "pickup_lng"]),
    },
    earner_column="earner_uuid",
    timestamp_column="event_timestamp",
    min_seq_length=5,
    max_seq_length=100,
    max_events_per_earner=500,
    special_tokens={"PAD": 0, "UNK": 1, "MASK": 2},
    embedding_dims=EmbeddingDimsConfig(
        numerical_hidden=128,
        numerical_output=128,
        geo_hidden=64,
        geo_output=64,
    ),
    train_split_ratio=0.8,
)

TRAIN_CONFIG = TrainConfig(
    transformer_config=TransformerConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=100,
        pos_encoding="sinusoidal",
    ),
    # task_config keys must match _TASK_NAME_TO_PRED_OUTPUT in multitask_lightning.py
    task_config={
        "next_event_type": {
            "task_type": "classification",
            "mask_type": "token",
            "target_key": "next_event_type_target",
            "num_classes": 64,      # updated automatically from vocab size at runtime
            "hidden_dims": [256],
            "weight": 1.0,
        },
        "churn": {
            "task_type": "classification",
            "mask_type": "sequence",
            "target_key": "churn_label",
            "num_classes": 2,
            "hidden_dims": [64],
            "weight": 0.5,
        },
    },
    train_params=TrainParams(
        batch_size=4096,
        num_epochs=30,
        learning_rate=1.5e-4,
        warmup_steps=1000,
        weight_decay=0.01,
        gradient_clip=1.0,
        early_stopping_patience=5,
    ),
    save_model_config=SaveModelConfig(
        # Change to your S3 bucket or local path
        model_dir="s3://your-bucket/models",
        project_name="foundation_model",
        experiment_name="v1",
    ),
    eval_callback_fn="michelangelo.lib.foundation_model.callbacks.churn_auc.churn_eval",
    eval_callback_every_n_epochs=1,
)
