"""Standalone model test — no Spark, no Ray required.

Validates that MultitaskSequenceLightning trains correctly end-to-end using
synthetic tensors. Run this for rapid iteration on model architecture changes.

Usage::

    cd python/
    python examples/foundation_model/test_model.py

Takes ~10 seconds on CPU. Pass --gpu to use a GPU if available.
"""

import argparse
import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from michelangelo.lib.foundation_model.model.multitask_lightning import MultitaskSequenceLightning

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def make_synthetic_batch(
    batch_size: int = 8,
    seq_len: int = 20,
    num_event_types: int = 32,
    num_cities: int = 10,
    d_numerical: int = 3,
) -> dict[str, torch.Tensor]:
    """Return one batch dict matching the expected input schema."""
    seq_lengths = torch.randint(5, seq_len, (batch_size,))
    return {
        # Categorical features (B, S)
        "event_type_indexed": torch.randint(0, num_event_types, (batch_size, seq_len)),
        "city_id_indexed": torch.randint(0, num_cities, (batch_size, seq_len)),
        # Numerical features stacked (B, S, D)
        "numerical_features": torch.randn(batch_size, seq_len, d_numerical),
        # Geo features (B, S, 2)
        "geo_features": torch.randn(batch_size, seq_len, 2),
        # Required by BaseSequenceLightning
        "derived_sequence_length": seq_lengths,
        # Targets
        "next_event_type_target": torch.randint(0, num_event_types, (batch_size, seq_len)),
        "churn_label": torch.randint(-1, 2, (batch_size, seq_len)),
    }


def build_model(
    num_event_types: int = 32,
    num_cities: int = 10,
) -> MultitaskSequenceLightning:
    embedding_config = {
        "categoricals": [
            ["event_type_indexed", num_event_types, 16],
            ["city_id_indexed", num_cities, 8],
        ],
        "numerical": [
            ["numerical_features", 64, 32, 3],   # name, hidden, output, num_features
        ],
        "geo": [
            ["geo_features", 32, 16, 2],
        ],
    }

    architecture_config = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "max_len": 20,
        "pos_encoding": "sinusoidal",
    }

    task_config = {
        "next_event_type": {
            "task_type": "classification",
            "mask_type": "token",
            "target_key": "next_event_type_target",
            "num_classes": num_event_types,
            "hidden_dims": [32],
            "weight": 1.0,
        },
        "churn": {
            "task_type": "classification",
            "mask_type": "sequence",
            "target_key": "churn_label",
            "num_classes": 2,
            "hidden_dims": [16],
            "weight": 0.5,
        },
    }

    return MultitaskSequenceLightning(
        embedding_config=embedding_config,
        architecture_config=architecture_config,
        task_config=task_config,
        learning_rate=1e-3,
        warmup_steps=2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    logger.info("Building model...")
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", total_params)

    # Build a tiny synthetic dataset (16 batches)
    n_batches = 16
    batches = [make_synthetic_batch(args.batch_size) for _ in range(n_batches)]

    def collate(item):
        return item  # each "sample" is already a full batch dict

    # Wrap list of batch dicts as a dataset where each item is one full batch
    dataset = batches
    train_loader = DataLoader(dataset, batch_size=None, collate_fn=collate)
    val_loader = DataLoader(dataset[:4], batch_size=None, collate_fn=collate)

    accelerator = "gpu" if (args.gpu and torch.cuda.is_available()) else "cpu"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    logger.info("Starting training (accelerator=%s, epochs=%d)...", accelerator, args.epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("Training complete. Running a forward pass check...")
    model.eval()
    with torch.no_grad():
        batch = make_synthetic_batch(batch_size=2, seq_len=10)
        outputs = model.forward(batch)
        logger.info("Forward pass OK — %d output tensors", len(outputs))
        for i, t in enumerate(outputs):
            logger.info("  output[%d]: shape=%s", i, list(t.shape))

    logger.info("All checks passed.")


if __name__ == "__main__":
    main()
