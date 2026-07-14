"""Pit Crew Advisor training workflow for the KubeCon "Pit Stop" demo.

Workflow entry point that orchestrates the full Pit Crew Advisor pipeline:
synthetic lap data generation, XGBoost training, Triton packaging, and model
registration. The individual task implementations live in sibling modules
(``generate_data``, ``train``). See README.md for the revision/deployment
apply sequence that turns the registered model into something LaneRun's
controller can actually query.
"""

from __future__ import annotations

import michelangelo.uniflow.core as uniflow
from examples.pipelines.pitstop_advisor.generate_data import (
    GeneratedData,
    generate_data,
)
from examples.pipelines.pitstop_advisor.train import TrainResult, train

__all__ = ["GeneratedData", "TrainResult", "generate_data", "train", "train_workflow"]


@uniflow.workflow()
def train_workflow(num_samples: int = 2000, seed: int = 0) -> TrainResult:
    """End-to-end workflow: generate synthetic data, train, package, register.

    Args:
        num_samples: Number of synthetic lap rows to generate.
        seed: RNG seed for reproducible synthetic data.

    Returns:
        TrainResult with the packaged model's storage URI and eval metrics.
    """
    data = generate_data(num_samples=num_samples, seed=seed)
    return train(data)


if __name__ == "__main__":
    ctx = uniflow.create_context()

    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"

    ctx.run(train_workflow)
