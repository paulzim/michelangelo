"""Minimal Michelangelo pipeline: generate random numbers, compute statistics.

Two-task Ray-only pipeline with no external data dependencies.
Good starting point for understanding the full pipeline creation workflow.
"""

import logging
from dataclasses import dataclass

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask

log = logging.getLogger(__name__)


@dataclass
class StatsResult:
    n: int
    mean: float
    std: float


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="2Gi",
        worker_instances=0,
    ),
)
def generate_data(n: int = 100, seed: int = 42) -> list:
    """Generate n random numbers using a fixed seed."""
    import random

    rng = random.Random(seed)
    data = [rng.gauss(0, 1) for _ in range(n)]
    log.info("Generated %d numbers, first 5: %s", n, data[:5])
    return data


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="2Gi",
        worker_instances=0,
    ),
)
def compute_stats(data: list) -> StatsResult:
    """Compute mean and std of a list of numbers."""
    import math

    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)
    result = StatsResult(n=n, mean=round(mean, 4), std=round(std, 4))
    log.info("Stats: %s", result)
    print("Pipeline result:", result)
    return result


@uniflow.workflow()
def hello_world_workflow(n: int = 100, seed: int = 42):
    """Generate random data and compute statistics.

    Args:
        n: Number of random samples to generate.
        seed: Random seed for reproducibility.
    """
    data = generate_data(n=n, seed=seed)
    stats = compute_stats(data=data)
    return stats


if __name__ == "__main__":
    ctx = uniflow.create_context()
    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"
    ctx.run(hello_world_workflow, n=100, seed=42)
