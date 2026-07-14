"""Build-time import diagnostic — fails the docker build with the real traceback.

All of pitstop_advisor's tasks are pure-Python (RayTask, no Spark), so every
import below is safe to exercise at build time without a cluster.
"""

import sys
import traceback

steps = [
    ("xgboost", "import xgboost"),
    ("minio", "from minio import Minio"),
    (
        "model schema",
        "from michelangelo.lib.model_manager.schema import "
        "DataType, ModelSchema, ModelSchemaItem",
    ),
    (
        "packager",
        "from michelangelo.lib.model_manager.packager.custom_triton "
        "import CustomTritonPackager",
    ),
    (
        "registry",
        "from michelangelo.lib.model_manager.registry.api_client "
        "import APIRegistryClient",
    ),
    (
        "model module",
        "from examples.pipelines.pitstop_advisor.model import PitStopAdvisorModel",
    ),
    (
        "generate_data",
        "from examples.pipelines.pitstop_advisor.generate_data import generate_data",
    ),
    (
        "train module",
        "from examples.pipelines.pitstop_advisor.train import train",
    ),
    (
        "workflow",
        "from examples.pipelines.pitstop_advisor.pitstop_advisor import train_workflow",
    ),
]

ok = True
for name, stmt in steps:
    try:
        exec(stmt)
        print(f"OK: {name}", flush=True)
    except Exception:
        print(f"FAIL: {name}", flush=True)
        traceback.print_exc()
        ok = False
        break

sys.exit(0 if ok else 1)
