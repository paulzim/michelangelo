"""Build-time import diagnostic — fails the docker build with the real traceback.

SparkTask and the push module are NOT tested here: both trigger SparkIO import
which needs a live JVM/SparkContext (not available in the build environment).
They are expected to work correctly in the actual Spark driver pod.
"""
import sys
import traceback

# These imports are pure-Python and must succeed without a JVM.
steps = [
    ("pusher schema", "from michelangelo.workflow.schema.pusher import PusherConfig, PusherPluginConfig, ModelPluginConfig, DatasetPluginConfig, EvalReportPluginConfig"),
    ("pusher pkg",    "from michelangelo.workflow.tasks.pusher import push"),
    ("variables",     "from michelangelo.workflow.variables import DatasetVariable, ModelMetadata, AssembledModel, ModelArtifact, PusherResult"),
    ("metadata",      "from michelangelo.workflow.variables.metadata import ModelMetadata"),
    ("types",         "from michelangelo.workflow.variables.types import AssembledModel, ModelArtifact, PusherResult"),
    ("model_mgr",     "from michelangelo.lib.model_manager.registry.client import InMemoryRegistryClient"),
    ("artifact_mgr",  "from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend"),
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
