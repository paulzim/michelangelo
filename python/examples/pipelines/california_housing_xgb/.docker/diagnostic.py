"""Build-time import diagnostic — fails the docker build with the real traceback.

push_step previously used SparkTask (needs JVM). It now uses RayTask, so all
imports in push.py are pure-Python and safe to test here.
"""
import sys
import traceback

steps = [
    ("pusher schema", "from michelangelo.workflow.schema.pusher import PusherConfig, PusherPluginConfig, ModelPluginConfig, DatasetPluginConfig, EvalReportPluginConfig"),
    ("pusher pkg",    "from michelangelo.workflow.tasks.pusher import push"),
    ("variables",     "from michelangelo.workflow.variables import DatasetVariable, ModelMetadata, AssembledModel, ModelArtifact, PusherResult"),
    ("metadata",      "from michelangelo.workflow.variables.metadata import ModelMetadata"),
    ("types",         "from michelangelo.workflow.variables.types import AssembledModel, ModelArtifact, PusherResult"),
    ("model_mgr",     "from michelangelo.lib.model_manager.registry.client import InMemoryRegistryClient"),
    ("artifact_mgr",  "from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend"),
    ("sinks schema",  "from michelangelo.workflow.schema.sinks.s3 import S3SinkConfig; from michelangelo.workflow.schema.sinks.local import LocalFileSinkConfig"),
    ("sinks funcs",   "from michelangelo.workflow.tasks.functions.sinks import S3Sink, LocalFileSink"),
    ("push module",   "from examples.pipelines.california_housing_xgb.push import push_step"),
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
