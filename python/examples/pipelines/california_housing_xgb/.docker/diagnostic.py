"""Build-time import diagnostic — fails the docker build with the real traceback."""
import sys
import traceback

steps = [
    ("SparkTask",     "from michelangelo.uniflow.plugins.spark import SparkTask"),
    ("pusher schema", "from michelangelo.workflow.schema.pusher import PusherConfig, PusherPluginConfig, ModelPluginConfig, DatasetPluginConfig, EvalReportPluginConfig"),
    ("pusher pkg",    "from michelangelo.workflow.tasks.pusher import push"),
    ("push module",   "import examples.pipelines.california_housing_xgb.push"),
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
