# California Housing XGBoost — Live Demo Script

Audience: broader/onboarding (assume no prior UniFlow exposure).
Slot: 5-10 minutes.
Presenter runs everything on the Mac (`/Users/pzimme1/GitHub/michelangelo`), per the
hybrid dev workflow. MA Studio and a terminal should both be visible (split screen
or fast alt-tab).

The core risk in a 5-10 min slot is that the full run takes ~4 minutes
(`feature_prep` → `preprocess` → `train` → `push_step`, ~1 min apart per
RUNBOOK.md). That's most of the budget. The fix: do all the slow/fragile sandbox
work *before* the audience joins, and only run the parts that are worth watching
live.

---

## Pre-flight (30-60 min before the demo — not shown to the audience)

Run the full RUNBOOK.md sequence once, start to finish, as a dry run:

1. `ma sandbox start` (if needed) → confirm pods `Running`
2. Pre-run cleanup: delete zombie RayClusters + failed pods
3. Verify `ma-examples` namespace + Project CR exist
4. Pull latest branch changes if any
5. Build + import the pipeline image (skip if unchanged since last dry run)
6. Rebuild uniflowTar (skip if task configs unchanged)
7. Submit a **dry-run PipelineRun** and watch it to completion
8. Verify model registration: `poetry run ma model get --namespace ma-examples`
9. Open MA Studio (`localhost:8090/ma-examples`) and confirm the pipeline, the
   dry-run's PipelineRun, and its tables all render — this is your check that the
   Envoy `connect_grpc_bridge` filter hasn't regressed (RUNBOOK.md's 415 section)
10. **Delete the dry-run PipelineRun** so the live run has a clean slate:
    `kubectl delete pipelinerun california-housing-xgb-run -n ma-examples`
11. Re-run the zombie cleanup from step 2 (the dry run leaves its own RayClusters
    behind)
12. Leave the sandbox warm — don't `stop`/`delete` between now and the demo

If step 7-8 fails, fix it now using RUNBOOK.md's failure-mode table. **Do not
attempt a first-ever run live.**

### Fallback if the sandbox breaks between dry run and demo time

Keep a terminal scrollback or `script`-captured log of the successful dry run's
`kubectl logs ... | grep task_state` output, and keep MA Studio open on the
dry run's completed PipelineRun (before you delete it in step 10 — grab a
screenshot first). If the live submission fails in front of the audience, pivot to
"here's what a completed run looks like" using that captured output/screenshot
rather than debugging live.

---

## Live segment (5-10 min, audience present)

**0:00–0:45 — Frame it**
"Michelangelo pipelines are DAGs of plain Python functions. Each one declares
whether it runs on Ray or Spark, and the engine passes data between them by
reference, not by value — so a step handling gigabytes of data across the
network only ever forwards a storage URI." Point at the pipeline diagram:

```
feature_prep  →  preprocess  →  train  →  push_step
   (Ray)           (Spark)       (Ray)      (Spark)
```

**0:45–1:15 — Kick off the run**
```bash
kubectl apply -f examples/pipelines/california_housing_xgb/pipeline.yaml
kubectl apply -f examples/pipelines/california_housing_xgb/pipelinerun.yaml
```
Switch immediately to MA Studio's pipeline runs view so the audience sees it
enter `RUNNING`.

**1:15–3:15 — Narrate while it runs (this is the bulk of your live time)**
Tail logs in one pane, MA Studio DAG view in the other:
```bash
kubectl logs -n default deployment/michelangelo-worker --tail=50 -f | \
  grep -E "task_state|SUCCEEDED|FAILED"
```
Talking points, timed to whichever stage is active:
- *feature_prep (Ray)*: "This loads the California Housing dataset — 20,640
  rows, 8 features — bundled as a local CSV so nothing needs network access
  inside the cluster. It does the train/test split and returns two
  `DatasetVariable` handles."
- *preprocess (Spark)*: "Same handles, now loaded as a Spark DataFrame in a
  different task, on a different runtime, with no explicit data transfer code —
  the framework handled that."
- *train (Ray)*: "Distributed XGBoost training. In the sandbox we cap workers at
  0 extras (head-only) to fit an 8GB memory budget, but the same task definition
  scales out unchanged on a real cluster."
- *push_step (Spark)*: "One task pushes four artifacts in a single call: the
  model checkpoint, the eval report, and both train/validation datasets as
  Parquet — to the model registry and object storage."

**3:15–4:00 — Show the payoff**
Switch to MA Studio's tables view for the completed run (model, eval report,
datasets), or run:
```bash
poetry run ma model get --namespace ma-examples
```
to show the registered model row.

**4:00–5:00 (or up to 10:00 if time allows) — Wrap + pointers**
"The full pipeline, decorators, and `DatasetVariable` mechanics are documented
in the example's README; the operational playbook — sandbox recovery, common
failure modes — is in RUNBOOK.md in the same directory. Both are meant to be
followed standalone if you want to run this yourself."

---

## If something goes wrong live

- **Run doesn't progress past a stage after ~90s past its expected time**: don't
  debug live. Say "let's not wait on this" and cut to the pre-flight
  screenshot/log capture from the dry run.
- **MA Studio shows "Unable to fetch data" / blank tables**: known Envoy
  regression (RUNBOOK.md "MA Studio 415 errors"). Don't attempt the fix live —
  fall back to CLI output (`ma model get`, `kubectl logs`) for the rest of the
  demo.
- **PipelineRun vanishes right after apply**: Project CR or namespace missing —
  should have been caught in pre-flight step 3; if it wasn't, pivot to fallback.

The rule of thumb: the live segment is for narration and payoff, not
troubleshooting. Every failure mode above has a known fix in RUNBOOK.md — apply
it after the demo, not during.
